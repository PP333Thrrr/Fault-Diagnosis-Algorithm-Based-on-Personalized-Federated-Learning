import os
import sys
import json
import time
import uuid
import gc
import threading
import queue
import numpy as np
import psutil
import torch
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

# 历史训练记录存储文件
HISTORY_FILE = 'training_history.json'
# 最大历史记录数
MAX_HISTORY = 20

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datautil.prepare_data import get_data
from util.config import img_param_init, normalize_dataset_name, set_random_seed
from util.traineval import TrainingCancelled
from alg import algs
from alg.algs import ALGORITHMS

app = Flask(__name__)
CORS(app)  # 启用CORS支持

# 加载历史训练记录
def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f)
                for record in history:
                    memory_used = record.get('memory_used')
                    if isinstance(memory_used, (int, float)) and memory_used < 0:
                        record['memory_used'] = 0
                return history
        except:
            return []
    return []

# 保存历史训练记录
def save_history(history):
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# 添加新的训练记录
def add_history_record(record):
    history = load_history()
    # 添加新记录到开头
    history.insert(0, record)
    # 限制历史记录数量
    if len(history) > MAX_HISTORY:
        history = history[:MAX_HISTORY]
    save_history(history)
    return history

# 轻量级训练任务管理：job_id -> {queue, cancel_event, thread, running, ...}
training_jobs = {}
training_jobs_lock = threading.Lock()
SSE_QUEUE_TIMEOUT = 1.0
FINISHED_JOB_TTL_SECONDS = 300
PROGRESS_MIN_INTERVAL_SECONDS = 0.3
PROGRESS_MIN_DELTA_PERCENT = 1


def get_memory_mb(process):
    return process.memory_info().rss / 1024 / 1024


def cleanup_training_resources():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_client_accuracy_curve_point(algclass, test_loaders, round_index):
    client_accuracies = []
    for client_idx, test_loader in enumerate(test_loaders):
        _, acc, _, _, _ = algclass.client_eval(client_idx, test_loader)
        client_accuracies.append(float(acc))

    return {
        'round': int(round_index),
        'client_accuracies': client_accuracies,
        'average_accuracy': float(np.mean(client_accuracies)) if client_accuracies else 0.0
    }


def emit_accuracy_curve_point(algclass, test_loaders, round_index, accuracy_curve):
    curve_point = evaluate_client_accuracy_curve_point(
        algclass,
        test_loaders,
        round_index
    )
    accuracy_curve.append(curve_point)
    return curve_point


def should_run_evaluation(round_index, total_rounds, eval_every):
    return (round_index % eval_every == 0) or (round_index == total_rounds - 1)


def emit_training_event(event_queue, data):
    job_id = data.get('job_id')
    if data.get('type') in {'result', 'error', 'cancelled'} and job_id:
        with training_jobs_lock:
            job = training_jobs.get(job_id)
            if job:
                job['terminal_event_sent'] = True
    event_queue.put(data)


def close_training_event_queue(event_queue):
    event_queue.put(None)


class ProgressEmitter:
    def __init__(self, event_queue, total_steps):
        self.event_queue = event_queue
        self.total_steps = max(1, total_steps)
        self.last_progress = -1
        self.last_emit_time = 0.0

    def emit(self, current_step, message, force=False):
        progress = int(current_step / self.total_steps * 100)
        progress = min(100, max(0, progress))
        now = time.monotonic()
        should_emit = (
            force or
            progress >= 100 or
            progress - self.last_progress >= PROGRESS_MIN_DELTA_PERCENT or
            now - self.last_emit_time >= PROGRESS_MIN_INTERVAL_SECONDS
        )

        if not should_emit:
            return

        self.last_progress = progress
        self.last_emit_time = now
        emit_training_event(self.event_queue, {
            'type': 'progress',
            'progress': progress,
            'current_iter': current_step,
            'total_iter': self.total_steps,
            'message': message
        })


def ensure_not_cancelled(cancel_event):
    if cancel_event.is_set():
        raise TrainingCancelled()


def cleanup_finished_jobs():
    now = time.time()
    with training_jobs_lock:
        stale_job_ids = [
            job_id
            for job_id, job in training_jobs.items()
            if not job.get('running') and
            job.get('finished_at') and
            now - job['finished_at'] > FINISHED_JOB_TTL_SECONDS
        ]
        for job_id in stale_job_ids:
            training_jobs.pop(job_id, None)


def get_training_job(job_id):
    with training_jobs_lock:
        return training_jobs.get(job_id)


def remove_training_job(job_id):
    with training_jobs_lock:
        training_jobs.pop(job_id, None)


def find_running_job():
    with training_jobs_lock:
        for job_id, job in training_jobs.items():
            if job.get('running'):
                return job_id, job
    return None, None


def mark_job_finished(job_id):
    remove_after_finish = False
    with training_jobs_lock:
        job = training_jobs.get(job_id)
        if job:
            job['running'] = False
            job['finished_at'] = time.time()
            remove_after_finish = job.get('stream_disconnected', False)
    if remove_after_finish:
        remove_training_job(job_id)


def run_training_job(data, job_id, event_queue, cancel_event):
    """后台线程中的核心训练逻辑，通过队列向 SSE 端点发送事件。"""
    try:
        # 构建参数对象
        class Args:
            pass

        frontend_requested_n_clients = data.get('n_clients', 20)
        args = Args()
        args.alg = data.get('alg', 'fedavg')
        args.dataset = data.get('dataset', 'medmnist')
        args.iters = data.get('iters', 300)
        args.wk_iters = data.get('wk_iters', 1)
        args.eval_every = max(1, int(data.get('eval_every', 1)))
        args.non_iid_alpha = data.get('non_iid_alpha', 0.1)
        args.datapercent = data.get('datapercent', 0.1)
        args.root_dir = data.get('root_dir', '../data/')
        args.save_path = data.get('save_path', '../cks/')
        args.device = data.get('device', 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu')
        args.batch = data.get('batch', 32)
        args.lr = data.get('lr')
        if args.lr is None:
            default_lrs = {
                'base': 0.01,
                'fedavg': 0.01,
                'fedprox': 0.005,
                'fedbn': 0.01,
                'fedap': 0.005,
                'metafed': 0.001
            }
            args.lr = default_lrs.get(args.alg, 0.01)
        args.n_clients = data.get('n_clients', 20)
        args.partition_data = data.get('partition_data', 'non_iid_dirichlet')
        args.plan = data.get('plan', 1)
        args.pretrained_iters = data.get('pretrained_iters', 150)
        args.seed = data.get('seed', 0)
        args.nosharebn = data.get('nosharebn', False)

        # 算法特定参数
        args.mu = data.get('mu', 0.001)
        args.threshold = data.get('threshold', 0.6)
        args.lam = data.get('lam', 1.0)
        args.model_momentum = data.get('model_momentum', 0.5)
        args.cancel_checker = cancel_event.is_set
        requested_iters = args.iters

        # 设置随机种子
        args.dataset = normalize_dataset_name(args.dataset)
        args.random_state = np.random.RandomState(1)
        set_random_seed(args.seed)

        # 处理图像数据集的特殊参数
        uses_domain_split = args.dataset in ['vlcs', 'pacs', 'officehome', 'office-caltech']
        if uses_domain_split:
            args = img_param_init(args)
            args.cancel_checker = cancel_event.is_set
            args.n_clients = 4

        ensure_not_cancelled(cancel_event)

        # 准备数据
        train_loaders, val_loaders, test_loaders = get_data(args.dataset)(args)

        # 检查是否有客户端数据
        if len(train_loaders) == 0:
            emit_training_event(
                event_queue,
                {
                    'type': 'error',
                    'job_id': job_id,
                    'error': '没有可用的客户端数据，请尝试减少客户端数量或调整数据集'
                }
            )
            return

        ensure_not_cancelled(cancel_event)

        # 使用实际的客户端数量
        actual_n_clients = len(train_loaders)
        args.n_clients = actual_n_clients
        args.cancel_checker = cancel_event.is_set

        config_data = {
            'type': 'training_config',
            'job_id': job_id,
            'requested_n_clients': int(frontend_requested_n_clients),
            'actual_n_clients': int(actual_n_clients),
            'eval_every': int(args.eval_every),
            'client_count_adjusted': bool(actual_n_clients != frontend_requested_n_clients),
            'uses_domain_split': uses_domain_split,
            'message': (
                f'{args.dataset} 按域划分，客户端数量固定为 {actual_n_clients}，'
                '前端设置的 n_clients 和 non_iid_alpha 不参与实际切分。'
                if uses_domain_split
                else (
                    f'实际参与训练的客户端数量为 {actual_n_clients}。'
                    if actual_n_clients == frontend_requested_n_clients
                    else f'请求客户端数为 {frontend_requested_n_clients}，实际参与训练的客户端数为 {actual_n_clients}；'
                         '部分客户端因划分后训练/验证/测试样本为空被自动跳过。'
                )
            )
        }
        emit_training_event(event_queue, config_data)

        # 初始化算法
        algclass = algs.get_algorithm_class(args.alg)(args)

        # 记录算法性能统计的开始时间和内存使用
        algorithm_start_time = time.perf_counter()
        process = psutil.Process(os.getpid())
        start_memory = get_memory_mb(process)
        peak_memory = start_memory

        # 特殊处理 FedAP 和 MetaFed
        if args.alg == 'fedap':
            ensure_not_cancelled(cancel_event)
            algclass.set_client_weight(train_loaders)
        elif args.alg == 'metafed':
            ensure_not_cancelled(cancel_event)
            algclass.init_model_flag(train_loaders, val_loaders)
            args.iters = args.iters - 1
        peak_memory = max(peak_memory, get_memory_mb(process))

        ensure_not_cancelled(cancel_event)

        # 训练过程
        start_iter = 0
        accuracy_curve = []

        # 计算总步骤数
        personalization_steps = actual_n_clients if args.alg == 'metafed' else 0
        if args.alg == 'metafed':
            total_steps = args.iters * actual_n_clients + personalization_steps
        else:
            total_steps = args.iters * args.wk_iters * actual_n_clients
        total_steps = max(1, total_steps)
        current_step = 0
        progress_emitter = ProgressEmitter(event_queue, total_steps)

        # 开始训练
        for a_iter in range(start_iter, args.iters):
            ensure_not_cancelled(cancel_event)
            if args.alg == 'metafed':
                for client_idx in range(actual_n_clients):
                    ensure_not_cancelled(cancel_event)
                    mapped_client_idx = algclass.csort[client_idx]
                    algclass.client_train(client_idx, train_loaders[mapped_client_idx], a_iter)
                    peak_memory = max(peak_memory, get_memory_mb(process))
                    ensure_not_cancelled(cancel_event)

                    current_step += 1
                    progress = int(current_step / total_steps * 100)
                    progress_emitter.emit(
                        current_step,
                        f'训练进度: {progress}% (第 {a_iter+1}/{requested_iters} 轮, 客户端 {client_idx+1}/{actual_n_clients})'
                    )
                ensure_not_cancelled(cancel_event)
                algclass.update_flag(val_loaders)
            else:
                # 客户端训练
                for wi in range(args.wk_iters):
                    for client_idx in range(actual_n_clients):
                        ensure_not_cancelled(cancel_event)
                        algclass.client_train(client_idx, train_loaders[client_idx], a_iter)
                        peak_memory = max(peak_memory, get_memory_mb(process))
                        ensure_not_cancelled(cancel_event)

                        current_step += 1
                        progress = int(current_step / total_steps * 100)
                        progress_emitter.emit(
                            current_step,
                            f'训练进度: {progress}% (第 {a_iter+1}/{requested_iters} 轮, 客户端 {client_idx+1}/{actual_n_clients})'
                        )

                # 服务器聚合
                ensure_not_cancelled(cancel_event)
                algclass.server_aggre()
                peak_memory = max(peak_memory, get_memory_mb(process))

            ensure_not_cancelled(cancel_event)
            if should_run_evaluation(a_iter, args.iters, args.eval_every):
                curve_point = emit_accuracy_curve_point(
                    algclass,
                    test_loaders,
                    a_iter + 1,
                    accuracy_curve
                )
                emit_training_event(event_queue, {'type': 'round_metrics', **curve_point})

        # MetaFed 个性化阶段
        if args.alg == 'metafed':
            for c_idx in range(actual_n_clients):
                ensure_not_cancelled(cancel_event)
                algclass.personalization(
                    c_idx,
                    train_loaders[algclass.csort[c_idx]],
                    val_loaders[algclass.csort[c_idx]]
                )
                peak_memory = max(peak_memory, get_memory_mb(process))
                current_step += 1
                progress = int(current_step / total_steps * 100)
                progress_emitter.emit(
                    current_step,
                    f'训练进度: {progress}% (第 {requested_iters}/{requested_iters} 轮, 个性化阶段 {c_idx+1}/{actual_n_clients})'
                )

            ensure_not_cancelled(cancel_event)
            curve_point = emit_accuracy_curve_point(
                algclass,
                test_loaders,
                requested_iters,
                accuracy_curve
            )
            emit_training_event(event_queue, {'type': 'round_metrics', **curve_point})

        ensure_not_cancelled(cancel_event)
        progress_emitter.emit(
            total_steps,
            '训练进度: 100%（训练完成，正在生成最终结果）',
            force=True
        )

        # 计算算法性能耗时
        algorithm_end_time = time.perf_counter()
        training_duration = algorithm_end_time - algorithm_start_time

        # 评估模型
        test_accs = []
        # 对于故障诊断数据集，计算精确率、召回率、F1分数
        precision = []
        recall = []
        f1_scores = []

        for client_idx in range(actual_n_clients):
            ensure_not_cancelled(cancel_event)
            _, acc, prec, rec, f1 = algclass.client_eval(client_idx, test_loaders[client_idx])
            test_accs.append(float(acc))
            precision.append(float(prec))
            recall.append(float(rec))
            f1_scores.append(float(f1))

        # 计算平均准确率
        mean_acc = float(np.mean(test_accs))

        # 检查是否为故障诊断数据集
        is_fault_diagnosis = args.dataset in ['cwru', 'seu']

        # 计算平均精确率、召回率、F1分数
        mean_precision = float(np.mean(precision)) if precision else 0
        mean_recall = float(np.mean(recall)) if recall else 0
        mean_f1 = float(np.mean(f1_scores)) if f1_scores else 0

        # 计算内存使用
        memory_used = max(0.0, peak_memory - start_memory)

        # 格式化训练时长
        if training_duration < 60:
            training_time_str = f"{training_duration:.2f}秒"
        elif training_duration < 3600:
            minutes = int(training_duration // 60)
            seconds = int(training_duration % 60)
            training_time_str = f"{minutes}分{seconds}秒"
        else:
            hours = int(training_duration // 3600)
            minutes = int((training_duration % 3600) // 60)
            seconds = int(training_duration % 60)
            training_time_str = f"{hours}小时{minutes}分{seconds}秒"

        # 发送最终结果
        result_data = {
            'type': 'result',
            'job_id': job_id,
            'algorithm': args.alg,
            'dataset': args.dataset,
            'non_iid_alpha': args.non_iid_alpha,
            'lr': args.lr,
            'eval_every': int(args.eval_every),
            'requested_n_clients': int(frontend_requested_n_clients),
            'actual_n_clients': int(actual_n_clients),
            'uses_domain_split': uses_domain_split,
            'test_accuracies': test_accs,
            'average_accuracy': mean_acc,
            'accuracy_curve': accuracy_curve,
            'training_time': training_time_str,
            'training_duration_seconds': training_duration
        }
        emit_training_event(event_queue, result_data)

        # 添加到历史记录
        history_record = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'algorithm': args.alg,
            'dataset': args.dataset,
            'non_iid_alpha': args.non_iid_alpha,
            'n_clients': actual_n_clients,
            'iters': requested_iters,
            'wk_iters': args.wk_iters,
            'eval_every': int(args.eval_every),
            'lr': args.lr,
            'average_accuracy': mean_acc,
            'training_time': training_time_str,
            'training_duration_seconds': training_duration,
            'memory_used': round(memory_used, 2)  # MB
        }

        # 添加算法特定参数
        if args.alg == 'metafed':
            history_record['threshold'] = args.threshold
        elif args.alg == 'fedprox':
            history_record['mu'] = args.mu
        elif args.alg == 'fedap':
            history_record['model_momentum'] = args.model_momentum

        # 添加故障诊断专用指标
        if is_fault_diagnosis:
            history_record['precision'] = round(mean_precision, 4)
            history_record['recall'] = round(mean_recall, 4)
            history_record['f1_score'] = round(mean_f1, 4)
        add_history_record(history_record)

    except TrainingCancelled:
        emit_training_event(event_queue, {
            'type': 'cancelled',
            'job_id': job_id,
            'message': '训练已取消'
        })
    except Exception as e:
        emit_training_event(event_queue, {
            'type': 'error',
            'job_id': job_id,
            'error': str(e)
        })
    finally:
        mark_job_finished(job_id)
        cleanup_training_resources()
        close_training_event_queue(event_queue)


@app.route('/api/run-model', methods=['POST'])
def run_model():
    """异步启动联邦学习训练任务，返回 job_id 和 SSE 流地址。"""
    cleanup_finished_jobs()
    data = request.get_json(silent=True) or {}

    required_params = ['alg', 'dataset', 'iters', 'non_iid_alpha']
    for param in required_params:
        if param not in data:
            return jsonify({
                'success': False,
                'error': f'Missing required parameter: {param}'
            }), 400

    running_job_id, _ = find_running_job()
    if running_job_id:
        return jsonify({
            'success': False,
            'error': '当前已有训练任务正在运行，请先取消或等待完成',
            'job_id': running_job_id
        }), 409

    job_id = str(uuid.uuid4())
    event_queue = queue.Queue()
    cancel_event = threading.Event()
    thread = threading.Thread(
        target=run_training_job,
        args=(data, job_id, event_queue, cancel_event),
        daemon=True
    )

    with training_jobs_lock:
        training_jobs[job_id] = {
            'queue': event_queue,
            'cancel_event': cancel_event,
            'thread': thread,
            'running': True,
            'created_at': time.time(),
            'finished_at': None,
            'stream_attached': False,
            'stream_disconnected': False,
            'terminal_event_sent': False
        }

    thread.start()
    return jsonify({
        'success': True,
        'job_id': job_id,
        'stream_url': f'/api/training-stream/{job_id}'
    })


@app.route('/api/training-stream/<job_id>', methods=['GET'])
def training_stream(job_id):
    """消费后台训练队列并向前端推送 SSE 事件。"""
    job = get_training_job(job_id)
    if not job:
        return jsonify({'success': False, 'error': '训练任务不存在或已清理'}), 404

    with training_jobs_lock:
        if job_id in training_jobs:
            training_jobs[job_id]['stream_attached'] = True

    event_queue = job['queue']
    cancel_event = job['cancel_event']

    def generate_stream():
        try:
            while True:
                try:
                    event = event_queue.get(timeout=SSE_QUEUE_TIMEOUT)
                except queue.Empty:
                    yield ': heartbeat\n\n'
                    continue

                if event is None:
                    break

                yield f"data: {json.dumps(event)}\n\n"
        except GeneratorExit:
            cancelled_job = False
            with training_jobs_lock:
                current_job = training_jobs.get(job_id)
                if (
                    current_job and
                    current_job.get('running') and
                    not current_job.get('terminal_event_sent')
                ):
                    current_job['stream_disconnected'] = True
                    current_job['cancel_event'].set()
                    cancelled_job = True
            if cancelled_job:
                print(f'SSE client disconnected; cancelling training job {job_id}.')
            raise
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as e:
            cancelled_job = False
            with training_jobs_lock:
                current_job = training_jobs.get(job_id)
                if current_job and not current_job.get('terminal_event_sent'):
                    current_job['stream_disconnected'] = True
                    current_job['cancel_event'].set()
                    cancelled_job = True
            if cancelled_job:
                print(f'SSE connection lost ({type(e).__name__}); cancelling training job {job_id}.')
        finally:
            remove_now = False
            with training_jobs_lock:
                current_job = training_jobs.get(job_id)
                if current_job:
                    current_job['stream_attached'] = False
                    if current_job.get('running'):
                        current_job['stream_disconnected'] = True
                    remove_now = not current_job.get('running')
            if remove_now:
                remove_training_job(job_id)

    return Response(
        generate_stream(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )


@app.route('/api/cancel-training', methods=['POST'])
def cancel_training():
    """取消指定训练任务；未传 job_id 时取消当前正在运行的任务。"""
    data = request.get_json(silent=True) or {}
    job_id = data.get('job_id')

    if job_id:
        job = get_training_job(job_id)
        if not job or not job.get('running'):
            return jsonify({'success': False, 'message': '指定训练任务不存在或已结束'}), 400
    else:
        job_id, job = find_running_job()
        if not job:
            return jsonify({'success': False, 'message': '当前没有正在运行的训练任务'}), 400

    job['cancel_event'].set()
    return jsonify({'success': True, 'message': '已发送取消请求', 'job_id': job_id})


@app.route('/api/algorithms', methods=['GET'])
def get_algorithms():
    """获取可用的算法列表"""
    return jsonify({'algorithms': ALGORITHMS})

@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    """获取可用的数据集列表"""
    datasets = ['vlcs', 'pacs', 'officehome', 'pamap', 'covid', 'medmnist', 'cwru', 'seu']
    return jsonify({'datasets': datasets})

@app.route('/api/history', methods=['GET'])
def get_history():
    """获取历史训练记录"""
    history = load_history()
    return jsonify({'history': history})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
