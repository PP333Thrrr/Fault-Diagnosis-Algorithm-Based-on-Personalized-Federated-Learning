import os
import sys
import json
import time
import uuid
import numpy as np
import psutil
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

# 全局变量存储训练进度
training_progress = {
    'running': False,
    'cancel_requested': False,
    'job_id': None
}


def get_memory_mb(process):
    return process.memory_info().rss / 1024 / 1024


def is_cancel_requested(job_id):
    return (
        training_progress.get('running') and
        training_progress.get('job_id') == job_id and
        training_progress.get('cancel_requested', False)
    )


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

@app.route('/api/run-model', methods=['POST'])
def run_model():
    """运行联邦学习模型并返回结果（带进度）"""
    data = request.json
    
    def generate():
        job_id = str(uuid.uuid4())
        try:
            if training_progress.get('running'):
                yield f"data: {json.dumps({'type': 'error', 'error': '当前已有训练任务正在运行，请先取消或等待完成'})}\n\n"
                return

            training_progress['running'] = True
            training_progress['cancel_requested'] = False
            training_progress['job_id'] = job_id

            # 验证必要参数
            required_params = ['alg', 'dataset', 'iters', 'non_iid_alpha']
            for param in required_params:
                if param not in data:
                    yield f"data: {json.dumps({'type': 'error', 'error': f'Missing required parameter: {param}'})}\n\n"
                    return
            
            # 构建参数对象
            class Args:
                pass
            
            args = Args()
            args.alg = data.get('alg', 'fedavg')
            args.dataset = data.get('dataset', 'medmnist')
            args.iters = data.get('iters', 300)
            args.wk_iters = data.get('wk_iters', 1)
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
            requested_iters = args.iters
            
            # 设置随机种子
            args.dataset = normalize_dataset_name(args.dataset)
            args.random_state = np.random.RandomState(1)
            set_random_seed(args.seed)
            
            # 处理图像数据集的特殊参数
            if args.dataset in ['vlcs', 'pacs', 'officehome', 'office-caltech']:
                args = img_param_init(args)
                args.n_clients = 4
            
            # 准备数据
            train_loaders, val_loaders, test_loaders = get_data(args.dataset)(args)
            
            # 检查是否有客户端数据
            if len(train_loaders) == 0:
                yield f"data: {json.dumps({'type': 'error', 'error': '没有可用的客户端数据，请尝试减少客户端数量或调整数据集'})}\n\n"
                return

            if is_cancel_requested(job_id):
                yield f"data: {json.dumps({'type': 'cancelled', 'message': '训练已取消'})}\n\n"
                return
            
            # 使用实际的客户端数量
            requested_n_clients = args.n_clients
            actual_n_clients = len(train_loaders)
            args.n_clients = actual_n_clients
            args.cancel_checker = lambda: is_cancel_requested(job_id)

            config_data = {
                'type': 'training_config',
                'requested_n_clients': int(requested_n_clients),
                'actual_n_clients': int(actual_n_clients),
                'client_count_adjusted': bool(actual_n_clients != requested_n_clients),
                'message': (
                    f'实际参与训练的客户端数量为 {actual_n_clients}。'
                    if actual_n_clients == requested_n_clients
                    else f'请求客户端数为 {requested_n_clients}，实际参与训练的客户端数为 {actual_n_clients}；'
                         '部分客户端因划分后训练/验证/测试样本为空被自动跳过。'
                )
            }
            yield f"data: {json.dumps(config_data)}\n\n"
            
            # 初始化算法
            algclass = algs.get_algorithm_class(args.alg)(args)
            
            # 记录算法性能统计的开始时间和内存使用
            algorithm_start_time = time.perf_counter()
            process = psutil.Process(os.getpid())
            start_memory = get_memory_mb(process)
            peak_memory = start_memory

            # 特殊处理 FedAP 和 MetaFed
            if args.alg == 'fedap':
                algclass.set_client_weight(train_loaders)
            elif args.alg == 'metafed':
                algclass.init_model_flag(train_loaders, val_loaders)
                args.iters = args.iters - 1
            peak_memory = max(peak_memory, get_memory_mb(process))

            if is_cancel_requested(job_id):
                yield f"data: {json.dumps({'type': 'cancelled', 'message': '训练已取消'})}\n\n"
                return
            
            # 训练过程
            best_acc = [0] * actual_n_clients
            best_tacc = [0] * actual_n_clients
            start_iter = 0
            accuracy_curve = []
            
            # 计算总步骤数
            personalization_steps = actual_n_clients if args.alg == 'metafed' else 0
            if args.alg == 'metafed':
                total_steps = args.iters * actual_n_clients + personalization_steps
            else:
                total_steps = args.iters * args.wk_iters * actual_n_clients
            current_step = 0
            
            # 开始训练
            for a_iter in range(start_iter, args.iters):
                if args.alg == 'metafed':
                    for client_idx in range(actual_n_clients):
                        if is_cancel_requested(job_id):
                            yield f"data: {json.dumps({'type': 'cancelled', 'message': '训练已取消'})}\n\n"
                            return
                        mapped_client_idx = algclass.csort[client_idx]
                        algclass.client_train(client_idx, train_loaders[mapped_client_idx], a_iter)
                        peak_memory = max(peak_memory, get_memory_mb(process))

                        if is_cancel_requested(job_id):
                            yield f"data: {json.dumps({'type': 'cancelled', 'message': '训练已取消'})}\n\n"
                            return
                        
                        # 更新进度
                        current_step += 1
                        progress = int(current_step / total_steps * 100)
                        
                        # 发送进度更新
                        progress_data = {
                            'type': 'progress',
                            'progress': progress,
                            'current_iter': current_step,
                            'total_iter': total_steps,
                            'message': f'训练进度: {progress}% (第 {a_iter+1}/{requested_iters} 轮, 客户端 {client_idx+1}/{actual_n_clients})'
                        }
                        yield f"data: {json.dumps(progress_data)}\n\n"
                    algclass.update_flag(val_loaders)
                else:
                    # 客户端训练
                    for wi in range(args.wk_iters):
                        for client_idx in range(actual_n_clients):
                            if is_cancel_requested(job_id):
                                yield f"data: {json.dumps({'type': 'cancelled', 'message': '训练已取消'})}\n\n"
                                return
                            algclass.client_train(client_idx, train_loaders[client_idx], a_iter)
                            peak_memory = max(peak_memory, get_memory_mb(process))

                            if is_cancel_requested(job_id):
                                yield f"data: {json.dumps({'type': 'cancelled', 'message': '训练已取消'})}\n\n"
                                return
                            
                            # 更新进度
                            current_step += 1
                            progress = int(current_step / total_steps * 100)
                            
                            # 发送进度更新
                            progress_data = {
                                'type': 'progress',
                                'progress': progress,
                                'current_iter': current_step,
                                'total_iter': total_steps,
                                'message': f'训练进度: {progress}% (第 {a_iter+1}/{requested_iters} 轮, 客户端 {client_idx+1}/{actual_n_clients})'
                            }
                            yield f"data: {json.dumps(progress_data)}\n\n"
                    
                    # 服务器聚合
                    algclass.server_aggre()
                    peak_memory = max(peak_memory, get_memory_mb(process))

                curve_point = emit_accuracy_curve_point(
                    algclass,
                    test_loaders,
                    a_iter + 1,
                    accuracy_curve
                )
                yield f"data: {json.dumps({'type': 'round_metrics', **curve_point})}\n\n"

                if is_cancel_requested(job_id):
                    yield f"data: {json.dumps({'type': 'cancelled', 'message': '训练已取消'})}\n\n"
                    return
            
            # MetaFed 个性化阶段
            if args.alg == 'metafed':
                for c_idx in range(actual_n_clients):
                    if is_cancel_requested(job_id):
                        yield f"data: {json.dumps({'type': 'cancelled', 'message': '训练已取消'})}\n\n"
                        return
                    algclass.personalization(c_idx, train_loaders[algclass.csort[c_idx]], val_loaders[algclass.csort[c_idx]])
                    peak_memory = max(peak_memory, get_memory_mb(process))
                    current_step += 1
                    progress = int(current_step / total_steps * 100)
                    progress_data = {
                        'type': 'progress',
                        'progress': progress,
                        'current_iter': current_step,
                        'total_iter': total_steps,
                        'message': f'训练进度: {progress}% (第 {requested_iters}/{requested_iters} 轮, 个性化阶段 {c_idx+1}/{actual_n_clients})'
                    }
                    yield f"data: {json.dumps(progress_data)}\n\n"

                curve_point = emit_accuracy_curve_point(
                    algclass,
                    test_loaders,
                    requested_iters,
                    accuracy_curve
                )
                yield f"data: {json.dumps({'type': 'round_metrics', **curve_point})}\n\n"

            if is_cancel_requested(job_id):
                yield f"data: {json.dumps({'type': 'cancelled', 'message': '训练已取消'})}\n\n"
                return

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
                'algorithm': args.alg,
                'dataset': args.dataset,
                'non_iid_alpha': args.non_iid_alpha,
                'lr': args.lr,
                'requested_n_clients': int(requested_n_clients),
                'actual_n_clients': int(actual_n_clients),
                'test_accuracies': test_accs,
                'average_accuracy': mean_acc,
                'accuracy_curve': accuracy_curve,
                'training_time': training_time_str,
                'training_duration_seconds': training_duration
            }
            yield f"data: {json.dumps(result_data)}\n\n"
            
            # 添加到历史记录
            history_record = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'algorithm': args.alg,
                'dataset': args.dataset,
                'non_iid_alpha': args.non_iid_alpha,
                'n_clients': actual_n_clients,
                'iters': requested_iters,
                'wk_iters': args.wk_iters,
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
            yield f"data: {json.dumps({'type': 'cancelled', 'message': '训练已取消'})}\n\n"
        except Exception as e:
            error_data = {
                'type': 'error',
                'error': str(e)
            }
            yield f"data: {json.dumps(error_data)}\n\n"
        finally:
            if training_progress.get('job_id') == job_id:
                training_progress['running'] = False
                training_progress['cancel_requested'] = False
                training_progress['job_id'] = None
    
    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )


@app.route('/api/cancel-training', methods=['POST'])
def cancel_training():
    """取消当前训练任务"""
    if not training_progress.get('running'):
        return jsonify({'success': False, 'message': '当前没有正在运行的训练任务'}), 400

    training_progress['cancel_requested'] = True
    return jsonify({'success': True, 'message': '已发送取消请求'})

@app.route('/api/algorithms', methods=['GET'])
def get_algorithms():
    """获取可用的算法列表"""
    from alg.algs import ALGORITHMS
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
