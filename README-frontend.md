# PersonalizedFL 前后端系统使用说明

## 系统概述

本系统为 PersonalizedFL 联邦学习框架添加了前后端支持，使用户能够通过网页界面方便地调用模型并查看结果。系统由以下部分组成：

- **后端 API 服务**：基于 Flask 实现，提供模型调用和结果返回的接口
- **前端界面**：基于 HTML、CSS 和 JavaScript 实现，提供用户交互界面
- **PersonalizedFL 核心**：联邦学习算法实现

## 安装与配置

### 1. 环境要求

- Python 3.7+
- PyTorch 1.7+ 
- Flask 2.0+
- Flask-CORS 3.0+
- NumPy 1.21+
- Pillow 8.0+

### 2. 安装步骤

1. **克隆仓库**

   ```bash
   git clone https://github.com/microsoft/PersonalizedFL.git
   cd PersonalizedFL
   ```

2. **创建并激活虚拟环境**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **安装核心依赖**

   ```bash
   pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple torch torchvision pillow
   ```

4. **安装后端 API 依赖**

   ```bash
   cd backend
   source venv/bin/activate
   pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple Flask Flask-CORS numpy torch torchvision pillow
   cd ..
   ```

### 3. 数据准备

1. **创建数据目录**

   ```bash
   mkdir -p data/medmnist
   ```

2. **下载或生成测试数据**

   可以从以下链接下载 MedMNIST 数据集：
   https://wjdcloud.blob.core.windows.net/dataset/cycfed/medmnist.tar.gz

   或者使用以下命令生成模拟数据：

   ```bash
   source venv/bin/activate
   python3 -c "import numpy as np; np.save('data/medmnist/xdata.npy', np.random.rand(1000, 28, 28)); np.save('data/medmnist/ydata.npy', np.random.randint(0, 10, 1000))"
   ```

## 运行系统

### 1. 启动后端 API 服务

```bash
cd backend
source venv/bin/activate
python app.py
```

后端服务将在 `http://localhost:5001` 上运行。

### 2. 启动前端服务器

```bash
cd frontend
python3 -m http.server 8000
```

前端界面将在 `http://localhost:8000` 上可用。

### 3. 访问系统

打开浏览器，访问 `http://localhost:8000` 即可使用系统。

## 使用方法

1. **选择算法**：从下拉菜单中选择要使用的联邦学习算法（如 fedavg、fedprox、fedbn、fedap、metafed 或 base）。

2. **选择数据集**：从下拉菜单中选择要使用的数据集（如 medmnist、pacs、vlcs 等）。

3. **设置参数**：
   - 非独立同分布程度 (alpha)：值越小，数据分布越不均匀
   - 通信轮数：服务器与客户端之间的通信次数
   - 本地训练轮数：每个客户端本地训练的轮数
   - 客户端数量：参与联邦学习的客户端数量

4. **算法特定参数**：根据选择的算法，设置相应的参数：
   - FedProx：正则化参数 (mu)
   - FedAP：动量参数
   - MetaFed：阈值参数

5. **运行模型**：点击 "运行模型" 按钮开始训练。

6. **查看结果**：训练完成后，系统将显示每个客户端的测试准确率和平均准确率，并生成准确率分布图。

## API 文档

### 1. 获取算法列表

**URL**：`/api/algorithms`
**方法**：GET
**响应**：
```json
{
  "algorithms": ["fedavg", "fedprox", "fedbn", "base", "fedap", "metafed"]
}
```

### 2. 获取数据集列表

**URL**：`/api/datasets`
**方法**：GET
**响应**：
```json
{
  "datasets": ["vlcs", "pacs", "officehome", "pamap", "covid", "medmnist"]
}
```

### 3. 运行模型

**URL**：`/api/run-model`
**方法**：POST
**请求体**：
```json
{
  "alg": "fedavg",
  "dataset": "medmnist",
  "iters": 300,
  "wk_iters": 1,
  "non_iid_alpha": 0.1,
  "n_clients": 20,
  "device": "cuda",
  "mu": 0.001,
  "model_momentum": 0.5,
  "threshold": 0.6
}
```

**响应**：
```json
{
  "algorithm": "fedavg",
  "dataset": "medmnist",
  "non_iid_alpha": 0.1,
  "test_accuracies": [0.75, 0.8, 0.78, ...],
  "average_accuracy": 0.77
}
```

## 注意事项

1. **计算资源**：联邦学习训练可能需要大量计算资源，特别是当客户端数量较多或训练轮数较大时。建议在具有足够计算能力的设备上运行。

2. **数据准备**：确保已正确准备数据集，否则可能会导致训练失败。

3. **参数调整**：不同的算法和数据集可能需要不同的参数设置，建议根据实际情况进行调整。

4. **端口冲突**：如果端口 5001 或 8000 已被占用，需要修改相应的端口配置。

5. **网络连接**：确保前端和后端服务在同一网络环境中，以便正常通信。

## 故障排除

### 1. 后端服务启动失败

- 检查端口是否被占用
- 检查依赖是否已正确安装
- 检查数据是否已正确准备

### 2. 前端无法连接后端

- 检查后端服务是否正在运行
- 检查前端代码中的 API 地址是否正确
- 检查网络连接是否正常

### 3. 模型训练失败

- 检查数据格式是否正确
- 检查参数设置是否合理
- 检查计算资源是否足够

## 扩展与定制

1. **添加新算法**：在 `alg/` 目录中实现新算法，并在 `alg/algs.py` 中注册。

2. **添加新数据集**：在 `datautil/prepare_data.py` 中添加新数据集的处理逻辑。

3. **自定义前端界面**：修改 `frontend/index.html` 文件，添加新的功能或改进界面设计。

4. **扩展 API**：在 `backend/app.py` 中添加新的 API 端点，提供更多功能。

## 联系与支持

如果您在使用过程中遇到问题，请联系项目维护者：

- Wang lu: luwang@ict.ac.cn
- Jindong Wang: jindongwang@outlook.com

## 引用

如果您使用本系统进行研究，请引用以下论文：

```
@Misc{PersonalizedFL,
howpublished = {\url{https://github.com/microsoft/PersonalizedFL}},
title = {PersonalizedFL: Personalized Federated Learning Toolkit},
author = {Lu, Wang and Wang, Jindong}
}
```