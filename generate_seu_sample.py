import os
import numpy as np

# 确保输出目录存在
output_path = 'data/seu'
os.makedirs(output_path, exist_ok=True)

# 生成示例数据
# 假设有 1000 个样本，每个样本有 1024 个时间点
x = np.random.rand(1000, 1024)
# 生成 10 个类别的标签
y = np.random.randint(0, 10, 1000)

# 保存数据
np.save(os.path.join(output_path, 'x.npy'), x)
np.save(os.path.join(output_path, 'y.npy'), y)

print('SEU 示例数据生成完成！')
print(f'样本形状: {x.shape}')
print(f'标签形状: {y.shape}')
print(f'标签类别: {np.unique(y)}')
