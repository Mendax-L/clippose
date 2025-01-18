import numpy as np
from scipy.sparse.linalg import eigs

def quat_wavg_markley(Q, weights):
    """
    计算四元数的加权平均值（Markley等方法）。
    Q: 一个形状为 [n, 4] 的 numpy 数组，表示 n 个四元数。
    weights: 一个形状为 [n,] 的 numpy 数组，表示每个四元数的权重。
    返回加权平均的四元数。
    """
    # 将 Q 转换为 numpy 数组并确保权重也是 numpy 数组
    Q = np.array(Q)
    weights = np.array(weights)
    
    # 初始化 4x4 对称矩阵 M
    M = np.zeros((4, 4))
    n = len(Q)  # 四元数的数量
    w_sum = 0  # 权重的总和
    
    # 构造加权矩阵 M
    for i in range(n):
        q = Q[i]  # 第 i 个四元数
        w_i = weights[i]  # 第 i 个四元数的权重
        M += w_i * np.outer(q, q)  # 加入加权的外积
        w_sum += w_i  # 累加权重和
    
    # 对矩阵 M 进行归一化
    M /= w_sum
    
    # 计算 M 的特征值和特征向量
    # 使用 eigs 求解特征值和特征向量，选择最大的特征值对应的特征向量
    eigvals, eigvecs = eigs(M, k=1, which='LM')  # k=1 表示我们只需要最大的特征值和对应的特征向量
    
    # 返回最大特征值对应的特征向量，取实部
    Qavg = np.real(eigvecs[:, 0])
    
    return Qavg

# 测试示例
Q = np.array([
    [1, 0, 0, 0],       # 0°
    [0.999, 0, 0, 0.044],  # 5°
    [0.999, 0, 0, 0.035],  # 4°
    [1, 0, 0, 0.026],    # 3°
    [1, 0, 0, 0.017],    # 2°
    [1, 0, 0, 0.009]     # 1°
])
weights = np.ones(6) / 6  # 权重设置为相等，简单平均

# 调用加权平均函数
Qavg = quat_wavg_markley(Q, weights)
print("加权平均四元数：", Qavg)
