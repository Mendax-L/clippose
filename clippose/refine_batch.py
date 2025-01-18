import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as spatial_Rotation
import numpy as np
import torch
import numpy as np
from scipy.spatial.transform import Rotation as spatial_Rotation

class RotationAvg:
    def __init__(self, threshold=0.5):
        self.threshold = threshold  # 用于剔除异常旋转矩阵的差异阈值
    
    def rotation_difference(self, R1, R2):
        """
        计算两个旋转矩阵之间的差异度量（使用角度差异）
        :param R1: 旋转矩阵1, torch.Tensor, 形状 [3, 3]
        :param R2: 旋转矩阵2, torch.Tensor, 形状 [3, 3]
        :return: 旋转差异角度（度数）
        """
        r1 = spatial_Rotation.from_matrix(R1.detach().cpu().numpy())
        r2 = spatial_Rotation.from_matrix(R2.detach().cpu().numpy())
        
        # 计算旋转矩阵之间的差异（旋转向量表示）
        diff_rotation = r1.inv() * r2
        angle_diff = diff_rotation.magnitude()  # 获取旋转差异的角度（弧度）
        
        # 将弧度转换为度数
        return np.degrees(angle_diff)  # 返回角度差异，单位为度
    
    def average_rotation_matrices(self, R_matrices):
        """
        计算多个旋转矩阵的平均旋转矩阵，剔除异常矩阵
        :param R_matrices: 旋转矩阵集合, torch.Tensor, 形状 [batch_size, m, 3, 3]
        :return: 平均旋转矩阵
        """
        batch_size, m, _, _ = R_matrices.shape
        
        # 用于存储剔除异常后的旋转矩阵集合
        filtered_R_matrices = []
        
        for i in range(batch_size):
            batch_matrices = R_matrices[i]
            
            # 计算所有旋转矩阵的差异矩阵
            diff_matrix = np.zeros((m, m))  # [m, m] 矩阵用于存储旋转矩阵间的差异
            for j in range(m):
                for k in range(m):
                    diff_matrix[j, k] = self.rotation_difference(batch_matrices[j], batch_matrices[k])
            
            # 计算每个旋转矩阵与其他矩阵的差异的平均值和标准差
            mean_diff = np.mean(diff_matrix, axis=1)  # 每个矩阵与其他矩阵的平均差异
            std_diff = np.std(diff_matrix, axis=1)    # 每个矩阵与其他矩阵差异的标准差
            
            # 判定哪些矩阵是异常的
            valid_matrices = []
            for j in range(m):
                if mean_diff[j] < self.threshold and std_diff[j] < self.threshold:  # 如果差异小于阈值，则认为是正常矩阵
                    valid_matrices.append(batch_matrices[j])
            
            # 如果没有有效矩阵，退回第一个矩阵
            if valid_matrices:
                filtered_R_matrices.append(torch.stack(valid_matrices))
            else:
                filtered_R_matrices.append(batch_matrices[0:1])
        
        # 将所有批次的有效旋转矩阵进行平均
        all_valid_matrices = torch.stack(filtered_R_matrices)
        avg_rotation_matrices = torch.mean(all_valid_matrices, dim=1)  # 对有效矩阵取平均

        return avg_rotation_matrices

class RotationAttention(torch.nn.Module):
    def __init__(self, d_model=512, temperature=0.3, hidden_size=512):
        super(RotationAttention, self).__init__()
        self.d_model = d_model
        self.temperature = temperature
        
        # 定义encoder和decoder
        self.rot_encoder = self._create_encoder()
        self.rot_decoder = self._create_decoder(hidden_size)
        
    def _create_encoder(self):
        return torch.nn.Sequential(
            torch.nn.Linear(6, self.d_model),
        )
    
    def _create_decoder(self, hidden_size):
        return torch.nn.Sequential(
            torch.nn.Linear(self.d_model, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 6)
        )

    def forward(self, img_feature, refer_feature, refer_r):
        """
        - Q: [batch_size, d_model]  查询
        - K: [batch_size, m, d_model]  键
        - V: [batch_size, m, d_model]  值
        - r: [batch_size, m, 3, 3]  旋转矩阵
        """
        batch_size, m, _ = refer_feature.shape  # 获取批次大小

        Q = img_feature.unsqueeze(1).expand(-1, m, -1) - refer_feature  # [batch_size, m, d_model]
        
        # 计算K
        K = refer_feature.unsqueeze(1) - refer_feature.unsqueeze(2)  # [batch_size, m, m, d_model]
        # 检查对角线上的元素是否接近零
        # batch_size, m, _, d_model = K.shape
        # diagonal_values = K[:, torch.arange(m), torch.arange(m), :]  # 提取对角线元素
        # print(f"对角线元素：{diagonal_values}")

        # # 检查是否接近零（可以设定一个阈值）
        # threshold = 1e-6
        # is_diagonal_zero = torch.all(torch.abs(diagonal_values) < threshold)
        # print(f"对角线元素是否接近零: {is_diagonal_zero.item()}")
        K = K.view(batch_size, -1, self.d_model)  # [batch_size, m * m, d_model]

        # 计算V
        refer_r = refer_r.view(batch_size, m, 3, 3)  # [batch_size, m, 3, 3]
        refer_r_inv = torch.inverse(refer_r)  # [batch_size, m, 3, 3]
        refer_r_expand = refer_r.unsqueeze(2)  # [batch_size, m, 1, 3, 3]
        refer_r_inv_expand = refer_r_inv.unsqueeze(1)  # [batch_size, 1, m, 3, 3]

        V = torch.matmul(refer_r_inv_expand, refer_r_expand)  # [batch_size, m, m, 3, 3]

        # batch_size, m, _, _, _ = V.shape
        # print(f"V的形状：{V.shape}")
        # # 提取对角线元素
        # diagonal_values = V[:, torch.arange(m), torch.arange(m), :, :]  # 提取对角线元素 [batch_size, m, m, 3, 3]
        # print(f"对角线元素：{diagonal_values}")
        # # 创建单位矩阵 [3, 3]
        # identity_matrix = torch.eye(3, device=diagonal_values.device)  # [3, 3]单位矩阵

        # # 扩展单位矩阵到 [batch_size, m, m, 3, 3]
        # identity_matrix_expanded = identity_matrix.unsqueeze(0).unsqueeze(0).expand(batch_size, m, 3, 3)
        # is_identity_matrix = torch.allclose(diagonal_values, identity_matrix_expanded, atol=1e-4)
        # print(f"对角线是否为单位矩阵: {is_identity_matrix}")
        V = V.view(batch_size, m * m, 9)[:, :, :6]  # [batch_size, m * m, 6]
        V = self.rot_encoder(V)  # [batch_size, m * m, d_model]

        # 计算Q和K的相似度得分（点积注意力）
        attention_scores = torch.matmul(Q, K.transpose(1, 2)) / self.temperature  # [batch_size, m, m]
        
        # 使用softmax计算注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, m, m]
        
        # 计算加权和值
        output = torch.matmul(attention_weights, V)  # [batch_size, m, d_model]

        r = self.rot_decoder(output)  # 解码得到旋转矩阵 [batch_size, m, 6]

        # print(r.shape)
        R = self.recoverR(r)  # [batch_size, m, 3, 3]

        # 可以选择进一步的旋转矩阵操作，如恢复整体旋转
        R = torch.matmul(refer_r, R)  # [batch_size, m, 3, 3]

        return R


    def recoverR(self, r):
        """
        根据旋转矩阵的前两个向量 r1, r2 和 r3，恢复旋转矩阵
        :param r: [batch_size, m, 6] - 包含旋转矩阵的前两个向量
        :return: [batch_size, m, 3, 3] - 恢复的旋转矩阵
        """
        batch_size, m, _ = r.size()  # 获取批次大小和 m 的数量

        # 切分 r 为 r1 和 r2
        r1, r2 = r[:, :, :3], r[:, :, 3:]  # [batch_size, m, 3] 和 [batch_size, m, 3]

        # 计算 r2 的正交化
        r2 = r2 - torch.sum(r1 * r2, dim=2, keepdim=True) * r1  # [batch_size, m, 3]

        # 计算 r3 为 r1 和 r2 的叉积
        r3 = torch.cross(r1, r2, dim=2)  # [batch_size, m, 3]

        # 归一化 r1, r2 和 r3
        r1 = F.normalize(r1, p=2, dim=2)  # 归一化到单位向量 [batch_size, m, 3]
        r2 = F.normalize(r2, p=2, dim=2)  # 归一化到单位向量 [batch_size, m, 3]
        r3 = F.normalize(r3, p=2, dim=2)  # 归一化到单位向量 [batch_size, m, 3]

        # 将 r1, r2, r3 合并为旋转矩阵 R
        R = torch.stack([r1, r2, r3], dim=3)  # [batch_size, m, 3, 3]

        return R


if __name__ == "__main__":
    # 示例数据
    d_model = 512
    batch_size = 8  # 假设有8个样本
    m = 10  # 每个样本有10个参考矩阵

    # 随机生成Q, K, V
    img_feature = torch.randn(batch_size, d_model)  # 查询 [batch_size, d_model]
    refer = torch.randn(batch_size, m, d_model)  # 键 [batch_size, m, d_model]
    r = torch.randn(batch_size, m, 3, 3)  # 旋转矩阵 [batch_size, m, 3, 3]

    # 创建模型实例
    rotation_attention = RotationAttention(d_model=d_model)

    # 计算注意力输出
    R = rotation_attention(img_feature, refer, r)

    # 计算旋转矩阵的平均值
    R_Avg = RotationAvg()
    avg_R = R_Avg.average_rotation_matrices(R)

    # 输出结果
    print("R:", R.shape)
    print("Average Rotation Matrix:", avg_R.shape)
