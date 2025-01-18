import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as spatial_Rotation
import numpy as np

class RotationAttention(torch.nn.Module):
    def __init__(self, d_model=512, temperature=0.1, hidden_size=512):
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
        - Q: [1, d_model]  查询
        - K: [m, d_model]  键
        - V: [m, d_model]  值
        - r: [m, 9]  旋转矩阵的前两个向量
        """
        
        m = refer_feature.shape[0] 

        Q = img_feature.expand(m, -1) - refer_feature 

        K = refer_feature.view(m, 1, -1) - refer_feature.view(1, m, -1) 
        K = K.view(-1, self.d_model) 

        refer_r = refer_r.view(m, 3, 3)
        refer_r_inv = torch.inverse(refer_r)
        refer_r_expand = refer_r.view(m, 1, 3, 3)
        refer_r_inv = refer_r_inv.view(1, m, 3, 3)

        V = torch.matmul(refer_r_expand,refer_r_inv)  
        V = V.view(-1, 9)[:,:6]
        V = self.rot_encoder(V) 


        # 计算Q和K的相似度得分（点积注意力）
        attention_scores = torch.matmul(Q, K.T) / self.temperature
        
        # 使用softmax计算注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 计算加权和值
        output = torch.matmul(attention_weights, V)

        r = self.rot_decoder(output)  # 解码得到旋转矩阵
        R = self.recoverR(r).view(-1,3,3)
        print(R.shape)
        print(refer_r.shape)

        R = torch.matmul(R,refer_r) 


        
        return R
    def average_rotation_matrices(self, R_matrices):
        """
        计算多个旋转矩阵的平均旋转矩阵
        :param R_matrices: 旋转矩阵集合, torch.Tensor, 形状 [batch_size, 3, 3]
        :return: 平均旋转矩阵
        """
        # 将旋转矩阵转换为四元数
        quaternions = []
        for i in range(R_matrices.shape[0]):
            r = spatial_Rotation.from_matrix(R_matrices[i].detach().cpu().numpy())
            quaternions.append(r.as_quat())  # 获取四元数

        quaternions = np.array(quaternions)

        # 使用四元数计算平均值
        avg_quaternion = np.mean(quaternions, axis=0)

        # 将平均四元数转换回旋转矩阵
        avg_rotation = spatial_Rotation.from_quat(avg_quaternion).as_matrix()

        return torch.tensor(avg_rotation).float().cuda()

    def recoverR(self, r):
        r1, r2 = r[:, :3], r[:, 3:]
        r2 = r2 - torch.sum(r1 * r2, dim=1, keepdim=True) * r1
        r3 = torch.cross(r1, r2, dim=1)
        
        r1 = F.normalize(r1, p=2, dim=1)  # 归一化到单位向量
        r2 = F.normalize(r2, p=2, dim=1)  # 归一化到单位向量
        r3 = F.normalize(r3, p=2, dim=1)  # 归一化到单位向量
        
        R = torch.cat([r1, r2, r3], dim=1)
        return R


if __name__ == "__main__":
    # 示例数据
    d_model = 512
    m = 10  # 假设有10个键值对

    # 随机生成Q, K, V
    img_feature = torch.randn(d_model)  # 查询
    refer = torch.randn(m, d_model)  # 键
    r = torch.randn(m, 9)  # 旋转矩阵的前两个向量

    # 创建模型实例
    rotation_attention = RotationAttention(d_model=d_model)

    # 计算注意力输出
    R = rotation_attention(img_feature, refer, r)

    R = rotation_attention.average_rotation_matrices(R)

    # 输出结果
    print("R:", R.shape)
