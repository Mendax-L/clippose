import numpy as np
import torch

def get_allorot(centeruv, rot_pred, Kc_inv):
    device = centeruv.device
    centeruv_3d = torch.cat([centeruv, torch.ones((centeruv.shape[0],1)).to(device)], dim=1)
    p = torch.tensor([0, 0, 1], dtype=torch.float32).repeat(centeruv.shape[0], 1).to(device)
    # print(f'p:{p.shape}')   
    # print(f'centeruv_3d:{centeruv_3d.shape}')        
    # print(f'Kc_inv:{Kc_inv.shape}')        
    q = torch.matmul(Kc_inv, centeruv_3d.unsqueeze(-1)).squeeze(-1)
    # print(f'q:{q.shape}')

    Rc=psi_tensor(p, q)
    # print(f'Rc:{Rc.shape}')

    rot_pred = rot_pred.view(-1, 3, 3)

    R_pred = torch.matmul(Rc, rot_pred)

    return R_pred


def psi_tensor(p, q):
    # 计算 p 和 q 的范数并归一化
    p = p / torch.norm(p, dim=1, keepdim=True)
    q = q / torch.norm(q, dim=1, keepdim=True)
    
    # 计算 p 和 q 的叉积
    r = torch.cross(p, q, dim=1)
    
    # 计算 r 的范数平方
    r_norm_sq = torch.sum(r * r, dim=1, keepdim=True)
    
    # 计算 p 和 q 的点积
    p_dot_q = torch.sum(p* q, dim=1, keepdim=True)
    # p_dot_q = torch.matmul(p.T, q).T
    # print(f"p_dot_q: {p_dot_q.shape}")

    
    # 构造单位矩阵
    I = torch.eye(3, device=p.device).unsqueeze(0).repeat(p.shape[0], 1, 1)
    # print(f"I: {I.shape}")
    
    # 构造 r 的反对称矩阵（3x3）
    r_cross = torch.zeros((p.shape[0], 3, 3), device=p.device)
    r_cross[:, 0, 1] = -r[:, 2]
    r_cross[:, 0, 2] = r[:, 1]
    r_cross[:, 1, 0] = r[:, 2]
    r_cross[:, 1, 2] = -r[:, 0]
    r_cross[:, 2, 0] = -r[:, 1]
    r_cross[:, 2, 1] = r[:, 0]
    
    # 小的 epsilon 值，防止除以 0
    epsilon = torch.tensor(1e-12, device=p.device).unsqueeze(0).repeat(p.shape[0], 1)

    denominator = (torch.ones([p.shape[0], 1], device=p.device) + p_dot_q + epsilon).unsqueeze(2)
    # print(f"denominator: {denominator.shape}")
    # 计算旋转矩阵
    rotation_matrix = I + r_cross + (torch.bmm(r_cross, r_cross)) / denominator
    
    return rotation_matrix

# Function Ψ(p, q) to compute the rotation matrix that aligns vector p with vector q
def psi(p, q):
    p = p / np.linalg.norm(p)
    q = q / np.linalg.norm(q)
    r = np.cross(p, q)  # Compute the cross product of p and q
    r_norm_sq = np.dot(r, r)  # Compute the norm squared of r
    p_dot_q = np.dot(p, q)  # Compute the dot product of p and q
    
    # Compute the rotation matrix
    I = np.eye(3)  # 3x3 identity matrix
    r_cross = np.array([
        [0, -r[2], r[1]],
        [r[2], 0, -r[0]],
        [-r[1], r[0], 0]
    ])  # Skew-symmetric matrix of r
    epsilon = 1e-8
    rotation_matrix = I + r_cross + (r_cross @ r_cross) / (1 + p_dot_q+epsilon)
    
    return rotation_matrix


if __name__=="__main__":
    # Example usage
    p=np.array([0,0,1])
    q=np.array([8, 0, 3])
    p = np.array(p, dtype=np.float32)  # 确保 p 是 float32 类型
    q = np.array(q, dtype=np.float32) 
    Rc=psi(p,q)

    Rc_inv = np.linalg.inv(Rc) 
    print("Rc_inv", Rc_inv*q)
    print("psi(q,p)",psi(q,p)*q)    
    print(Rc @ Rc_inv)
    print(Rc @ psi(q, p))
    print(np.allclose(Rc @ Rc_inv, np.eye(3)))
    print(np.allclose(Rc @ psi(q, p), np.eye(3)))