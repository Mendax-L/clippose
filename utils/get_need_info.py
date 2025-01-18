import json
import numpy as np
import os
import warnings
import random


# script_dir = os.path.dirname(os.path.abspath(__file__))
# os.chdir(script_dir)

# 确认当前工作目录
print(f"当前工作目录: {os.getcwd()}")
with open(f'../Datasets/ycbv/ycbv/test_targets_bop19.json') as f:
    test_targets = json.load(f)
# 初始化一个空字典
grouped_by_im_id = {}

# 遍历 test_targets 列表
for entry in test_targets:
    im_id = str(entry["im_id"])
    obj_id = entry["obj_id"]
    
    # 如果 im_id 不在字典中，初始化为一个空列表
    if im_id not in grouped_by_im_id:
        grouped_by_im_id[im_id] = []
    
    # 将 obj_id 添加到对应的 im_id 列表中
    grouped_by_im_id[im_id].append(obj_id)
# print(grouped_by_im_id)
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

def R2euler(R: np.ndarray) -> np.ndarray:
    """
    Converts a 3x3 rotation matrix to Euler angles (Yaw, Pitch, Roll) in ZYX order.
    
    Parameters
    ----------
    R : np.ndarray
        A 3x3 rotation matrix.
    
    Returns
    -------
    np.ndarray
        A numpy array of Euler angles [yaw, pitch, roll] in radians.
    """
    assert R.shape == (3, 3), "Rotation matrix must be 3x3."
    
    # Yaw (ψ) = atan2(R21, R11)
    yaw = np.arctan2(R[1, 0], R[0, 0])
    
    # Pitch (θ) = atan2(-R31, sqrt(R32^2 + R33^2))
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    
    # Roll (φ) = atan2(R32, R33)
    roll = np.arctan2(R[2, 1], R[2, 2])
    
    angles = np.degrees(np.array([yaw, pitch, roll]))



    # Map angles to the specified ranges
    yaw = np.mod(angles[0], 360)/10  # [0, 359]
    pitch = np.mod(angles[1], 360)/10  # [360, 319]
    roll = np.mod(angles[2], 360)/10  # [720, 1080]

    yaw = int(np.floor(yaw))
    pitch = int(np.floor(pitch))
    roll = int(np.floor(roll))
    
    # Adjust for negative values
    if pitch < 36:
        pitch += 36
    if roll < 72:
        roll += 72

    return np.array([yaw, pitch, roll])


def generate_view_info(target_dir, model_dir = f"../Datasets/ycbv/models/", img_w=640, img_h=480, scene_id=0):

    # 读取 scene_camera.json、scene_gt.json 和 scene_gt_info.json 文件
    with open(f'{target_dir}/scene_camera.json') as f_cam, \
        open(f'{target_dir}/scene_gt.json') as f_gt, \
        open(f'{target_dir}/scene_gt_info.json') as f_info: 
        # open(f'{model_dir}/models_info.json') as models_info
        

        scene_camera = json.load(f_cam)
        scene_gt = json.load(f_gt)
        scene_gt_info = json.load(f_info)
        # models = json.load(models_info)
        
    # 初始化结果
    results = []

    # 图像的宽度和高度
    img_w = 640
    img_h = 480

    # 遍历所有编号的视图
    R0 = np.zeros((3, 3))  # 3x3零矩阵
    t0 = np.zeros((3, 1))  # 3x1零向量
    for imgidx, (img_id, objs) in enumerate(scene_gt.items()):
        # if img_id not in grouped_by_im_id:
        #     continue
        for idx, obj in enumerate(objs):
            obj_id = obj['obj_id']
            # if obj_id not in grouped_by_im_id[img_id]:
            #     continue
            img_id = str(img_id)  # 确保键是字符串
            cam_data = scene_camera[img_id]
            gt_data = scene_gt[img_id][idx]  # 假设每个视图中只有一个物体
            gt_info = scene_gt_info[img_id][idx]  # 获取 bbox 信息
            # 获取相机内参
            Kc = np.array(cam_data['cam_K']).reshape(3, 3)
            fx, fy = Kc[0, 0], Kc[1, 1]
            cx, cy = Kc[0, 2], Kc[1, 2]
            Kc_inv = np.linalg.inv(Kc)

            # 获取平移向量 cam_t_m2c 和物体的 ID
            if np.array(gt_data['obj_id'])!= obj_id:
                raise IndexError("obj_id 不匹配")
            
            if img_id == '1' or imgidx == 0:
                R0 = np.array(gt_data['cam_R_m2c']).reshape(3, 3)
                t0 = np.array(gt_data['cam_t_m2c']).reshape(3, 1)
            elif np.array_equal(R0, np.zeros((3, 3))) or np.array_equal(t0, np.zeros((3, 1))):
                warnings.warn("R0 or t0 is still a zero matrix. Please check the input data.")
            
            Rk = np.array(gt_data['cam_R_m2c']).reshape(3, 3)
            tk = np.array(gt_data['cam_t_m2c']).reshape(3, 1)
            tk = tk.reshape(3, 1) 

            Rc = np.linalg.inv(Rk) @ R0
            # tc = t0 - np.linalg.inv(Rc) @ tk
            tc = tk - Rc@t0



            R = Rk.tolist()
            t = tk.flatten().tolist()
            
            # diameter = models[str(obj_id)]['diameter']
            # r = diameter / 2 

            bbox_gt = gt_info['bbox_obj']
            print(f"bbox_gt: {bbox_gt}")
            bbox_gt_w, bbox_gt_h = bbox_gt[2], bbox_gt[3]
            bbox_gt = [bbox_gt[0], bbox_gt[1], bbox_gt[0] + bbox_gt_w, bbox_gt[1] + bbox_gt_h]


            t_x,t_y,t_z = t[0],t[1],t[2]


            # # 计算物体中心在图像中的投影坐标
            uv = (fx * (t_x / t_z) + cx, fy * (t_y / t_z) + cy)
            
            # # 获取 bbox_obj 的 xyxy 信息
            
            # cors = [[t_x-r,t_y-r,t_z],[t_x+r,t_y+r,t_z]]
            # bbox = []
            # for i, cor in enumerate(cors):
            # # print(i)
            #     bbox.append(int(cor[0]*fx/cor[2] + cx)) 
            #     bbox.append(int(cor[1]*fy/cor[2] + cy))
            # print(bbox)


            half_edge = max(bbox_gt_w, bbox_gt_h)/2
            print(f"half_edge: {half_edge}")
            bbox =[uv[0]-half_edge,uv[1]-half_edge,uv[0]+half_edge,uv[1]+half_edge]
            print(f"bbox: {bbox}")
            bbox_w, bbox_h = bbox[2]-bbox[0], bbox[3]-bbox[1]


            uv_relative = ((uv[0] - bbox[0]) / bbox_w, (uv[1] - bbox[1]) / bbox_h)



                    
            centeruv_3d = np.array([uv[0],uv[1],1])
            p = np.array([0,0,1])
            q = Kc_inv @ centeruv_3d

            Rc=psi(q, p)

            R_relative = Rc @ Rk

            rpy_relative = R2euler(R_relative)  


            # 检查投影点是否在图像范围内（可选）
            if 0 <= uv[0] < img_w and 0 <= uv[1] < img_h:
                obj_info = {
                    'img_id':int(img_id),
                    'idx':idx,                
                    'obj_id': obj_id,
                    'uv': uv,
                    'uv_relative': uv_relative,
                    'bbox': bbox,
                    'bbox_gt': bbox_gt,
                    'R': R,
                    't': t,
                    'R_relative': R_relative.tolist(),
                    'rpy_relative' : rpy_relative.tolist(),
                    'scene_id': scene_id,
                    # 'keypoints':points,
                    # 'diameter': diameter,
                    'Kc': Kc.tolist(),
                    'Kc_inv': Kc_inv.tolist()
                }
            else:
                obj_info = {
                    'img_id':int(img_id),
                    'idx':idx,                
                    'obj_id': obj_id,
                    'uv': uv,
                    'uv_relative': uv_relative,
                    'bbox': bbox,
                    'bbox_gt': bbox_gt,
                    'R': R,
                    't': t,
                    'R_relative': R_relative.tolist(),
                    'rpy_relative': rpy_relative.tolist(),
                    'scene_id': scene_id,
                    # 'keypoints':points,
                    # 'diameter': diameter,
                    'Kc':Kc.tolist(),
                    'Kc_inv': Kc_inv.tolist(),
                    'note': 'center outside image bounds'
                }

            # 将结果存入字典
            results.append(obj_info)

    # 将结果保存为 JSON 文件
    view_dir_path = os.path.join(target_dir, f'view_000')
    os.makedirs(view_dir_path, exist_ok=True)  
    with open(f'{view_dir_path}/view_000_info.json', 'w') as f_out:
        json.dump(results, f_out, indent=4)

    print(f"新视角物体位姿相关信息已成功计算并保存至 {view_dir_path}/view_0000_info.json")

if __name__ == '__main__':
        

    model_dir = f"../Datasets/space_station/models/"
    for scene_id in range(0, 4):
        target_dir = f'../Datasets/space_station/test/socket/train_pbr/{str(scene_id).zfill(6)}'
        if os.path.exists(target_dir):
            generate_view_info(target_dir = target_dir, model_dir = model_dir, scene_id = scene_id)
        else:
            print(f"{target_dir} does not exist.")
