import os
import random
import sys

sys.path.append('/home/mendax/project/CLIPPose')
import json
import cv2
import torch
import numpy as np
from PIL import Image
from clippose import load
from lib.find_refer import CLIPPoseSimilarity
from lib.matcher import extract_and_match, extract_and_match_parallel
from lib.pose_estimation2d2d import pose_estimation_2d2d
from utils.visiblize import plot_3d_bounding_boxes,plot_3d_axes_on_images, plot_3d_axes_on_images_numpy,display_top_m_refer_for_all_images_numpy, plot_difference
from torchvision import transforms
from scipy.spatial.transform import Rotation as R
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
from lightglue import viz2d
from lightglue.utils import load_image
from lib.depth_estiamtion import Depth_Net
from sklearn.cluster import DBSCAN

class PoseEstimation:
    def __init__(self, checkpoint_path = "checkpoint/model_10.pt", obj_id = 1):
        self.checkpoint_path = checkpoint_path
        self.depth_net = Depth_Net()
        self.depth_net.load_state_dict(torch.load(f"/home/mendax/project/CLIPPose/checkpoint/depth_{obj_id}.pth"))
        self.depth_net.eval()
        self.radius = 500
        refer_dirs = f"/home/mendax/project/Datasets/space_station/train/socket/train_pbr/000003/view_000"
        refer_file = f"{refer_dirs}/view_000_info.json"
        self.refer_Rs = []
        self.refer_ts = []
        self.refer_images = []
        self.refer_xyxys = []
        self.refer_Ks = []
        self.refer_R_relatives = []
        with open(refer_file, 'r') as f:
            gt = json.load(f)

        for data in gt:
            # 直接使用 idx 来索引
            self.refer_Rs.append(np.array(data["R"], dtype=np.float32))  # 将 R_relative 转为 tensor
            self.refer_ts.append(np.array(data["t"], dtype=np.float32))  # 将 R_relative 转为 tensor
            self.refer_xyxys.append(np.array(data["bbox"], dtype=np.float32))  # 将 R_relative 转为 tensor
            self.refer_images.append(Image.open(f"{refer_dirs}/crop/{str(data['img_id']).zfill(6)}_{str(data['idx']).zfill(6)}.png"))       # 存储 refer_feature（512维）
            self.refer_Ks.append(np.array(data["Kc"], dtype=np.float32))  # 将 R_relative 转为 tensor
            self.refer_R_relatives.append(np.array(data["R_relative"], dtype=np.float32))  # 将 R_relative 转为 tensor
        # 将 refer_items 和 refer_imgs 转换为合适的张量，并迁移到 GPU
        # self.refer_Rs = torch.stack(self.refer_Rs).cuda()  # [batchsize, m, 6]
        self.clippose_similarity = CLIPPoseSimilarity(checkpoint_path=checkpoint_path)
        self.refer_K = np.array([[572.4114, 0.0, 325.2611],
                        [0.0, 573.57043, 242.04899],
                        [0.0, 0.0, 1.0]])

    def visiblize(self, test_images, imgs_refer, R_gt, refer_R, R_pred, uvs_pred, uvs_gt, similaritys):
        display_top_m_refer_for_all_images_numpy(test_images, imgs_refer,similaritys)
        plot_3d_axes_on_images_numpy(
            test_images, imgs_refer, R_gt, refer_R, R_pred, uvs_pred, uvs_gt, axis_length=125
        )
    
    def visiblize_3d(self, images, R_gt, R_final_pred, t_pred, t_gt, K, dimensions):
        plot_3d_bounding_boxes(images, R_gt, R_final_pred, t_pred, t_gt, K, dimensions)
        
        


    

    def average_rotation(self, R_preds, successful_matches):
        """
        对一组旋转矩阵 (R_preds) 进行角度平均。
        R_preds 是一个包含多个旋转矩阵（每个形状为 [3, 3]）的 numpy 数组列表。
        返回计算得到的平均旋转矩阵（numpy 数组）。
        """
        # 将旋转矩阵转换为四元数
        quaternions = []
        for R_mat in R_preds:
            # 使用 scipy.spatial.transform 中的 Rotation 类将旋转矩阵转换为四元数
            rot = R.from_matrix(R_mat)  # 转换为 scipy 的 Rotation 对象
            quaternions.append(rot.as_quat())  # 获取四元数

        # 将四元数列表转为 numpy 数组
        quaternions = np.array(quaternions)
        print(f"successful_matches: {successful_matches}")
        weights = np.array(successful_matches)
        weights = weights / np.sum(weights)  # 归一化权重
        # 计算四元数的平均
        mean_quaternion = self.quat_wavg_markley(quaternions, weights)


        # 将平均四元数转换回旋转矩阵
        mean_rotation = R.from_quat(mean_quaternion).as_matrix()

        return mean_rotation  # 返回 numpy 数组

    def quat_wavg_markley(self, Q, weights):
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

        
    def average_transcation(self, t_preds, successful_matches):
        t_preds = np.array(t_preds)
        weights = np.array(successful_matches)
        
        # 排除 NaN 值：只选择有效的 t_preds 和对应的权重
        valid_mask = np.all(np.isfinite(t_preds), axis=1)  # 检查 t_preds 中是否有 NaN 或 Inf
        
        # 使用有效的预测值和权重进行计算
        t_preds_valid = t_preds[valid_mask]
        weights_valid = weights[valid_mask]
        
        # 归一化有效权重
        weights_valid = weights_valid / np.sum(weights_valid) if np.sum(weights_valid) > 0 else 1  # 防止权重和为零
        
        # 计算加权平均值
        t_pred = np.sum(t_preds_valid.T * weights_valid, axis=1)
        
        return t_pred
        
    def average_uv(self, uv_preds, successful_matches):
        uv_preds = np.array(uv_preds)
        weights = np.array(successful_matches)
        weights = weights / np.sum(weights)  # 归一化权重
        uv_preds = np.sum(uv_preds.T * weights, axis=1)
        return uv_preds
    def visualize_depth(self, center_uv, center_depth, depth_image):
        """
        可视化深度图并标注中心点的深度值。
        
        :param center_uv: 中心点的 UV 坐标 (u, v)
        :param center_depth: 中心点的深度值
        :param depth_image: 深度图 (numpy 数组)
        """
        # 创建一个图像
        plt.figure(figsize=(8, 6))
        plt.imshow(depth_image, cmap='viridis')  # 使用色图 'viridis' 可视化深度图
        plt.colorbar(label='Depth Value')  # 添加颜色条

        # 标注中心点
        u, v = int(center_uv[0]), int(center_uv[1])
        plt.scatter(u, v, color='red', label=f'Center Depth: {center_depth:.2f}')
        plt.text(u, v, f'{center_depth:.2f}', color='white', fontsize=10, ha='left', va='bottom')

        # 添加标题和标签
        plt.title('Depth Image with Center Depth Highlighted')
        plt.xlabel('U (Horizontal)')
        plt.ylabel('V (Vertical)')
        plt.legend()

        # 显示图像
        plt.savefig("depth.png")
    def depth_estimation(self, R, depth_image, center_uv, keypoints):
        # 计算关键点相对于中心的偏移量
        R = torch.tensor(R, dtype=torch.float32).flatten()
        # 估算深度的初始化：直接从深度图中获取关键点的深度值
        keypoints = np.vstack([keypoints, center_uv]) 
        depths = np.array([depth_image[int(v), int(u)] for u, v in keypoints])  # 获取关键点对应的深度

        # 使用DBSCAN聚类，筛选可信深度值
        depths_reshaped = depths.reshape(-1, 1)  # DBSCAN需要二维输入
        clustering = DBSCAN(eps=0.5, min_samples=3).fit(depths_reshaped)  # 调整eps和min_samples以适应数据
        labels = clustering.labels_

        # 筛选出标签不为-1（噪声）的深度值
        valid_depths = depths[labels != -1]
        valid_keypoints = keypoints[labels != -1]

        if len(valid_depths) > 0:
            # 如果存在可信的深度值，计算其平均值作为可信深度集合 Z'
            distances = np.linalg.norm(valid_keypoints - center_uv, axis=1)  # 到中心点的距离
            weights = 1.0 / (distances + 1e-5)  # 加权距离的倒数
            weights_normalized = weights / np.sum(weights)
            z_0 = np.sum(valid_depths * weights_normalized)  # 使用可信深度值加权平均计算粗略深度估计
        else:
            # 如果没有可信的深度值，返回默认值（如 NaN 或 0）
            z_0 = np.nan

        # 检查中心点深度是否异常
        center_depth = depth_image[int(center_uv[1]), int(center_uv[0])]
        self.visualize_depth(center_uv, center_depth, depth_image)
        if center_depth not in valid_depths:
            # 如果中心深度异常，使用 z_0 替代
            depth_pred = self.depth_net(R) * self.radius + z_0*10
        else:
            # 如果中心深度正常，直接使用中心深度
            print(f"Center depth: {center_depth}")
            output = self.depth_net(R) 
            print(f"Output: {output}")
            depth_pred = output* self.radius + center_depth*10

        # 返回估算的深度
        return depth_pred.numpy()
    
    def estimation(self, test_image, depth_image, xyxy, uv_pred, K, refer_num = 8):
        print(f"xyxy: {xyxy}")
        depth_scale = 0.1
        xyxy = np.array(xyxy)  # 确保 xyxy 是 NumPy 数组
        # 转换为整数类型
        xyxy = xyxy.astype(int)
        test_image_array = np.array(test_image)
        depth_image_array = np.array(depth_image)*depth_scale
        cropped_image = test_image_array[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
        # 使用 OpenCV 保存图像
        cropped_image_pil = Image.fromarray(cropped_image)
        print(f"xyxy: {xyxy}")
        cropped_image_pil.save('cropped_image.png')
        top_m_indices, similaritys = self.clippose_similarity.calculate_similarity(refer_num, self.refer_images, [cropped_image_pil])        
        print(f"Top {refer_num} similar references indices:\n{top_m_indices}")
        indices = top_m_indices[0].cpu().numpy()
        print(f"Top {refer_num} similar references indices:\n{indices}")
        similaritys = similaritys[0].cpu().numpy()
        print(f"Top {refer_num} similar references similarity:\n{similaritys}")
        refers = [self.refer_images[i] for i in indices]
        R_refer = [self.refer_Rs[i] for i in indices]
        R_relative_refer = [self.refer_R_relatives[i] for i in indices]
        xyxy_refer = [self.refer_xyxys[i] for i in indices]
        K_refer = [self.refer_Ks[i] for i in indices]
        filtered_matchers, raw_results = extract_and_match_parallel(cropped_image_pil, refers)
        R_preds, t_preds = [], []
        successful_matches = []
        # 估计相机姿态
        for idx, (keypoints1, keypoints2, matches) in enumerate(filtered_matchers):
            # print(f"keypoints1: {keypoints1}")
            if matches.shape[0] > 8:
                successful_matches.append(matches.shape[0])
                print(f"xyxy: {xyxy}")
                print(f"xyxy_refer: {xyxy_refer[idx]}")
                keypoints1[:,0] = keypoints1[:,0] + xyxy[0]
                keypoints1[:,1] = keypoints1[:,1] + xyxy[1]
                # rawkeypoints1[:,0] = rawkeypoints1[:,0] + xyxy[0]
                # rawkeypoints1[:,1] = rawkeypoints1[:,1] + xyxy[1]
                image0 = load_image(test_image)[[2, 1, 0], :, :]
                image1 = load_image(refers[idx])[[2, 1, 0], :, :]
                print(f"image0: {image0.shape}")  
                print(f"image1: {image1.shape}") 
                viz2d.plot_images([image0, image1])
                # viz2d.plot_matches(rawkeypoints1, rawkeypoints2, color="lime", lw=0.2)
                viz2d.plot_matches(keypoints1, keypoints2, color="red", lw=0.3)
                plt.savefig(f"zzzz.png")
                keypoints2[:,0] = keypoints2[:,0] + xyxy_refer[idx][0]
                keypoints2[:,1] = keypoints2[:,1] + xyxy_refer[idx][1]
                # print(f"keypoints1: {keypoints1}")
                # print(f"keypoints2: {keypoints2}")
                # R_pred, t_pred = pose_estimation_2d2d(keypoints1, keypoints2, K, K_refer[idx])
                R_pred, _ = pose_estimation_2d2d(keypoints2, keypoints1, K_refer[idx], K)
                print(f'self.refer_ts[idx]: {self.refer_ts[idx]}')
                print(f"R_pred:{R_pred}")
                R_pred = R_pred @ R_refer[idx]
                z_pred = self.depth_estimation(R_pred, depth_image_array, uv_pred , keypoints1).item()
                print(f"z_pred: {z_pred}")
                fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                print(f"fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}")
                print(f"uv_pred: {uv_pred}")
                x_pred = (uv_pred[0] - cx) * z_pred / fx
                y_pred = (uv_pred[1] - cy) * z_pred / fy
                t_pred = np.array([x_pred, y_pred, z_pred])
                print(f"t_pred: {t_pred}")

            else:
                successful_matches.append(0)
                R_pred, t_pred = np.eye(3), np.zeros(3)

            R_preds.append(R_pred)
            t_preds.append(t_pred)
            print(f"t_pred: {t_pred}")
        print(f"t_preds: {t_preds}")
        R_pred = self.average_rotation(R_preds,successful_matches)
        t_pred = self.average_transcation(t_preds,successful_matches)
 
        return R_pred, t_pred, R_refer, refers, similaritys


def main():
    # 初始化 PoseEstimation 对象，加载模型和参考数据
    checkpoint_path = "checkpoint/socket_model_20.pt"  # 这里替换为实际的 checkpoint 路径
    obj_id = 1  # 设置物体 ID
    pose_estimator = PoseEstimation(checkpoint_path=checkpoint_path, obj_id=obj_id)

    img_num = 60
    test_scene_ids = list(range(0,2))
    target_dirs = [f'/home/mendax/project/Datasets/space_station/test/socket/train_pbr/{str(id).zfill(6)}' for id in test_scene_ids]
    # target_dirs = ['/home/mendax/project/Datasets/space_station/test/socket/train_pbr/000000']
    items = []
 
    for dir in target_dirs:
        gt_file = f'{dir}/view_000/view_000_info.json'
        with open(gt_file, 'r') as f:
            gt = json.load(f)
        for data in gt:
            if data["obj_id"] == obj_id:
                items.append((dir, data))
    # selected_data = random.sample(items, img_num)  # 从 gt 中随机选择 img_num 个元素
    selected_data = items[:img_num]  # 从 gt 中随机选择 img_num 个元素
    test_images = []
    Rs_gt = []
    Rs_pred = []
    Rs_refer = []
    ts_gt = []
    ts_pred = []
    uvs_gt = []
    uvs_pred = []
    similaritys = []

    imgs_refer = []
    for (img_dir, item) in selected_data:
        image_path = f"{img_dir}/rgb/{str(item['img_id']).zfill(6)}.png"
        depth_path = f"{img_dir}/depth/{str(item['img_id']).zfill(6)}.png"
        image = Image.open(image_path)
        depth_img = Image.open(depth_path)
        R = np.array(item["R"], dtype=np.float32)
        t = np.array(item["t"], dtype=np.float32)
        xyxy = np.array(item["bbox_gt"], dtype=np.float32)
        xyxy = np.clip(xyxy, a_min=0, a_max=None)
        uv_gt = np.array(item["uv"], dtype=np.float32)
        Kc = np.array(item["Kc"], dtype=np.float32)
        K = Kc
        fx, fy, cx, cy = Kc[0, 0], Kc[1, 1], Kc[0, 2], Kc[1, 2]
        test_images.append(image)
        R_avg, t_avg, R_refer,img_refer,similarity = pose_estimator.estimation(image, depth_img, xyxy, uv_gt, Kc)
        t_x,t_y,t_z = t_avg[0],t_avg[1],t_avg[2]
        uv_pred = np.array([fx * (t_x / t_z) + cx, fy * (t_y / t_z) + cy], dtype=np.float32)
        Rs_gt.append(R)
        Rs_pred.append(R_avg)
        Rs_refer.append(R_refer)
        ts_gt.append(t)
        ts_pred.append(t_avg)
        uvs_gt.append(uv_gt)
        uvs_pred.append(uv_pred)
        imgs_refer.append(img_refer)
        similaritys.append(similarity)

        # 输出结果
        print("Estimated Rotation Matrix (R_avg):")
        print(R_avg)
        print("\nGround Truth Rotation Matrix (R):")
        print(R)
        
        print("\nEstimated Translation Vector (t_avg):")
        print(t_avg)
        print("\nGround Truth Translation Vector (t):")
        print(t)

        print("\nEstimated UV Coordinates (uv_pred):")
        print(uv_pred)
        print("\nGround Truth UV Coordinates (uv_gt):")
        print(uv_gt)

    # dimensions = [300,400,300]
    # pose_estimator.visiblize_3d(test_images, Rs_gt, Rs_pred, ts_pred, ts_gt, K, dimensions)
    # pose_estimator.visiblize(test_images, imgs_refer, Rs_gt, Rs_refer, Rs_pred, uvs_pred, uvs_gt,similaritys)
    plot_difference(Rs_gt, Rs_pred, ts_pred, ts_gt)

if __name__ == "__main__":
    main()

