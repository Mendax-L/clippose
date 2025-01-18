import json
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import PIL.Image as Image
import torchvision.transforms as transforms
import clippose


class Select_Refer_Dataset(Dataset):
    def __init__(self,preprocess, obj_id, target_dirs, refer_dirs, data_fraction=0.1):

        self.obj_id = obj_id
        self.preprocess = preprocess
        self.rgb_dirs = [f'{dir}/crop'for dir in target_dirs]
        self.gt_files = [f'{dir}/view_000_info.json'for dir in target_dirs]
        self.items = []
        for rgb_dir, gt_file in zip(self.rgb_dirs, self.gt_files):
            gt = self._load_gt(gt_file)
            for data in gt:
                if data["obj_id"] == self.obj_id:
                    self.items.append((rgb_dir, data))
        if data_fraction < 1.0:
            self.items = random.sample(self.items, int(len(self.items) * data_fraction))

        self.refer_dirs = [f'{dir}/crop'for dir in refer_dirs]
        self.refer_gt_files = [f'{dir}/view_000_info.json'for dir in refer_dirs]
        self.refer_items = []

        for rgb_dir, gt_file in zip(self.refer_dirs, self.refer_gt_files):
            gt = self._load_gt(gt_file)
            for data in gt:
                if data["obj_id"] == self.obj_id:
                    self.refer_items.append((rgb_dir, data))
        
                    
    # 加载中心点数据
    def _load_gt(self, gt_file):
        with open(gt_file, 'r') as f:
            gt = json.load(f)
        return gt
    
    def __len__(self):
        return len(self.items)
    
    def trans(self, image):
        # 放缩：随机裁剪并缩放
        resize_transform = transforms.RandomResizedCrop(size=(image.shape[2], image.shape[1]), scale=(0.8, 1.2))
        # 旋转：随机旋转
        rotate_transform = transforms.RandomRotation(degrees=(-30, 30))
        # 遮挡：随机遮挡区域
        erase_transform = transforms.RandomErasing(p=0.5, scale=(0.02, 0.35), ratio=(0.3, 3.3))

        image = resize_transform(image)
        image = rotate_transform(image)
        image = erase_transform(image)

        return image


    def __getitem__(self, idx):
        rgb_dir, item = self.items[idx]
        image_path = f"{rgb_dir}/{str(item['img_id']).zfill(6)}_{str(item['idx']).zfill(6)}.png"
        image = Image.open(image_path)
        image = self.preprocess(image) # Image from PIL module
        R_i = torch.tensor(item["R_relative"], dtype=torch.float32)
        
        refer_dir, refer_item = random.choice(self.refer_items)
        refer_path = f"{refer_dir}/{str(refer_item['img_id']).zfill(6)}_{str(refer_item['idx']).zfill(6)}.png"
        refer = Image.open(refer_path)
        refer = self.preprocess(refer) # Image from PIL module
        refer = self.trans(refer)
        R_r = torch.tensor(refer_item["R_relative"], dtype=torch.float32)

        return image, refer, R_i, R_r
    

class Refine_Dataset(Dataset):
    def __init__(self, obj_id, target_dirs,data_fraction=0.1):

        self.obj_id = obj_id
        self.rgb_dirs = [f'{dir}/masked_crop'for dir in target_dirs]
        self.gt_files = [f'{dir}/view_000_info.json'for dir in target_dirs]
        self.items = []
        self.m = 10

        device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
        self.clip, self.preprocess = clippose.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training
        checkpoint = torch.load("checkpoint/model_20.pt")
        self.clip.load_state_dict(checkpoint['model_state_dict'])
        self.clip.cuda().train()

        for rgb_dir, gt_file in zip(self.rgb_dirs, self.gt_files):
            gt = self._load_gt(gt_file)
            for data in gt:
                if data["obj_id"] == self.obj_id:
                    self.items.append((rgb_dir, data))
        
        if data_fraction < 1.0:
            self.items = random.sample(self.items, int(len(self.items) * data_fraction))
        
        
        refer_dirs = f"/home/mendax/project/CLIPPose/refers/obj_{obj_id}"
        refer_file = f"{refer_dirs}/obj_{obj_id}.json"
        refer_Rs = []
        refer_imgs = []
        with open(refer_file, 'r') as f:
            refer = json.load(f)
            
        # 遍历 top_m_indices，直接处理张量
        for data in refer:
            # 直接使用 idx 来索引
            refer_Rs.append(torch.tensor(data["R_relative"]))  # 将 R_relative 转为 tensor
            image_path = f"{refer_dirs}/{str(data['image_name'])}"
            image = Image.open(image_path)
            refer_imgs.append(self.preprocess(image))       # 存储 refer_feature（512维）
        
        # 将 refer_items 和 refer_imgs 转换为合适的张量，并迁移到 GPU
        refer_imgs = torch.stack(refer_imgs).cuda()  # [batchsize, m, 512]
        
        self.refer_Rs = torch.stack(refer_Rs).cuda()  # [batchsize, m, 6]
        # with torch.no_grad():
        self.refer_features = self.clip.encode_image(refer_imgs).float()
        self.refer_features_norm = self.refer_features/self.refer_features.norm(dim=-1, keepdim=True)




    # 加载中心点数据
    def _load_gt(self, gt_file):
        with open(gt_file, 'r') as f:
            gt = json.load(f)
        return gt
    
    def __len__(self):
        return len(self.items)
    
    def trans(self, image):
        # 放缩：随机裁剪并缩放
        resize_transform = transforms.RandomResizedCrop(size=(image.shape[2], image.shape[1]), scale=(0.8, 1.2))
        # 旋转：随机旋转
        rotate_transform = transforms.RandomRotation(degrees=(-8, 8))
        # 遮挡：随机遮挡区域
        erase_transform = transforms.RandomErasing(p=0.5, scale=(0.02, 0.35), ratio=(0.3, 3.3))

        image = resize_transform(image)
        image = rotate_transform(image)
        image = erase_transform(image)

        return image


    def __getitem__(self, idx):
        rgb_dir, item = self.items[idx]
        image_path = f"{rgb_dir}/{str(item['img_id']).zfill(6)}_{str(item['idx']).zfill(6)}.png"
        image = Image.open(image_path)
        image_input = self.preprocess(image).cuda() # Image from PIL module
        image_input = image_input.unsqueeze(0)
        R_i = torch.tensor(item["R_relative"], dtype=torch.float32).cuda()
        # with torch.no_grad():
        image_feature = self.clip.encode_image(image_input).float()
        image_features_norm = image_feature/image_feature.norm(dim=-1, keepdim=True)
        similarity = image_features_norm @ self.refer_features_norm.T
        _, top_m_indices = torch.topk(similarity, self.m, dim=1, largest=True, sorted=True)
        # 根据 top_m_indices 获取对应的 refer_features 和 refer_Rs
        top_m_refer_features = self.refer_features[top_m_indices]  # 获取对应的 refer_features
        top_m_refer_Rs = self.refer_Rs[top_m_indices]  # 获取对应的 refer_Rs
        image_feature = image_feature.squeeze(0)
        top_m_refer_features = top_m_refer_features.squeeze(0)
        top_m_refer_Rs = top_m_refer_Rs.squeeze(0)

        return image_feature, top_m_refer_features, R_i, top_m_refer_Rs
    
class Depth_Dataset(Dataset):
    def __init__(self,obj_id, target_dirs, data_fraction=0.1):

        self.obj_id = obj_id
        self.img_dirs = [f'{dir}'for dir in target_dirs]
        self.gt_files = [f'{dir}/view_000/view_000_info.json'for dir in target_dirs]
        self.items = []
        for img_dir, gt_file in zip(self.img_dirs, self.gt_files):
            gt = self._load_gt(gt_file)
            for data in gt:
                if data["obj_id"] == self.obj_id:
                    self.items.append((img_dir, data))
        if data_fraction < 1.0:
            self.items = random.sample(self.items, int(len(self.items) * data_fraction))
        
                    
    # 加载中心点数据
    def _load_gt(self, gt_file):
        with open(gt_file, 'r') as f:
            gt = json.load(f)
        return gt
    
    def __len__(self):
        return len(self.items)
    
    def find_closest_visible_point(self,u, v, mask):
        """
        寻找给定坐标 (u, v) 周围最近的一个值为 1 的点。

        :param u: 当前点的 u 坐标
        :param v: 当前点的 v 坐标
        :param mask: 二值化的 mask 图像
        :return: 最近的可见点 (u, v)
        """
        # 获取 mask 中为 1 的所有点的坐标
        visible_points = np.column_stack(np.where(mask > 0))  # shape (n, 2)
        # print(f"Visible points: {visible_points}")
        # 计算每个可见点与 (u, v) 的欧几里得距离
        distances = np.linalg.norm(visible_points - np.array([v, u]), axis=1)

        k = 40
        # 排序并返回第 k 个最小距离对应的点
        sorted_indices = np.argsort(distances)  # 按照距离从小到大排序
        closest_points = visible_points[sorted_indices]  # 排序后的所有点
        closest_point_idx = sorted_indices[k-1]  # 获取第 k 个点（k-1 因为索引从 0 开始）
        
        return closest_points[k-1]

    def __getitem__(self, idx):
        img_dir, item = self.items[idx]
        depthimg_path = f"{img_dir}/depth/{str(item['img_id']).zfill(6)}.png"
        mask_path = f"{img_dir}/mask_visib/{str(item['img_id']).zfill(6)}_{str(item['idx']).zfill(6)}.png"  # 读取mask文件路径

        # 读取深度图像
        depthimg = cv2.imread(depthimg_path, cv2.IMREAD_UNCHANGED)
        if depthimg is None:
            raise FileNotFoundError(f"Depth image not found at {depthimg_path}")

        # 读取mask图像（假设是二值化的mask，0表示不可见，1表示可见）
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask image not found at {mask_path}")
        # print(f"Mask min: {mask.min()}, max: {mask.max()}")
        # 提取 uv 坐标
        uv = item["uv"]
        u, v = int(uv[0]), int(uv[1])

        # 检查坐标是否在图像范围内
        if u < 0 or v < 0 or u >= depthimg.shape[1] or v >= depthimg.shape[0]:
            raise ValueError(f"UV coordinates {uv} are out of bounds for image with shape {depthimg.shape}")

        # 检查该点是否在mask内（即mask值是否为1）
        if mask[v, u] == 0:
            v, u = self.find_closest_visible_point(u, v, mask)

        # 提取深度值
        depth_cu = depthimg[v, u]  # 获取深度值
        # print(f"Depth value at {uv}: {depth_cu}")
        # 转换为 tensor
        depth_cu = torch.tensor(depth_cu, dtype=torch.float32)

        # 获取旋转矩阵和平移向量
        R = torch.tensor(item["R_relative"], dtype=torch.float32).flatten()
        depth_gt = torch.tensor(item["t"][2], dtype=torch.float32)

        return R, depth_cu, depth_gt
