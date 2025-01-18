import random
import cv2
import numpy as np
import torch
from pkg_resources import packaging

print("Torch version:", torch.__version__)
import sys
sys.path.append('/home/mendax/project/CLIPPose')
# print(sys.path)

import clippose
from clippose.refine_batch import RotationAttention, RotationAvg

clippose.available_models()

device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
model, preprocess = clippose.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training
checkpoint = torch.load("checkpoint/model_20.pt")

# # Use these 3 lines if you use default model setting(not training setting) of the clip. For example, if you set context_length to 100 since your string is very long during training, then assign 100 to checkpoint['model_state_dict']["context_length"] 
# checkpoint['model_state_dict']["input_resolution"] = model.input_resolution #default is 224
# checkpoint['model_state_dict']["context_length"] = model.context_length # default is 77
# checkpoint['model_state_dict']["vocab_size"] = model.vocab_size 

model.load_state_dict(checkpoint['model_state_dict'])
model.cuda().eval()
refine_model = RotationAttention().to(device)
refine_checkpoint = torch.load("checkpoint/refine_20.pt")
refine_model.load_state_dict(refine_checkpoint['model_state_dict'])
refine_model.cuda().eval()
import os
import IPython.display
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from collections import OrderedDict
import torch
import json



original_images = []
original_refers = []
images = []
refers = []
R_gt = []
obj = 1
refer_dirs = f"/home/mendax/project/CLIPPose/refers/obj_{obj}"
test_dirs = f"/home/mendax/project/CLIPPose/refers/test_obj_{obj}"
test_file = f"/home/mendax/project/CLIPPose/refers/test_obj_{obj}/obj_{obj}.json"


plt.figure(figsize=(16, 5))
for filename in [filename for filename in os.listdir(refer_dirs) if filename.endswith(".png") or filename.endswith(".jpg")]:

    refer = Image.open(os.path.join(refer_dirs, filename)).convert("RGB")
    original_refers.append(refer)
    refers.append(preprocess(refer))
  

img_num = 4 


# 获取对应的图片和 R_relative
items = []
original_images = []
images = []
R_gt = []
test_scene_ids = list(range(48,60))
target_dirs = [f'../Datasets/ycbv/test/{str(id).zfill(6)}/view_000/' for id in test_scene_ids]
rgb_dirs = [f'{dir}/crop'for dir in target_dirs]
gt_files = [f'{dir}/view_000_info.json'for dir in target_dirs]
for rgb_dir, gt_file in zip(rgb_dirs, gt_files):
    with open(gt_file, 'r') as f:
        gt = json.load(f)
    for data in gt:
        if data["obj_id"] == obj:
           items.append((rgb_dir, data))
selected_data = random.sample(items, img_num)  # 从 gt 中随机选择 img_num 个元素
for (rgb_dir, item) in selected_data:
    image_path = f"{rgb_dir}/{str(item['img_id']).zfill(6)}_{str(item['idx']).zfill(6)}.png"
    image = Image.open(image_path)
    R = torch.tensor(item["R_relative"], dtype=torch.float32)
    image = image.resize((128, 128))
    original_images.append(image)
    images.append(preprocess(image))
    R_gt.append(R)

R_gt = torch.tensor(np.stack(R_gt)).cuda()


print(f"len of refers:{len(refers)}")
print(f"len of images:{len(images)}")

image_input = torch.tensor(np.stack(images)).cuda()
refers_input = torch.tensor(np.stack(refers)).cuda()
with torch.no_grad():
    image_features = model.encode_image(image_input).float()
    refer_features = model.encode_image(refers_input).float()

image_features_norm = image_features/image_features.norm(dim=-1, keepdim=True)
refer_features_norm = refer_features/refer_features.norm(dim=-1, keepdim=True)
similarity = image_features_norm @ refer_features_norm.T
print(f"similarity shape:{similarity.shape}")
m = 10  # 选择前5个相似的refer

_, top_m_indices = torch.topk(similarity, m, dim=1, largest=True, sorted=True)
print(f"top_m_indices:{top_m_indices}")

refer_file = f"{refer_dirs}/obj_{obj}.json"
refer_items = []
refer_imgs = []
with open(refer_file, 'r') as f:
    gt = json.load(f)
    
    # 遍历 top_m_indices，直接处理张量
    for row in top_m_indices:
        refer_item = []  # 存储当前 row 对应的 refer_items
        refer_img = []   # 存储当前 row 对应的 refer_imgs
        
        for idx in row:
            # 直接使用 idx 来索引
            refer_item.append(torch.tensor(gt[idx.item()]["R_relative"]))  # 将 R_relative 转为 tensor
            refer_img.append(refer_features[idx.item()])       # 存储 refer_feature（512维）
        
        # 将一个批次的 refer_items 和 refer_imgs 转换为张量并添加到列表
        refer_items.append(torch.stack(refer_item))  # 假设 refer_item 是 [m, 6] 的形状
        refer_imgs.append(torch.stack(refer_img))    # 假设 refer_img 是 [m, 512] 的形状

# 将 refer_items 和 refer_imgs 转换为合适的张量，并迁移到 GPU
refer_Rs = torch.stack(refer_items).cuda()  # [batchsize, m, 6]
refer_imgs = torch.stack(refer_imgs).cuda()  # [batchsize, m, 512]

print(f"image_features shape:{image_features.shape}")
print(f"refer_features shape:{refer_imgs.shape}")
print(f"refer_Rs shape:{refer_Rs.shape}")
with torch.no_grad():
    Rs = refine_model(image_features, refer_imgs, refer_Rs)

print(f"Rs shape:{Rs.shape}")
R_Avg = RotationAvg()

avg_Rs = R_Avg.average_rotation_matrices(Rs)
R_refer = R_Avg.average_rotation_matrices(refer_Rs)


def plot_3d_axes_on_images(images, Rs, R_gt, R_refer, k=4, axis_length=25):
    """
    批量绘制旋转后的三维坐标轴在多张图像上的投影，若R_gt不为None则绘制R_gt、Rs和R_refer的比较。
    :param images: 输入图像 (batch_size, H, W, 3)，通常是一个ndarray或tensor
    :param Rs: 旋转矩阵 (batch_size, 3, 3)
    :param R_gt: 真实旋转矩阵 (batch_size, 3, 3)，如果为None则只绘制Rs
    :param R_refer: 参考旋转矩阵 (batch_size, 3, 3)，要与R_gt和Rs一起比较
    :param axis_length: 坐标轴长度
    :return: 绘制坐标轴的图像
    """
    batch_size = images.shape[0]

    # 定义3个坐标轴的单位向量（X, Y, Z轴）
    axis_vectors = torch.tensor([[1.0, 0.0, 0.0],  # X轴
                                 [0.0, 1.0, 0.0],  # Y轴
                                 [0.0, 0.0, 1.0]])  # Z轴

    # 获取图像的大小 (height, width)
    H, W, _ = images.shape[1], images.shape[2], images.shape[3]
    origin = np.array([W // 2, H // 2])

    # 创建一个新的图像列表来保存所有处理过的图像
    processed_images_gt = []  # 用于存放R_gt的图像
    processed_images_pred = []  # 用于存放Rs的图像
    processed_images_refer = []  # 用于存放R_refer的图像

    fig, axes = plt.subplots(3, k, figsize=(k * 4, 12))  # 三行，每行k列
    if k == 1:
        axes = [axes]

    for i in range(batch_size):
        image = images[i].cpu().numpy()  # 获取当前图像
        R_pred = Rs[i].cpu().numpy()  # 获取当前预测旋转矩阵
        R_true = R_gt[i].cpu().numpy() if R_gt is not None else R_pred  # 使用R_gt进行比较
        R_ref = R_refer[i].cpu().numpy()  # 获取参考旋转矩阵

        # 将旋转矩阵应用到轴向量
        rotated_axes_pred = axis_vectors @ R_pred.T  # 旋转后的3个轴向量（预测）
        rotated_axes_true = axis_vectors @ R_true.T  # 旋转后的3个轴向量（真实）
        rotated_axes_ref = axis_vectors @ R_ref.T  # 旋转后的3个轴向量（参考）

        # 将旋转后的轴缩放到指定长度
        rotated_axes_pred = rotated_axes_pred * axis_length
        rotated_axes_true = rotated_axes_true * axis_length
        rotated_axes_ref = rotated_axes_ref * axis_length

        # 将旋转后的轴投影到2D平面 (假设是X-Y平面)
        projected_axes_pred = rotated_axes_pred[:, :2].numpy()
        projected_axes_true = rotated_axes_true[:, :2].numpy()
        projected_axes_ref = rotated_axes_ref[:, :2].numpy()

        # 创建图像副本以便在其上绘制
        img_copy = np.copy(image)
        img_copy = img_copy.astype(np.uint8)

        # 绘制坐标轴
        colors = [(0, 0, 255),  # 红色 -> X轴
                  (0, 255, 0),  # 绿色 -> Y轴
                  (255, 0, 0)]  # 蓝色 -> Z轴

        # 绘制真实的旋转矩阵（R_gt）
        for j, axis in enumerate(projected_axes_true):
            end_point = origin + axis
            img_copy = cv2.line(img_copy, tuple(origin), tuple(end_point.astype(int)), colors[j], 2)
            img_copy = cv2.circle(img_copy, tuple(end_point.astype(int)), 3, colors[j], -1)

        # 保存真实旋转矩阵（R_gt）的图像
        processed_images_gt.append(img_copy)

        # 创建一个新的图像副本用于绘制预测旋转矩阵（Rs）
        img_copy_pred = np.copy(image)
        img_copy_pred = img_copy_pred.astype(np.uint8)

        # 绘制预测的旋转矩阵（Rs）
        for j, axis in enumerate(projected_axes_pred):
            end_point = origin + axis
            img_copy_pred = cv2.line(img_copy_pred, tuple(origin), tuple(end_point.astype(int)), colors[j], 2)
            img_copy_pred = cv2.circle(img_copy_pred, tuple(end_point.astype(int)), 3, colors[j], -1)

        # 保存预测旋转矩阵（Rs）的图像
        processed_images_pred.append(img_copy_pred)

        # 创建一个新的图像副本用于绘制参考旋转矩阵（R_refer）
        img_copy_refer = np.copy(image)
        img_copy_refer = img_copy_refer.astype(np.uint8)

        # 绘制参考旋转矩阵（R_refer）
        for j, axis in enumerate(projected_axes_ref):
            end_point = origin + axis
            img_copy_refer = cv2.line(img_copy_refer, tuple(origin), tuple(end_point.astype(int)), colors[j], 2)
            img_copy_refer = cv2.circle(img_copy_refer, tuple(end_point.astype(int)), 3, colors[j], -1)

        # 保存参考旋转矩阵（R_refer）的图像
        processed_images_refer.append(img_copy_refer)

    # 将处理后的图像转换为Tensor形式
    processed_images_gt = torch.tensor(np.stack(processed_images_gt)).float()
    processed_images_pred = torch.tensor(np.stack(processed_images_pred)).float()
    processed_images_refer = torch.tensor(np.stack(processed_images_refer)).float()

    # 绘制图像
    for i in range(k):
        # 绘制真实旋转矩阵的图像
        axes[0][i].imshow(processed_images_gt[i].cpu().numpy().astype(np.uint8))
        axes[0][i].set_title(f"GT Image {i+1} with 3D axes")
        axes[0][i].axis('off')

        # 绘制预测旋转矩阵的图像
        axes[1][i].imshow(processed_images_pred[i].cpu().numpy().astype(np.uint8))
        axes[1][i].set_title(f"Pred Image {i+1} with 3D axes")
        axes[1][i].axis('off')

        # 绘制参考旋转矩阵的图像
        axes[2][i].imshow(processed_images_refer[i].cpu().numpy().astype(np.uint8))
        axes[2][i].set_title(f"Refer Image {i+1} with 3D axes")
        axes[2][i].axis('off')

    plt.tight_layout()
    plt.savefig("refer_images_with_gt_vs_pred_vs_refer_axes.png")
    plt.show()

    return processed_images_pred, processed_images_gt, processed_images_refer


def display_top_m_refer_for_all_images(original_images, original_refers, similarity, top_m_indices, m=5):
    """
    显示所有图像与最相似的 m 个 refer 图像
    :param original_images: 所有原始图像
    :param original_refers: 所有参考图像
    :param similarity: 所有图像与参考图像的相似度矩阵
    :param top_m_indices: 每个图像与参考图像的前 m 个相似度的索引
    :param m: 显示的最相似参考图像个数
    """
    num_images = similarity.shape[0]  # 获取图像的数量

    # 创建一个图形来显示所有图像
    plt.figure(figsize=(2*m, num_images))

    for image_idx in range(num_images):
        # 获取与当前图像最相似的 m 个参考图像的索引和相似度
        top_m_refer_indices = top_m_indices[image_idx][:m]
        top_m_similarities = similarity[image_idx][top_m_refer_indices]
        print(f"Displaying for image {image_idx}")
        print(f"Top {m} refer indices: {top_m_refer_indices}")
        print(f"Top {m} similarities: {top_m_similarities}")

        # 显示原始图像
        plt.subplot(num_images, m + 1, image_idx * (m + 1) + 1)
        plt.imshow(original_images[image_idx])
        plt.title(f"Original Image {image_idx}")
        plt.xticks([])  # 关闭x轴刻度
        plt.yticks([])  # 关闭y轴刻度

        # 显示与原始图像最相似的 m 个参考图像
        for i, refer_idx in enumerate(top_m_refer_indices):
            plt.subplot(num_images, m + 1, image_idx * (m + 1) + i + 2)
            plt.imshow(original_refers[refer_idx])
            plt.title(f"Refer {refer_idx}\n{top_m_similarities[i]:.3f}")
            plt.xticks([])  # 关闭x轴刻度
            plt.yticks([])  # 关闭y轴刻度

    # 调整布局，确保图像不重叠
    plt.tight_layout()

    # 保存并显示图像
    plt.savefig("result_all_images.png")
    plt.show()

display_top_m_refer_for_all_images(original_images, original_refers, similarity, top_m_indices, m)
imgs = torch.tensor(np.stack(original_images)).cuda()
print(imgs.shape)
plot_3d_axes_on_images(imgs, avg_Rs, R_gt, R_refer)
