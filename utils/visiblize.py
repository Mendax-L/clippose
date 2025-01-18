import json
import random
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2  # 用于在图像上绘制线条和圆点


def plot_3d_axes_on_images(images, R_gt, R_initial_pred, R_final_pred, axis_length=25):
    batch_size = images.shape[0]
    k = batch_size

    # 定义3个坐标轴的单位向量（X, Y, Z轴）
    axis_vectors = torch.tensor([[1.0, 0.0, 0.0],  # X轴
                                  [0.0, 1.0, 0.0],  # Y轴
                                  [0.0, 0.0, 1.0]])  # Z轴

    # 获取图像大小
    H, W = images.shape[1:3]
    origin = np.array([W // 2, H // 2])

    # 坐标轴颜色
    axis_colors = [(0, 0, 255),  # 红色 -> X轴
                   (0, 255, 0),  # 绿色 -> Y轴
                   (255, 0, 0)]  # 蓝色 -> Z轴

    # 创建图像列表
    processed_images_gt, processed_images_initial_pred, processed_images_final_pred = [], [], []

    # 图像显示设置
    fig, axes = plt.subplots(3, k, figsize=(k * 4, 12))
    if k == 1:
        axes = np.expand_dims(axes, axis=1)

    for i in range(batch_size):
        # 获取当前图像和旋转矩阵
        image = (images[i].cpu().numpy() * 255).astype(np.uint8)
        R_pred = R_final_pred[i].cpu().numpy()
        R_true = R_gt[i].cpu().numpy() if R_gt is not None else R_pred
        R_ref = R_initial_pred[i].cpu().numpy()

        # 应用旋转矩阵
        rotated_axes_pred = axis_vectors @ R_pred.T * axis_length
        rotated_axes_true = axis_vectors @ R_true.T * axis_length
        rotated_axes_ref = axis_vectors @ R_ref.T * axis_length

        # 投影到2D平面
        projected_axes_pred = rotated_axes_pred[:, :2].cpu().numpy()
        projected_axes_true = rotated_axes_true[:, :2].cpu().numpy()
        projected_axes_ref = rotated_axes_ref[:, :2].cpu().numpy()

        # 绘制真实旋转矩阵的轴
        img_copy = image.copy()
        for j, axis in enumerate(projected_axes_true):
            end_point = origin + axis.astype(int)
            img_copy = cv2.line(img_copy, tuple(origin), tuple(end_point), axis_colors[j], 2)
            img_copy = cv2.circle(img_copy, tuple(end_point), 3, axis_colors[j], -1)
        processed_images_gt.append(img_copy)

        # 绘制预测旋转矩阵的轴
        img_copy_refer = image.copy()
        for j, axis in enumerate(projected_axes_pred):
            end_point = origin + axis.astype(int)
            img_copy_pred = cv2.line(img_copy_pred, tuple(origin), tuple(end_point), axis_colors[j], 2)
            img_copy_pred = cv2.circle(img_copy_pred, tuple(end_point), 3, axis_colors[j], -1)
        processed_images_initial_pred.append(img_copy_pred)

        # 绘制参考旋转矩阵的轴
        img_copy_pred = image.copy()
        for j, axis in enumerate(projected_axes_ref):
            end_point = origin + axis.astype(int)
            img_copy_refer = cv2.line(img_copy_refer, tuple(origin), tuple(end_point), axis_colors[j], 2)
            img_copy_refer = cv2.circle(img_copy_refer, tuple(end_point), 3, axis_colors[j], -1)
        processed_images_final_pred.append(img_copy_refer)

    # 转换为Tensor
    processed_images_gt = torch.tensor(np.stack(processed_images_gt)).float()
    processed_images_initial_pred = torch.tensor(np.stack(processed_images_initial_pred)).float()
    processed_images_final_pred = torch.tensor(np.stack(processed_images_final_pred)).float()

    # 绘制
    for i in range(k):
        axes[0][i].imshow(processed_images_gt[i].cpu().numpy().astype(np.uint8))
        axes[0][i].set_title(f"GT Image {i+1}")
        axes[0][i].axis('off')

        axes[1][i].imshow(processed_images_initial_pred[i].cpu().numpy().astype(np.uint8))
        axes[1][i].set_title(f"Refer Image {i+1}")
        axes[1][i].axis('off')

        axes[2][i].imshow(processed_images_final_pred[i].cpu().numpy().astype(np.uint8))
        axes[2][i].set_title(f"Pred Image {i+1}")
        axes[2][i].axis('off')

    plt.tight_layout()
    plt.savefig("result.png")
    plt.show()

    return processed_images_gt, processed_images_initial_pred, processed_images_final_pred

def compute_3d_bounding_box(dimensions):
    """
    根据长、宽、高生成三维包围框的 8 个顶点。

    :param dimensions: 包围框的长、宽、高 [length, width, height]
    :return: 包围框的 8 个顶点 (8x3)
    """
    length, width, height = dimensions
    # 以中心点为原点，计算 8 个顶点
    box_3d = np.array([
        [-length / 2, -width / 2, -height / 2],  # 底面
        [length / 2, -width / 2, -height / 2],
        [length / 2, width / 2, -height / 2],
        [-length / 2, width / 2, -height / 2],
        [-length / 2, -width / 2, height / 2],  # 顶面
        [length / 2, -width / 2, height / 2],
        [length / 2, width / 2, height / 2],
        [-length / 2, width / 2, height / 2]
    ])
    return box_3d

def draw_3d_bounding_box(img, R, t, K, dimensions, color=(0, 255, 0), thickness=2):
    """
    在图像上绘制三维包围框。

    :param img: 输入图像
    :param R: 旋转矩阵 (3x3)
    :param t: 平移向量 (3,)
    :param K: 相机内参矩阵 (3x3)
    :param dimensions: 包围框的长、宽、高 [length, width, height]
    :param color: 包围框线条颜色
    :param thickness: 包围框线条厚度
    :return: 绘制包围框后的图像
    """
    # 生成包围框的顶点
    box_3d = compute_3d_bounding_box(dimensions)

    # 将 3D 点转换到相机坐标系
    points_cam = (R @ box_3d.T).T + t

    # 投影到 2D 图像平面
    points_2d_homo = K @ points_cam.T
    points_2d = (points_2d_homo[:2] / points_2d_homo[2]).T  # 归一化为 2D 坐标

    # 绘制包围框
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
        (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
        (0, 4), (1, 5), (2, 6), (3, 7)   # 连接底面和顶面
    ]
    for edge in edges:
        pt1 = tuple(points_2d[edge[0]].astype(int))
        pt2 = tuple(points_2d[edge[1]].astype(int))
        img = cv2.line(img, pt1, pt2, color, thickness)
    return img

def plot_3d_bounding_boxes(images, R_gt, R_final_pred, t_pred, t_gt, K, dimensions):
    """
    绘制三维包围框并可视化。

    :param images: 输入图像列表
    :param R_gt: Ground Truth 旋转矩阵列表
    :param R_final_pred: 预测旋转矩阵列表
    :param t_pred: 预测平移向量列表
    :param t_gt: Ground Truth 平移向量列表
    :param K: 相机内参矩阵 (3x3)
    :param dimensions: 包围框的长、宽、高 [length, width, height]
    """
    k = len(images)
    fig, axes = plt.subplots(1, k, figsize=(k * 4, 6))  # 单行显示
    if k == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(k):
        img_drawed = images[i]

        # 确保图像为 NumPy 数组
        if not isinstance(img_drawed, np.ndarray):
            img_drawed = np.array(img_drawed)

        # 确保图像为 RGB 格式
        # img_drawed = cv2.cvtColor(img_drawed, cv2.COLOR_BGR2RGB)

        # 绘制 Ground Truth 的包围框
        img_drawed = draw_3d_bounding_box(img_drawed, R_gt[i], t_gt[i], K, dimensions, color=(0, 255, 0))

        # 绘制预测的包围框
        img_drawed = draw_3d_bounding_box(img_drawed, R_final_pred[i], t_pred[i], K, dimensions, color=(0, 0, 255))

        # 显示结果
        axes[i].imshow(img_drawed)  # 使用 RGB 格式
        axes[i].set_title(f"Image {i+1}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig("result_bounding_boxes.png")
    plt.show()

def plot_3d_axes_on_images_numpy(images, imgs_refer, R_gt, R_initial_pred, R_final_pred, uvs_pred, uvs_gt, axis_length=125):
    k = len(images)

    # 定义3个坐标轴的单位向量（X, Y, Z轴）
    axis_vectors = np.array([[1.0, 0.0, 0.0],  # X轴
                             [0.0, 1.0, 0.0],  # Y轴
                             [0.0, 0.0, 1.0]])  # Z轴



    # 坐标轴颜色
    axis_colors = [(0, 0, 255),  # 红色 -> X轴
                   (0, 255, 0),  # 绿色 -> Y轴
                   (255, 0, 0)]  # 蓝色 -> Z轴

    # 创建图像列表
    processed_images_gt, processed_images_initial_pred, processed_images_final_pred = [], [], []

    # 图像显示设置
    fig, axes = plt.subplots(3, k, figsize=(k * 4, 12))
    if k == 1:
        axes = np.expand_dims(axes, axis=1)

    for i in range(k):
        # 获取当前图像和旋转矩阵
            # 获取图像大小
        image = images[i]
        img_refer = imgs_refer[i][0].resize(image.size)
        W, H = img_refer.size
        origin_refer = np.array([W // 2, H // 2])
        R_pred = R_final_pred[i]
        R_true = R_gt[i] if R_gt is not None else R_pred
        R_ref = R_initial_pred[i][0]
        print(R_pred)
        print(R_true)
        print(R_ref)
        # 应用旋转矩阵
        rotated_axes_pred = np.dot(axis_vectors, R_pred.T) * axis_length
        rotated_axes_true = np.dot(axis_vectors, R_true.T) * axis_length
        rotated_axes_ref = np.dot(axis_vectors, R_ref.T) * axis_length

        # 投影到2D平面
        projected_axes_pred = rotated_axes_pred[:, :2]
        projected_axes_true = rotated_axes_true[:, :2]
        projected_axes_ref = rotated_axes_ref[:, :2]

        # 
        img = np.array(image)
        img_refer = np.array(img_refer)
        img_copy = img.copy()
        for j, axis in enumerate(projected_axes_true):
            origin = (int(uvs_gt[i][0]), int(uvs_gt[i][1]))
            end_point = (origin[0] + int(axis[0]), origin[1] + int(axis[1])) 
            img_copy = cv2.line(img_copy, origin, end_point, axis_colors[j], 6)
            img_copy = cv2.circle(img_copy, end_point, 8, axis_colors[j], -1)
        processed_images_gt.append(img_copy)

        # 绘制预测旋转矩阵的轴
        img_copy_pred = img.copy()
        for j, axis in enumerate(projected_axes_pred):
            origin = (int(uvs_pred[i][0]), int(uvs_pred[i][1]))
            end_point = (origin[0] + int(axis[0]), origin[1] + int(axis[1])) 
            img_copy_pred = cv2.line(img_copy_pred, origin, end_point, axis_colors[j], 6)
            img_copy_pred = cv2.circle(img_copy_pred, end_point, 8, axis_colors[j], -1)
        processed_images_final_pred.append(img_copy_pred)

        # 绘制参考旋转矩阵的轴
        img_copy_refer = img_refer.copy()
        for j, axis in enumerate(projected_axes_ref):
            end_point = origin_refer + axis.astype(int)
            img_copy_refer = cv2.line(img_copy_refer, tuple(origin_refer), tuple(end_point), axis_colors[j], 6)
            img_copy_refer = cv2.circle(img_copy_refer, tuple(end_point), 8, axis_colors[j], -1)
        processed_images_initial_pred.append(img_copy_refer)

    # 转换为NumPy数组
    processed_images_gt = np.stack(processed_images_gt)
    processed_images_initial_pred = np.stack(processed_images_initial_pred)
    processed_images_final_pred = np.stack(processed_images_final_pred)

    # 绘制
    for i in range(k):
        axes[0][i].imshow(processed_images_gt[i])
        axes[0][i].set_title(f"GT Image {i+1}")
        axes[0][i].axis('off')

        axes[1][i].imshow(processed_images_initial_pred[i])
        axes[1][i].set_title(f"Refer Image {i+1}")
        axes[1][i].axis('off')

        axes[2][i].imshow(processed_images_final_pred[i])
        axes[2][i].set_title(f"Pred Image {i+1}")
        axes[2][i].axis('off')

    plt.tight_layout()
    plt.savefig("result_numpy.png")
    plt.show()

    return processed_images_gt, processed_images_initial_pred, processed_images_final_pred

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
        print(f"Displaying for image {image_idx + 1}")
        print(f"Top {m} refer indices: {top_m_refer_indices}")
        print(f"Top {m} similarities: {top_m_similarities}")

        # 显示原始图像
        plt.subplot(num_images, m + 1, image_idx * (m + 1) + 1)
        plt.imshow(original_images[image_idx])
        plt.title(f"Original Image {image_idx + 1}")
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


def display_top_m_refer_for_all_images_numpy(original_images, refer_images, similarity=None):
    n = len(original_images)  # 总图像数
    m = len(refer_images[0])  # 每张原始图像的参考图像数
    # 设定统一的图像大小
    target_size = (128, 128)  # 可以根据实际情况调整图像大小

    original_images = [img.resize((128, 128)) for img in original_images]
    refer_images = [[ref.resize((128, 128)) for ref in refs] for refs in refer_images]
        
    # 确定需要多少行和列
    rows = n  # 每行显示一张原图及其参考图像
    cols = m + 1  # 每行有m+1列（1列是原图，m列是参考图）

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

    # 如果axes不是2D数组，则转换为2D格式（通常不需要，因为plt.subplots应该返回2D数组）
    # axes = axes if isinstance(axes, np.ndarray) else np.array([[axes]])

    for i in range(n):
        # 显示原始图像
        if n == 1:
            row_axes = axes
        else:
            row_axes = axes[i]
        
        # 显示原始图像
        row_axes[0].imshow(original_images[i])
        row_axes[0].set_title(f"Original Image {i + 1}")
        row_axes[0].axis('off')
        # 显示参考图像
        for j in range(m):
            row_axes[j + 1].imshow(refer_images[i][j])
            title = f"Refer {j + 1}"
            if similarity is not None:
                title += f"\nSim: {similarity[i][j]:.3f}"
            row_axes[j + 1].set_title(title)
            row_axes[j + 1].axis('off')

    plt.tight_layout()
    plt.savefig("result_all_images.png")
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def plot_difference(Rs_gt, Rs_pred, ts_pred, ts_gt):
    """
    Plots the differences in xyz angles using radar chart and t values between ground truth and predictions.

    Parameters:
    Rs_gt : list of numpy arrays
        List of ground truth rotation matrices.
    Rs_pred : list of predicted rotation matrices.
    ts_pred : list of numpy arrays
        List of predicted translations (3D vectors).
    ts_gt : list of numpy arrays
        List of ground truth translations (3D vectors).
    """
    # Check if all inputs have the same length
    assert len(Rs_gt) == len(Rs_pred) == len(ts_pred) == len(ts_gt), "All input lists must have the same length."

    # Compute rotation differences (in degrees)
    angles_diff = []
    for R_gt, R_pred in zip(Rs_gt, Rs_pred):
        R_rel = R.from_matrix(R_gt).inv() * R.from_matrix(R_pred)
        angles = R_rel.as_euler('xyz', degrees=True)  # Get xyz angles in degrees
        angles_diff.append(angles)
    angles_diff = np.array(angles_diff)

    # Compute translation differences
    t_diff = np.linalg.norm(np.array(ts_gt) - np.array(ts_pred), axis=1)
    t_signed_diff = np.array(ts_pred) - np.array(ts_gt)  # Compute signed differences

    # Plot rotation differences as radar histograms
    labels = ["X-axis", "Y-axis", "Z-axis"]
    theta = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)  # 3 axes (x, y, z)
    plt.rcParams.update({
        'font.size': 22,  # General font size
        'axes.titlesize': 22,  # Title size
    })
    plt.figure(figsize=(18, 6))
    plt.figure(figsize=(18, 6))
    for i, axis in enumerate(labels):
        plt.subplot(1, 3, i + 1, polar=True)
        data = angles_diff[:, i]  # Collect data for the specific axis
        histogram, bins = np.histogram(data, bins=10, range=(data.min(), data.max()))
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax = plt.gca()
        ax.set_theta_zero_location('N')  # Set zero degrees at the top
        ax.set_theta_direction(-1)  # Clockwise direction
        plt.bar(bin_centers * np.pi / 180, histogram, width=np.diff(bins) * np.pi / 180, alpha=0.7, edgecolor='black')
        plt.title(f'{axis} Angle Differences', fontsize=12)
        # Set font sizes globally

    plt.tight_layout()
    plt.show()
    plt.savefig("xyz_difference.png")

    # Plot translation differences as Gaussian-like histogram
    plt.figure(figsize=(12, 6))
    plt.hist(t_signed_diff.flatten(), bins=30, alpha=0.75, edgecolor='black', density=True)
    mean = np.mean(t_signed_diff)
    std = np.std(t_signed_diff)
    x = np.linspace(t_signed_diff.min(), t_signed_diff.max(), 100)
    gaussian = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
    # plt.plot(x, gaussian, label='Gaussian Fit', color='red')
    # plt.xlabel('Signed Translation Difference (units)')
    plt.ylabel('Density')
    plt.title('Gaussian-like Histogram of Translation Differences')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig("t_difference.png")

def main():
    """
    Test function for plot_difference.
    """
    # Generate random test data
    num_samples = 30
    Rs_gt = [R.random().as_matrix() for _ in range(num_samples)]
    Rs_pred = [R.random().as_matrix() for _ in range(num_samples)]
    ts_gt = [np.random.rand(3) for _ in range(num_samples)]
    ts_pred = [t + np.random.normal(0, 0.1, 3) for t in ts_gt]  # Add small noise to gt translations

    # Call plot_difference with test data
    plot_difference(Rs_gt, Rs_pred, ts_pred, ts_gt)

if __name__ == "__main__":
    main()

from torchvision import transforms

transform = transforms.Compose(
     [transforms.Resize((256, 256)),
     transforms.ToTensor()]
)
# if __name__ == "__main__":
#     # 设置参数
#     batch_size = 2
#     H, W = 256, 256
#     num_refers = 10  # 假设有10张参考图像
#     m = 5  # 每个图像显示最相似的5个参考图像

#     # 生成一些随机图像数据 (batch_size, H, W, 3)
#     img_num = 2
#     test_scene_ids = list(range(48,60))
#     target_dirs = [f'../Datasets/ycbv/test/{str(id).zfill(6)}/view_000/' for id in test_scene_ids]
#     rgb_dirs = [f'{dir}/crop'for dir in target_dirs]
#     gt_files = [f'{dir}/view_000_info.json'for dir in target_dirs]
#     items = []
#     for rgb_dir, gt_file in zip(rgb_dirs, gt_files):
#         with open(gt_file, 'r') as f:
#             gt = json.load(f)
#         for data in gt:
#             if data["obj_id"] == 1:
#                 items.append((rgb_dir, data))
#     selected_data = random.sample(items, img_num)  # 从 gt 中随机选择 img_num 个元素
#     test_images = []
#     R_gt = []
#     for (rgb_dir, item) in selected_data:
#         image_path = f"{rgb_dir}/{str(item['img_id']).zfill(6)}_{str(item['idx']).zfill(6)}.png"
#         image = Image.open(image_path)
#         R = torch.tensor(item["R_relative"], dtype=torch.float32)
#         test_images.append(image)
#         R_gt.append(R)

#     R_gt = torch.tensor(np.stack(R_gt)).cpu()
#     images = torch.stack([transform(img) for img in test_images]).cpu()
#     images = images.permute(0, 2, 3, 1)  # (batch_size, 3, H, W)


#     # 生成一些随机旋转矩阵数据 (batch_size, 3, 3)
#     # R_gt = torch.rand((batch_size, 3, 3))
#     R_initial_pred = torch.rand((batch_size, 3, 3))
#     R_final_pred = torch.rand((batch_size, 3, 3))

#     # 生成一些随机的相似度矩阵数据 (num_images, num_refers)
#     similarity_matrix = torch.rand((batch_size, num_refers))

#     # 获取每个图像最相似的 m 个参考图像的索引
#     top_m_indices = torch.topk(similarity_matrix, m, dim=1).indices

#     # 假设我们有一些随机生成的参考图像
#     original_refers = [np.random.rand(H, W, 3) for _ in range(num_refers)]

#     # # 调用第二个函数 `display_top_m_refer_for_all_images`
#     # display_top_m_refer_for_all_images(
#     #     images, original_refers, similarity_matrix.numpy(), top_m_indices.numpy(), m=m
#     # )

#     # 调用第一个函数 `plot_3d_axes_on_images`
#     processed_images_gt, processed_images_initial_pred, processed_images_final_pred = plot_3d_axes_on_images_numpy(
#         images, R_gt, R_initial_pred, R_final_pred, axis_length=25
#     )
if __name__ == "__main__":
    # 设置参数
    batch_size = 2
    H, W = 256, 256
    num_refers = 10  # 假设有10张参考图像
    m = 5  # 每个图像显示最相似的5个参考图像

    # 生成一些随机图像数据 (batch_size, H, W, 3)
    img_num = 2
    test_scene_ids = list(range(48,60))
    target_dirs = [f'../Datasets/ycbv/test/{str(id).zfill(6)}/view_000/' for id in test_scene_ids]
    rgb_dirs = [f'{dir}/crop'for dir in target_dirs]
    gt_files = [f'{dir}/view_000_info.json'for dir in target_dirs]
    items = []
    for rgb_dir, gt_file in zip(rgb_dirs, gt_files):
        with open(gt_file, 'r') as f:
            gt = json.load(f)
        for data in gt:
            if data["obj_id"] == 1:
                items.append((rgb_dir, data))
    selected_data = random.sample(items, img_num)  # 从 gt 中随机选择 img_num 个元素
    test_images = []
    R_gt = []
    for (rgb_dir, item) in selected_data:
        image_path = f"{rgb_dir}/{str(item['img_id']).zfill(6)}_{str(item['idx']).zfill(6)}.png"
        image = Image.open(image_path)
        R = np.array(item["R_relative"], dtype=np.float32)
        test_images.append(image)
        R_gt.append(R)

    R_gt = np.stack(R_gt)
    images = np.stack([np.array(transform(img)) for img in test_images])  # Apply transform and stack the images
    images = images.transpose(0, 2, 3, 1)  # (batch_size, H, W, 3)


    # 生成一些随机旋转矩阵数据 (batch_size, 3, 3)
    # R_gt = torch.rand((batch_size, 3, 3))
    R_initial_pred = R_gt
    R_final_pred = R_gt

    # # 生成一些随机的相似度矩阵数据 (num_images, num_refers)
    # similarity_matrix = torch.rand((batch_size, num_refers))

    # # 获取每个图像最相似的 m 个参考图像的索引
    # top_m_indices = torch.topk(similarity_matrix, m, dim=1).indices

    # # 假设我们有一些随机生成的参考图像
    # original_refers = [np.random.rand(H, W, 3) for _ in range(num_refers)]

    # # 调用第二个函数 `display_top_m_refer_for_all_images`
    # display_top_m_refer_for_all_images(
    #     images, original_refers, similarity_matrix.numpy(), top_m_indices.numpy(), m=m
    # )

    # 调用第一个函数 `plot_3d_axes_on_images`
    processed_images_gt, processed_images_initial_pred, processed_images_final_pred = plot_3d_axes_on_images_numpy(
        images, R_gt, R_initial_pred, R_final_pred, axis_length=25
    )