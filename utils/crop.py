import os
import json
from PIL import Image
import concurrent.futures
import numpy as np
import matplotlib.pyplot as plt

# 假设每个视图中只有一个物体

# 切换到脚本所在的目录
# script_dir = os.path.dirname(os.path.abspath(__file__))
# os.chdir(script_dir)

# def apply_mask_to_rgb(rgb, mask):
#     r, g, b = rgb.split()
#     # 将 mask 转换为二进制掩码，以确保只有 0 和 255 值
#     binary_mask = mask.point(lambda p: p > 0 and 255)
    
#     # 将 R、G、B 通道中 mask 为 0 的区域设置为 0
#     r = Image.composite(r, Image.new("L", r.size, 0), binary_mask)
#     g = Image.composite(g, Image.new("L", g.size, 0), binary_mask)
#     b = Image.composite(b, Image.new("L", b.size, 0), binary_mask)

#     # 合并 R、G、B 通道和 mask 通道为一个四通道图像
#     rgb = Image.merge("RGB", (r, g, b))
#     return rgb
def apply_mask_to_rgb(rgb, mask):

    r, g, b = rgb.split()
    binary_mask = mask.point(lambda p: 255 if p > 0 else 0)
    alpha = binary_mask  # Alpha通道直接使用binary_mask
    rgba = Image.merge("RGBA", (r, g, b, alpha))
    
    return rgba

def draw_uv_points(image, uv_point, color=(255, 0, 0), radius=3):
    """
    在图像上绘制给定的 uv 点。
    :param image: 需要绘制的图像 (PIL Image)
    :param uv_points: uv 坐标列表
    :param color: 绘制点的颜色
    :param radius: 绘制点的半径
    :return: 绘制后的图像
    """
    img = np.array(image)  # 将PIL Image转换为NumPy数组
    u, v = int(uv_point[0]), int(uv_point[1])
    if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
        img[v - radius:v + radius, u - radius:u + radius] = color  # 在图像上绘制红色点
    return Image.fromarray(img)

def project_points(points, rotation, translation):
    # 执行旋转和位移：使用广播机制来处理多个点
    # points 是一个 N x 3 的数组，表示多个 3D 点
    # translation 是 1 x 3 的向量，表示平移
    # rotation 是 3 x 3 的旋转矩阵

    # 对每个点应用平移和旋转
    projected_points = (points - translation) @ rotation.T  # 先平移再旋转
    
    # 返回投影后的点（可以根据需要返回x, y, z或仅x和y）
    return projected_points  # 返回所有3D坐标点

def generate_point_cloud(rgb, depth, Kc):
    h, w = depth.shape
    fx, fy = Kc[0][0], Kc[1][1]
    cx, cy = Kc[0][2], Kc[1][2]

    # 生成像素坐标
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u, v = u.flatten(), v.flatten()
    z = depth.flatten()

    # 过滤有效深度
    valid = z > 0
    u, v, z = u[valid], v[valid], z[valid]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.stack((x, y, z), axis=-1)
    colors = rgb[v, u]

    return points, colors

def render_image(points, colors, img_size, Kc):
    img = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
    fx, fy = Kc[0][0], Kc[1][1]
    cx, cy = Kc[0][2], Kc[1][2]
    print(img.shape)
    for i, (point, color) in enumerate(zip(points, colors)):
        v, u = int(point[1]*fy/point[2] + cy), int(point[0]*fx/point[2] + cx)
        if 0 <= u < img_size[1] and 0 <= v < img_size[0]:
            img[v, u] = color
    return img

def crop_and_save(image, crop_area, output_dir, filename):  
    cropped_image = image.crop(crop_area)  
    output_path = os.path.join(output_dir, filename)  
    cropped_image.save(output_path) 

# 函数生成中心点数据并保存裁剪图像
def process_view(target_dir):
    masked_crop_dir_path = f"{target_dir}/view_000/masked_crop"
    crop_dir_path = f"{target_dir}/view_000/crop"

    if not os.path.exists(crop_dir_path):
        os.makedirs(crop_dir_path)
        print(f"目录 {crop_dir_path} 已创建。")    
    if not os.path.exists(masked_crop_dir_path):
        os.makedirs(masked_crop_dir_path)
        print(f"目录 {masked_crop_dir_path} 已创建。")

    with open(f"{target_dir}/view_000/view_000_info.json") as f:
        items = json.load(f)
    print(len(items))
    for item in items:
        img_id = item['img_id']
        obj_idx = item['idx']
        bbox = item['bbox']
        # uv = item['uv']
        # uv_relative = item['uv_relative']
        # R = item['R']
        # t = item['t']
        # Rc = item['Rc']
        # tc = item['tc']
        # Kc = item['Kc']

        
        # 加载 RGB 图像

        rgb_path = f"{target_dir}/rgb/{str(img_id).zfill(6)}.png"
        mask_path = f"{target_dir}/mask_visib/{str(img_id).zfill(6)}_{str(obj_idx).zfill(6)}.png"
        crop_path = f"{crop_dir_path}/{str(img_id).zfill(6)}_{str(obj_idx).zfill(6)}.png"
        masked_crop_path = f"{masked_crop_dir_path}/{str(img_id).zfill(6)}_{str(obj_idx).zfill(6)}.png"
        print(f"正在处理 {rgb_path}。")

        rgb = Image.open(rgb_path)
        mask = Image.open(mask_path)
        masked_rgb = apply_mask_to_rgb(rgb, mask)
        print(f"正在处理 {rgb_path}。")
        
        # Depth
        rgb_crop = rgb.crop(bbox)
        rgb_crop.save(crop_path)
        masked_crop = masked_rgb.crop(bbox)
        masked_crop.save(masked_crop_path)

    print(f"完成 {target_dir} 的裁剪。")

# 使用并行处理每个视图的处理任务
def main():
    view_num = 1
    scene_ids = []
    target_dirs = []

    for scene_id in range(0, 4):
        target_dir = f'Datasets/space_station/test/socket/train_pbr/{str(scene_id).zfill(6)}'
        if os.path.exists(target_dir):
            scene_ids.append(scene_id)
            target_dirs.append(target_dir)
            # process_view(target_dir)
        else:
            print(f"{target_dir} does not exist.")

    # 使用 ThreadPoolExecutor 来并行处理
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(process_view, target_dirs)

if __name__ == "__main__":
    main()
