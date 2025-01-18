import cv2
import numpy as np
import sys

def find_feature_matches(img1, img2):
    """
    使用ORB和BFMatcher检测并匹配两张图像之间的特征点。

    参数:
        img1 (np.ndarray): 第一张输入图像。
        img2 (np.ndarray): 第二张输入图像。

    返回:
        keypoints1 (list of cv2.KeyPoint): 第一张图像的关键点。
        keypoints2 (list of cv2.KeyPoint): 第二张图像的关键点。
        good_matches (list of cv2.DMatch): 过滤后的良好匹配。
    """
    # 初始化ORB检测器
    orb = cv2.ORB_create()

    # 检测关键点
    keypoints1 = orb.detect(img1, None)
    keypoints2 = orb.detect(img2, None)

    # 计算描述子
    keypoints1, descriptors1 = orb.compute(img1, keypoints1)
    keypoints2, descriptors2 = orb.compute(img2, keypoints2)

    # 初始化使用汉明距离的BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # 描述子匹配
    matches = bf.match(descriptors1, descriptors2)

    # 根据距离排序匹配
    matches = sorted(matches, key=lambda x: x.distance)

    # 过滤匹配：仅保留良好匹配（距离 < 2*最小距离 或 距离 < 30）
    min_dist = matches[0].distance
    max_dist = matches[-1].distance
    print(f"-- 最大距离 : {max_dist}")
    print(f"-- 最小距离 : {min_dist}")

    good_matches = [m for m in matches if m.distance <= max(2 * min_dist, 30.0)]

    return keypoints1, keypoints2, good_matches

def pixel2cam(p, K):
    """
    将像素坐标转换为相机归一化坐标（齐次坐标形式）。

    参数:
        p (tuple): 像素坐标 (x, y)。
        K (np.ndarray): 相机内参矩阵。

    返回:
        np.ndarray: 归一化的相机坐标 [x, y, 1]。
    """
    return np.array([
        (p[0] - K[0, 2]) / K[0, 0],
        (p[1] - K[1, 2]) / K[1, 1],
        1.0
    ])


def pose_estimation_2d2d(keypoints1, keypoints2, matches, K):
    """
    基于匹配的关键点估计相机姿态（旋转矩阵R和平移向量t）。

    参数:
        keypoints1 (list of cv2.KeyPoint): 第一张图像的关键点。
        keypoints2 (list of cv2.KeyPoint): 第二张图像的关键点。
        matches (list of cv2.DMatch): 匹配的关键点。
        K (np.ndarray): 相机内参矩阵。

    返回:
        R (np.ndarray): 旋转矩阵。
        t (np.ndarray): 平移向量。
    """
    # 将匹配的关键点转换为Point2f格式
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

    # 计算基础矩阵
    fundamental_matrix, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_8POINT)
    print("基础矩阵 Fundamental Matrix:\n", fundamental_matrix)

    # 计算本质矩阵
    principal_point = (K[0, 2], K[1, 2])  # (cx, cy)
    focal_length = K[0, 0]  # 假设 fx = fy
    essential_matrix, mask = cv2.findEssentialMat(points1, points2, focal_length, principal_point, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    print("本质矩阵 Essential Matrix:\n", essential_matrix)

    # 计算单应矩阵
    # homography_matrix, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 3)
    # print("单应矩阵 Homography Matrix:\n", homography_matrix)

    # 从本质矩阵恢复姿态（旋转和平移）
    _, R, t, mask = cv2.recoverPose(essential_matrix, points1, points2, K)
    print("旋转矩阵 R:\n", R)
    print("平移向量 t:\n", t)
    return R, t.flatten()

def pose_estimation_2d2d(keypoints1, keypoints2, K1, K2):
    """
    基于匹配的关键点估计相机姿态（旋转矩阵R和平移向量t）。
    参数:
        keypoints1 (Tensor): 第一张图像的关键点，大小为 [N, 2]。
        keypoints2 (Tensor): 第二张图像的关键点，大小为 [N, 2]。
        K1 (np.ndarray): 第一张图像的相机内参矩阵。
        K2 (np.ndarray): 第二张图像的相机内参矩阵。
    返回:
        R (np.ndarray): 旋转矩阵。
        t (np.ndarray): 平移向量。
    """
    # 将 PyTorch 张量转换为 numpy 数组
    points1 = keypoints1.numpy()  # 第一张图像的关键点
    points2 = keypoints2.numpy()  # 第二张图像的关键点

    # # 计算基础矩阵 F
    # F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_LMEDS)

    # # 计算本质矩阵 E
    # E = K2.T @ F @ K1

    dist1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    dist2 = dist1

    # 从本质矩阵恢复姿态（旋转和平移），这里的内参矩阵使用第一张图片的内参
    retval,E, R, t, mask = cv2.recoverPose(points1, points2, K1, dist1, K2, dist2)
    print("retval R:\n",retval)
    print("旋转矩阵 R:\n", R)
    print("平移向量 t:\n", t)
    print("本质矩阵 E:\n", E)
    print("重投影误差 mask:\n")

    return R, t.flatten()


def main():
    # 读取图像
    img1_path = "/home/mendax/project/CLIPPose/refers/obj_1/001_000200_000000.png"
    img2_path = "/home/mendax/project/CLIPPose/refers/obj_1/001_000400_000000.png"
    img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)

    if img1 is None or img2 is None:
        print("读取图像时出错。")
        sys.exit(1)

    # 查找特征匹配
    keypoints1, keypoints2, matches = find_feature_matches(img1, img2)
    print(f"良好匹配的数量: {len(matches)}")

    # 相机内参矩阵（假设来自TUM Freiburg2数据集）
    K = np.array([[520.9, 0, 325.1],
                  [0, 521.0, 249.7],
                  [0, 0, 1]])

    # 估计相机姿态
    R, t = pose_estimation_2d2d(keypoints1, keypoints2, matches, K)

    # 验证 E = [t]_x * R
    t_x = np.array([[0, -t[2,0], t[1,0]],
                    [t[2,0], 0, -t[0,0]],
                    [-t[1,0], t[0,0], 0]])
    E = t_x @ R
    print("t^R (斜对称矩阵与旋转矩阵 R 的乘积):\n", E)

    # 验证每对匹配点的对极约束
    print("\n验证每对匹配点的对极约束:")
    for m in matches:
        pt1 = keypoints1[m.queryIdx].pt
        pt2 = keypoints2[m.trainIdx].pt
        y1 = np.array([pixel2cam(pt1, K)]).reshape(3, 1)
        y2 = np.array([pixel2cam(pt2, K)]).reshape(3, 1)
        d = y2.T @ t_x @ R @ y1
        print(f"对极约束值 = {d[0][0]}")

if __name__ == "__main__":
    main()
