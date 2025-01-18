import os
import shutil
test_scene_ids = list(range(48,60))
train_scene_ids = list(set(range(0, 92)) - set(test_scene_ids))
def remove_files_and_dirs(base_dir, dirs_to_remove, files_to_remove):
    # 删除指定的目录
    for dir_name in dirs_to_remove:
        dir_path = os.path.join(base_dir, dir_name)
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"目录 {dir_path} 已删除。")
        else:
            print(f"目录 {dir_path} 不存在。")

    # 删除指定的文件
    for file_name in files_to_remove:
        file_path = os.path.join(base_dir, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"文件 {file_path} 已删除。")
        else:
            print(f"文件 {file_path} 不存在。")

if __name__ == "__main__":
    for scene_id in train_scene_ids:
        target_dir = f'../Datasets/ycbv/test/{str(scene_id).zfill(6)}'
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        # base_dir = f'/home/mendax/project/Datasets/lm/{str(obj_id).zfill(6)}'  # 设置基础目录路径
            # base_dir = target_dir  # 设置基础目录路径
            # dirs_to_remove = ['view_000/crop']  # 需要删除的目录
            # files_to_remove = ['view_000.json']  # 需要删除的文件

            # remove_files_and_dirs(base_dir, dirs_to_remove, files_to_remove)
