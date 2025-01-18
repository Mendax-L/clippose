import blenderproc as bproc
import argparse
import numpy as np
import sys
import os
import shutil
import bpy

models = {1: "socket", 2: "thruster"}

sys.path.append('/home/mendax/anaconda3/envs/pose/lib/python3.11/site-packages')

parser = argparse.ArgumentParser()
parser.add_argument('output_dir', nargs='?', default="/home/mendax/project/Datasets/space_station/train", help="Path to where the final files will be saved")

args = parser.parse_args()

bproc.init()
objs_path= []
for model_id in models.keys():
    object_path=f"/home/mendax/project/Datasets/space_station/models/{models[model_id]}.ply"

    objs_path.append(object_path)


# 设置相机内参
bproc.camera.set_intrinsics_from_K_matrix(
    [[572.4114, 0.0, 325.2611],
     [0.0, 573.57043, 242.04899],
     [0.0, 0.0, 1.0]], 640, 480
)



# 设置环境光
bpy.context.scene.render.engine = 'CYCLES'
world = bpy.context.scene.world
if world is None:
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
world.use_nodes = True
node_tree = world.node_tree
env_texture = node_tree.nodes.new(type='ShaderNodeTexEnvironment')
hdr_path = "/home/mendax/project/Datasets/space_station/background/RenderCrate-HDRI_Orbital_46_Sunset_prev_lg.png"
env_texture.image = bpy.data.images.load(hdr_path)
output_node = node_tree.nodes.get('World Output') or node_tree.nodes.new(type='ShaderNodeOutputWorld')
node_tree.links.new(env_texture.outputs['Color'], output_node.inputs['Surface'])
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name", "dataset_name"],
                                            default_values={"category_id": 0, "dataset_name": None})

# 渲染每个对象到不同的chunk
for obj_idx, obj_path in enumerate(objs_path):  # 可以添加更多对象到列表中
    obj = bproc.loader.load_obj(obj_path)[0]
    obj.set_cp("category_id", list(models.keys())[obj_idx])
    obj.set_scale([0.001, 0.001, 0.001])
    obj_name = obj.get_name()
    for mat in obj.get_materials():
        mat.map_vertex_color()
    obj_output_dir = os.path.join(args.output_dir, f"{obj_name}")
    os.makedirs(obj_output_dir, exist_ok=True)

    # 设置灯光


    bbox = obj.blender_obj.bound_box
    min_corner = np.min(bbox, axis=0)
    max_corner = np.max(bbox, axis=0)
    # 获取缩放比例
    scale = obj.get_scale()
    # 计算实际尺寸
    actual_size = (max_corner - min_corner) * scale
    print(actual_size)
    r  = actual_size.max()
    # 定义姿态采样函数
    def sample_pose_func(obj: bproc.types.MeshObject):
        obj.set_location([0, 0, 0])
        # obj.set_rotation_euler(bproc.sampler.uniformSO3())
        # obj.set_rotation_euler(np.random.uniform([-np.pi/4, -np.pi/4, -np.pi/4], [np.pi/4, np.pi/4, np.pi/4]))
        obj.set_rotation_euler([0, 0, 0])
        # obj.set_location(np.random.uniform([-r, -r, r], [r, r, r]))


    # 渲染多个相机视角
    scene_num = 6
    pose_num = 40
    for _ in range(scene_num):  # 每个对象生成10个不同的场景
        bproc.object.sample_poses(objects_to_sample=[obj],
                                  sample_pose_func=sample_pose_func,
                                  max_tries=1000)

        bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects([obj])
        poses = 0

        while poses < pose_num:
            location = bproc.sampler.shell(center=[0, 0, 0],
                                           radius_min=4*r,
                                           radius_max=7*r,
                                        #    elevation_min=1,
                                        #    elevation_max=89,
                                           uniform_volume=False)
            poi = bproc.object.compute_poi([obj])
            rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location,
                                                                     inplane_rot=np.random.uniform(-0.7854, 0.7854))
            cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
            
            light_point = bproc.types.Light()
            light_point.set_energy(100)
            light_point.set_location([10,10,10])
            if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": r,"max": 2*r}, bop_bvh_tree):
                bproc.camera.add_camera_pose(cam2world_matrix, frame=poses)
                poses += 1
                        # 获取物体的中心位置
        #     obj_center = obj.get_location()

        #     # 获取当前相机位置
        #     cam_location = bproc.camera.get_camera_pose(frame=poses)[:3, 3]

        #     # 计算相机与物体中心的距离
        #     distance = np.linalg.norm(cam_location - obj_center)
        #     print(f"物体 '{obj_name}' 到相机的中心点距离为: {distance} 米")
        # # 渲染当前场景
        # 获取当前活动相机
        camera = bpy.context.scene.camera

        # 设置相机的剪切距离
        camera.data.clip_start = 0.01  # 设置更小的近裁剪距离
        # camera.data.clip_end = 1000   # 设置更大的远裁剪距离
        data = bproc.renderer.render()

        # 将数据写入BOP格式，确保每个对象存储在单独的chunk中
        bproc.writer.write_bop(
            output_dir=obj_output_dir,
            target_objects=[obj],
            depths=data["depth"],
            colors=data["colors"],
            m2mm=True,
            append_to_existing_output=True,
            frames_per_chunk=scene_num * pose_num,
        )

    print(f"Object '{obj_name}' saved to chunk directory: {obj_output_dir}")
    bpy.ops.wm.memory_statistics()  # 可选：用于调试内存
    obj.delete()

