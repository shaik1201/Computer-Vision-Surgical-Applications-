import blenderproc as bproc
import argparse
import os
import json
import numpy as np
from blenderproc.python.camera import CameraUtility
import math
from mathutils import Euler
import random
import glob

def deg_to_rad(deg):
    return math.radians(deg)

def get_hdr_img_paths_from_haven(data_path: str) -> str:
    """ Returns .hdr file paths from the given directory.
    :param data_path: A path pointing to a directory containing .hdr files.
    :return: .hdr file paths
    """
    if os.path.exists(data_path):
        data_path = os.path.join(data_path, "hdris")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"The folder: {data_path} does not contain a folder named hdris. "
                                    f"Please use the download script.")
    else:
        raise FileNotFoundError(f"The data path does not exist: {data_path}")

    hdr_files = glob.glob(os.path.join(data_path, "*", "*.hdr"))
    # Ensure the call is deterministic
    hdr_files.sort()
    return hdr_files

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--palm_obj', type=str, default='/home/student/project/mlflow/synth_data/Hand_objects/hand2.obj', help="Path to palm .obj file")
parser.add_argument('--large_palm_obj', type=str, default='/home/student/project/mlflow/synth_data/Hand_objects/hand1.obj', help="Path to palm .obj file")
parser.add_argument('--arm_obj', type=str, default='/home/student/project/mlflow/synth_data/Hand_objects/arm.obj', help="Path to arm .obj file")
parser.add_argument('--needle_holder_obj', type=str, default='/datashare/project/surgical_tools_models/needle_holder/NH1.obj', help="Path to needle holder .obj file")
parser.add_argument('--tweezers_obj', type=str, default='/datashare/project/surgical_tools_models/tweezers/T1.obj', help="Path to needle holder .obj file")
parser.add_argument('--output_dir', type=str, default='./', help="Directory where rendered images and COCO annotations will be saved")
parser.add_argument('--camera_params', type=str, default='/home/student/project/camera.json', help="Path to camera parameters JSON file")
parser.add_argument('--haven_path', default="/datashare/project/haven/", help="Path to the haven hdri images")

args = parser.parse_args()

# Initialize BlenderProc
bproc.init()

# Load the objects
palm = bproc.loader.load_obj(args.palm_obj)[0]
palm.set_cp("category_id", 5)

palm_2 = bproc.loader.load_obj(args.palm_obj)[0]
palm_2.set_cp("category_id", 6)

arm = bproc.loader.load_obj(args.arm_obj)[0]
arm.set_cp("category_id", 3)

needle_holder = bproc.loader.load_obj(args.needle_holder_obj)[0]
needle_holder.set_cp("category_id", 1)

needle_holder_2 = bproc.loader.load_obj(args.needle_holder_obj)[0]
needle_holder_2.set_cp("category_id", 7)

tweezers = bproc.loader.load_obj(args.tweezers_obj)[0]
tweezers.set_cp("category_id", 2)

# Create materials and assign colors
skin_material = bproc.material.create('skin_material')
skin_material.set_principled_shader_value('Base Color', [0.8, 0.6, 0.5, 1])  # light brownish skin tone
arm.replace_materials(skin_material)

glove_material = bproc.material.create('glove_material')
glove_material.set_principled_shader_value('Base Color', [1.0, 1.0, 1.0, 1])  # white color
palm.replace_materials(glove_material)
palm_2.replace_materials(glove_material)

tweezers_material = bproc.material.create('tweezers_material')
tweezers_material.set_principled_shader_value('Base Color', [0.5, 0.5, 0.5, 1])  # gray/silver
tweezers_material.set_principled_shader_value('Metallic', 1.0)
tweezers_material.set_principled_shader_value('Roughness', 0.2)  # make it shiny
tweezers.replace_materials(tweezers_material)

needle_holder.replace_materials(tweezers_material)
needle_holder_2.replace_materials(tweezers_material)


# Set up a simple light source
light = bproc.types.Light()
light.set_location([np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(3, 6)])
light.set_energy(np.random.uniform(300, 1000))  # Random energy level
light.set_color([np.random.uniform(0.8, 1), np.random.uniform(0.8, 1), np.random.uniform(0.8, 1)])  # Slightly colored light

# Load camera parameters
with open(args.camera_params, "r") as file:
    camera_params = json.load(file)

fx, fy = camera_params["fx"], camera_params["fy"]
cx, cy = camera_params["cx"], camera_params["cy"]
im_width, im_height = camera_params["width"], camera_params["height"]

K = np.array([[fx, 0, cx], 
              [0, fy, cy], 
              [0, 0, 1]])
CameraUtility.set_intrinsics_from_K_matrix(K, im_width, im_height)

# Set the camera location and rotation
camera_location = np.array([2.2, 0.37, 4.12])
camera_rotation_euler = np.array([deg_to_rad(34.8), deg_to_rad(0.82), deg_to_rad(94.3982)])

# Create a camera pose from location and rotation
cam2world_matrix = bproc.math.build_transformation_mat(camera_location, Euler(camera_rotation_euler).to_matrix())

# Add the camera pose
bproc.camera.add_camera_pose(cam2world_matrix)

# Load HDRIs and apply to world background
hdr_files = get_hdr_img_paths_from_haven(args.haven_path)

# Apply a random HDRI to the scene
random_hdr_file = random.choice(hdr_files)
bproc.world.set_world_background_hdr_img(random_hdr_file)

# Set a random world lighting strength using BlenderProc's API
# bproc.world.set_world_background_strength(np.random.uniform(0.1, 1.5))

# Render the scene from different viewpoints
for i in range(15):  # 3 different viewpoints as an example
    try:
        # location
        palm.set_location([-0.359651 + np.random.uniform(-0.1, 0.1), -0.882351 + np.random.uniform(-0.1, 0.1), 0.138183 + np.random.uniform(-0.05, 0.05)])
        palm_2.set_location([-0.65634 + np.random.uniform(-0.1, 0.1), 0.083 + np.random.uniform(-0.1, 0.1), 0.4075 + np.random.uniform(-0.05, 0.05)])
        arm.set_location([0.007737, -0.023905, -0.012758])
        needle_holder.set_location([-0.332743 + np.random.uniform(-0.1, 0.1), -0.499926 + np.random.uniform(-0.1, 0.1), 0.102616 + np.random.uniform(-0.05, 0.05)])
        needle_holder_2.set_location([0.079361 + np.random.uniform(-0.1, 0.1), -0.796936 + np.random.uniform(-0.1, 0.1), -0.163984 + np.random.uniform(-0.05, 0.05)])
        tweezers.set_location([-0.526652 + np.random.uniform(-0.1, 0.1), -0.017417 + np.random.uniform(-0.1, 0.1), 0.29179 + np.random.uniform(-0.05, 0.05)])
        
        # scale
        palm.set_scale([1, 1, 1])
        palm_2.set_scale([1, 1, 1])
        arm.set_scale([3.20532, 3.20532, 3.20532])
        needle_holder.set_scale([0.351965, 0.351965, 0.351965])
        needle_holder_2.set_scale([0.251965, 0.251965, 0.251965])
        tweezers.set_scale([0.304363, 0.304363, 0.304363])
        
        # rotation
        arm_rotation_euler = np.array([deg_to_rad(-90), deg_to_rad(0), deg_to_rad(0)])
        arm.set_rotation_euler(arm_rotation_euler)
        
        needle_holder_rotation_euler = np.array([deg_to_rad(0 + np.random.uniform(-50, 50)), deg_to_rad(0 + np.random.uniform(-50, 50)), deg_to_rad(-12.9577 + np.random.uniform(-50, 50))])
        needle_holder.set_rotation_euler(needle_holder_rotation_euler)
        
        needle_holder_2_rotation_euler = np.array([deg_to_rad(0 + np.random.uniform(-50, 50)), deg_to_rad(0 + np.random.uniform(-50, 50)), deg_to_rad(-12.9577 + np.random.uniform(-50, 50))])
        needle_holder_2.set_rotation_euler(needle_holder_2_rotation_euler)
        
        tweezers_rotation_euler = np.array([deg_to_rad(205.973 + np.random.uniform(-50, 50)), deg_to_rad(-3.62783 + np.random.uniform(-50, 50)), deg_to_rad(71.7503 + np.random.uniform(-50, 50))])
        tweezers.set_rotation_euler(tweezers_rotation_euler)
        
        palm_rotation_euler = np.array([deg_to_rad(-1.18589 + np.random.uniform(-50, 50)), deg_to_rad(300.931 + np.random.uniform(-50, 50)), deg_to_rad(-179.903 + np.random.uniform(-50, 50))])
        palm.set_rotation_euler(palm_rotation_euler)
        
        palm_2_rotation_euler = np.array([deg_to_rad(-137.508 + np.random.uniform(-50, 50)), deg_to_rad(144.624 + np.random.uniform(-50, 50)), deg_to_rad(-72.729 + np.random.uniform(-50, 50))])
        palm_2.set_rotation_euler(palm_2_rotation_euler)
        
        # Perform the render
        bproc.renderer.set_output_format(enable_transparency=False)
        bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"])

        data = bproc.renderer.render()
        # Write data to coco file
        bproc.writer.write_coco_annotations(os.path.join(args.output_dir, 'coco_data'),
                                instance_segmaps=data["instance_segmaps"],
                                instance_attribute_maps=data["instance_attribute_maps"],
                                colors=data["colors"],
                                mask_encoding_format="polygon",
                                append_to_existing_output=True)
        
    except Exception as e:
        print(f"Error in rendering loop: {e}")

print("Render completed.")
