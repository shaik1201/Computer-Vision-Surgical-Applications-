import numpy as np
import argparse
import random
import os
import json
import subprocess
import shutil


class Pipeline:
    def __init__(self, NH_folder_path=None, T_folder_path=None, hand_paths=None, camera_params_path=None):
        self.camera_params_path = camera_params_path
        
        self.NH_folder_path = [os.path.join(NH_folder_path, f) for f in os.listdir(NH_folder_path) 
                       if os.path.isfile(os.path.join(NH_folder_path, f)) and f.endswith('.obj')]

        self.T_folder_path = [os.path.join(T_folder_path, f) for f in os.listdir(T_folder_path) 
                        if os.path.isfile(os.path.join(T_folder_path, f)) and f.endswith('.obj')]
        
        self.hand_paths = [os.path.join(hand_paths, f) for f in os.listdir(hand_paths) 
                        if os.path.isfile(os.path.join(hand_paths, f)) and f.endswith('.obj')]
        
        self.NH_output_dirs = []
        self.T_output_dirs = []
        self.make_dirs()
        
    def make_dirs(self):
        ''' Create NH<i>_output and T<j>_output dirs '''
        current_directory = os.getcwd()
        for o_path in self.NH_folder_path:
            base_name = os.path.splitext(os.path.basename(o_path))[0]
            new_directory_name = f"{base_name}_output"
            new_directory_path = os.path.join(current_directory, new_directory_name)
            os.makedirs(new_directory_path, exist_ok=True)
            self.NH_output_dirs.append(new_directory_path)
            print(f"Directory '{new_directory_name}' created at '{new_directory_path}'.")
            
        # for o_path in self.T_folder_path:
        #     base_name = os.path.splitext(os.path.basename(o_path))[0]
        #     new_directory_name = f"{base_name}_output"
        #     new_directory_path = os.path.join(current_directory, new_directory_name)
        #     os.makedirs(new_directory_path, exist_ok=True)
        #     self.T_output_dirs.append(new_directory_path)
        #     print(f"Directory '{new_directory_name}' created at '{new_directory_path}'.")
            

    def render_and_save_coco(self, generate_data=False, delete_data=False, palm_obj_path=None,
                             large_palm_obj_path=None, arm_obj_path=None, camera_params=None):
        if generate_data:
            for NH_path, output_path in zip(self.NH_folder_path, self.NH_output_dirs):
                for T_path in self.T_folder_path:
                    args = [
                        '--palm_obj', palm_obj_path,
                        '--large_palm_obj', large_palm_obj_path,
                        '--arm_obj', arm_obj_path,
                        '--needle_holder_obj', NH_path,
                        '--tweezers_obj', T_path,
                        '--output_dir', output_path,
                        '--camera_params', camera_params
                    ]
                    command = ['blenderproc', 'run', 'main.py'] + args
                    subprocess.run(command)
                    
        if delete_data:
            for NH_path in self.NH_output_dirs:
                if os.path.exists(NH_path):
                    try:
                        shutil.rmtree(NH_path)
                        print(f"Deleted: {NH_path}")
                    except Exception as e:
                        print(f"Failed to delete {NH_path}: {e}")
                else:
                    print(f"Directory not found: {NH_path}")
                    
            for T_path in self.T_output_dirs:
                if os.path.exists(T_path):
                    try:
                        shutil.rmtree(T_path)
                        print(f"Deleted: {T_path}")
                    except Exception as e:
                        print(f"Failed to delete {T_path}: {e}")
                else:
                    print(f"Directory not found: {T_path}")
                    
    def post_process(self):
        print('Running change_ids.py...')
        subprocess.run(['python', 'change_ids.py'])
        
        print('Running synthetic_data_generator.py...')
        subprocess.run(['python', 'organize.py'])
        
        print('Running coco_to_yolo.py...')
        subprocess.run(['python', 'coco_to_yolo.py'])
        
        # print('training exp3.py...')
        # subprocess.run(['python', '/home/student/project/mlflow/finetuning/exp3/exp3.py'])

pipeline = Pipeline(NH_folder_path='/datashare/project/surgical_tools_models/needle_holder',
                    T_folder_path='/datashare/project/surgical_tools_models/tweezers',
                    camera_params_path='/home/student/project/camera.json',
                    hand_paths='/home/student/project/mlflow/synth_data/Hand_objects')


pipeline.render_and_save_coco(generate_data=True, delete_data=False, 
                    palm_obj_path='/home/student/project/mlflow/synth_data/Hand_objects/hand2.obj',
                    large_palm_obj_path='/home/student/project/mlflow/synth_data/Hand_objects/hand1.obj',
                    arm_obj_path='/home/student/project/mlflow/synth_data/Hand_objects/arm.obj',
                    camera_params='/home/student/project/camera.json')
pipeline.post_process()


