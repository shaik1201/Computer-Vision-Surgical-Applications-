import json
import os

def change_id(file_path):
    # Check if the file exists and is readable
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    if os.path.getsize(file_path) == 0:
        print(f"File is empty: {file_path}")
        return
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Filter out annotations with category_id 3 and change IDs for others
        data['annotations'] = [
            annotation for annotation in data['annotations']
            if annotation['category_id'] != 3 and annotation['category_id'] != 4 and annotation['category_id'] != 5 and annotation['category_id'] != 6 and annotation['category_id'] != 7
        ]
        
        # Change IDs for remaining annotations
        for annotation in data['annotations']:
            if annotation['category_id'] == 1:
                annotation['category_id'] = 0
            elif annotation['category_id'] == 2:
                annotation['category_id'] = 1
        
        # Write the modified data back to the file
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)  # Write with indentation for readability

        print(f"Successfully updated IDs in: {file_path}")

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

dirs = os.listdir('/home/student/project/mlflow/synth_data/main/')
dirs = [d for d in dirs if d.endswith('output')]
for d in dirs:
    change_id(f'/home/student/project/mlflow/synth_data/main/{d}/coco_data/coco_annotations.json')
# change_id('/home/student/project/mlflow/synth_data/main/mock_annotations.json')