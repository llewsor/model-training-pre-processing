import os
import json
import cv2

# === CONFIGURATION ===
# DATASET_PATHS = {
#     "train": {"images": "dataset/train/images", "labels": "dataset/train/labels"},
#     "val": {"images": "dataset/val/images", "labels": "dataset/val/labels"},
#     # "test": {"images": "dataset/test/images", "labels": "dataset/test/labels"},
# }

# OUTPUT_JSON_FILES = {
#     "train": "dataset/train/annotations.json",
#     "val": "dataset/val/annotations.json",
#     # "test": "dataset/test/annotations.json",
# }

DATASET_PATHS = {
    "brightness": {
        "images": "dataset_environment_tests/dataset_test_brightness/images", 
        "labels": "dataset_environment_tests/dataset_test_brightness/labels"},
    "background": {
        "images": "dataset_environment_tests/dataset_test_background_complexity/images", 
        "labels": "dataset_environment_tests/dataset_test_background_complexity/labels"},
    "motion": {
        "images": "dataset_environment_tests/dataset_test_motion_blur/images", 
        "labels": "dataset_environment_tests/dataset_test_motion_blur/labels"},
}

OUTPUT_JSON_FILES = {
    "brightness": "dataset_environment_tests/dataset_test_brightness/annotations.json",
    "background": "dataset_environment_tests/dataset_test_background_complexity/annotations.json",
    "motion": "dataset_environment_tests/dataset_test_motion_blur/annotations.json",
}

CLASS_NAMES = ["whole", "broken"]  # Change this to match your YOLO classes

# === FUNCTION TO CONVERT YOLO TO COCO ===
def convert_yolo_to_coco(image_dir, label_dir, output_json):
    coco_dict = {"images": [], "annotations": [], "categories": []}
    
    # Define COCO categories
    for idx, name in enumerate(CLASS_NAMES):
        coco_dict["categories"].append({
            "id": idx + 1,  # COCO category IDs start from 1
            "name": name,
            "supercategory": "none"
        })

    annotation_id = 1  # Unique ID for each annotation
    image_id = 1       # Unique ID for each image

    # Loop through label files
    for label_file in sorted(os.listdir(label_dir)):
        if not label_file.endswith(".txt"):
            continue

        image_name = label_file.replace(".txt", ".jpg")  # Change if images are PNG
        image_path = os.path.join(image_dir, image_name)

        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found. Skipping...")
            continue

        # Load image to get width and height
        image = cv2.imread(image_path)
        height, width, _ = image.shape

        # Add image metadata
        coco_dict["images"].append({
            "id": image_id,
            "file_name": image_name,
            "width": width,
            "height": height
        })

        # Read YOLO annotation file
        with open(os.path.join(label_dir, label_file), "r") as file:
            lines = file.readlines()

        # Convert YOLO annotations to COCO
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0]) + 1  # COCO category IDs start from 1
            x_center, y_center, w, h = map(float, parts[1:])

            # Convert YOLO bbox to COCO bbox format
            x_min = (x_center - w / 2) * width
            y_min = (y_center - h / 2) * height
            bbox_width = w * width
            bbox_height = h * height

            # Save annotation
            coco_dict["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id,
                "bbox": [x_min, y_min, bbox_width, bbox_height],
                "area": bbox_width * bbox_height,
                "iscrowd": 0
            })

            annotation_id += 1

        image_id += 1

    # Save to JSON
    with open(output_json, "w") as json_file:
        json.dump(coco_dict, json_file, indent=4)

    print(f"✅ Converted YOLO to COCO: {output_json}")

def main():
    # === RUN CONVERSION FOR train, val, test ===
    for dataset_type, paths in DATASET_PATHS.items():
        convert_yolo_to_coco(paths["images"], paths["labels"], OUTPUT_JSON_FILES[dataset_type])

if __name__ == "__main__":
    main()