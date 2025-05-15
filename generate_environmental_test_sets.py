import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
import torch
from tqdm import tqdm

# === Environmental condition transforms ===
condition_transforms = {
    "brightness": A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.1, p=1.0),
    "motion_blur": A.MotionBlur(blur_limit=(3, 5), p=1.0),
    "background_complexity": A.OneOf([
        A.RandomFog(fog_coef_range=(0.1, 0.2), alpha_coef=0.05, p=0.5),
        A.RandomRain(drop_length=15, drop_width=1, blur_value=2, brightness_coefficient=0.9, p=0.5),
        A.RandomShadow(num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, p=0.5),
    ], p=1.0),
}

# === Read YOLO format bounding boxes ===
def read_yolo_bbox(label_path):
    with open(label_path, "r") as f:
        return [list(map(float, line.strip().split())) for line in f.readlines()]

# === Save YOLO format bounding boxes ===
def save_yolo_bbox(label_path, bboxes):
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    with open(label_path, "w") as f:
        for bbox in bboxes:
            bbox = [max(0, min(1, val)) for val in bbox]
            f.write(" ".join(map(str, bbox)) + "\n")

# === Apply transformation ===
def apply_condition(image, bboxes, class_labels, transform):
    clipped_bboxes = []
    for bbox in bboxes:
        clipped = [max(0.0, min(1.0, val)) for val in bbox]
        clipped_bboxes.append(clipped)

    composed = A.Compose(
        [transform],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])
    )
    
    return composed(image=image, bboxes=clipped_bboxes, class_labels=class_labels)

# === Generate a single dataset variation ===
def generate_condition_dataset(image_dir, label_dir, output_dir, condition_name, transform):
    image_output_dir = os.path.join(output_dir, f"dataset_test_{condition_name}", "images")
    label_output_dir = os.path.join(output_dir, f"dataset_test_{condition_name}", "labels")
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(label_output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

    for filename in tqdm(image_files, desc=f"[{condition_name}] {os.path.basename(image_dir)}"):
        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, filename.replace(".jpg", ".txt"))

        if not os.path.exists(label_path):
            continue

        image = cv2.imread(image_path)
        if image is None:
            continue

        bboxes = read_yolo_bbox(label_path)
        if len(bboxes) == 0:
            continue

        class_labels = [int(b[0]) for b in bboxes]
        bbox_list = [b[1:] for b in bboxes]

        try:
            result = apply_condition(image, bbox_list, class_labels, transform)
        except Exception as e:
            print(f"❌ Failed on {filename} due to {e}")
            continue

        aug_image = result["image"]
        if isinstance(aug_image, torch.Tensor):
            aug_image = aug_image.permute(1, 2, 0).cpu().numpy()

        new_img_path = os.path.join(image_output_dir, filename)
        new_label_path = os.path.join(label_output_dir, filename.replace(".jpg", ".txt"))

        cv2.imwrite(new_img_path, aug_image)
        clipped_bboxes = [[class_labels[i]] + [max(0, min(1, coord)) for coord in result["bboxes"][i]] for i in range(len(class_labels))]
        save_yolo_bbox(new_label_path, clipped_bboxes)

# === Generate all environmental condition variants ===
def generate_all_conditions(original_image_dir, original_label_dir, base_output_dir):
    for condition, transform in condition_transforms.items():
        class_output_dir = os.path.join(base_output_dir)
        generate_condition_dataset(original_image_dir, original_label_dir, class_output_dir, condition, transform)

# === Main Entrypoint ===
if __name__ == "__main__":
    dataset_sets = [
        {
            "name": "whole",
            "image_dir": "dataset_original/whole/train/images",
            "label_dir": "dataset_original/whole/train/labels",
            "output_base": "dataset_environment_tests/whole"
        },
        {
            "name": "break",
            "image_dir": "dataset_original/break/train/images",
            "label_dir": "dataset_original/break/train/labels",
            "output_base": "dataset_environment_tests/break"
        }
    ]

    for ds in dataset_sets:
        print(f"\n📁 Processing class: {ds['name']}")
        generate_all_conditions(ds["image_dir"], ds["label_dir"], ds["output_base"])
