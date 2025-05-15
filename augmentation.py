import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import os
import torch
from tqdm import tqdm
import itertools

# ✅ Define all individual augmentations with unique names
augmentations = [
    ("brightness", A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0)),
    ("motion_blur", A.MotionBlur(blur_limit=(3, 3), p=1.0)),
    ("gauss_noise", A.GaussNoise(std_range=(0.05, 0.15), p=1.0)),
    # ("hsv_shift", A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=1.0)),
    ("rotate", A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=1.0, fill=1)),
    ("scale", A.RandomScale(scale_limit=0.2, p=1.0)),
    # ("safe_crop", A.RandomSizedBBoxSafeCrop(width=640, height=640, p=1.0)),
    # ("fog", A.RandomFog(fog_coef_range=(0.1, 0.1), alpha_coef=0.01, p=1.0)),
    ("rain", A.RandomRain(blur_value=1, brightness_coefficient=0.8, p=1.0)),
    # ("distortion", A.OpticalDistortion(p=1.0)),
]

# ✅ Load YOLO format bounding boxes
def read_yolo_bbox(label_path):
    with open(label_path, "r") as f:
        bboxes = [list(map(float, line.strip().split())) for line in f.readlines()]
    return bboxes

# ✅ Save YOLO format bounding boxes (clip values to [0, 1])
def save_yolo_bbox(label_path, bboxes):
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    with open(label_path, "w") as f:
        for bbox in bboxes:
            bbox = [max(0, min(1, val)) for val in bbox]
            f.write(" ".join(map(str, bbox)) + "\n")

# ✅ Compose and apply a specific list of augmentations
def apply_combination(image, bboxes, class_labels, transforms):
    composed = A.Compose(transforms, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))
    return composed(image=image, bboxes=bboxes, class_labels=class_labels)

# ✅ Perform all augmentation combinations (1 to N) for a given image
# def augment_image(image_path, label_path, output_dir):
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"Skipping {image_path}: Unable to read image.")
#         return

#     # Load bounding boxes and class labels
#     bboxes = read_yolo_bbox(label_path)
#     if len(bboxes) == 0:
#         return

#     bbox_list = [bbox[1:] for bbox in bboxes]  # Remove class ID
#     class_labels = [int(bbox[0]) for bbox in bboxes]
#     filename = os.path.splitext(os.path.basename(image_path))[0]

#     # Generate all combinations from 1 to N augmentations
#     # ✅ Compute total combinations for logging
#     total_combos = sum([len(list(itertools.combinations(augmentations, n))) for n in range(1, len(augmentations)+1)])
#     combo_iter = (combo for n in range(1, len(augmentations)+1) for combo in itertools.combinations(augmentations, n))

#     for combo in tqdm(combo_iter, total=total_combos, desc=f"↳ {filename}"):
#         combo_names = [name for name, _ in combo]
#         combo_transforms = [aug for _, aug in combo]

#         try:
#             augmented = apply_combination(image, bbox_list, class_labels, combo_transforms)
#         except Exception as e:
#             print(f"❌ Skipped {filename} with {combo_names} due to error: {e}")
#             continue

#         # Convert back to NumPy array if needed
#         aug_image = augmented["image"]
#         if isinstance(aug_image, torch.Tensor):
#             aug_image = aug_image.permute(1, 2, 0).cpu().numpy()

#         # Prepare output directories
#         image_output_dir = os.path.join(output_dir, "images")
#         label_output_dir = os.path.join(output_dir, "labels")
#         os.makedirs(image_output_dir, exist_ok=True)
#         os.makedirs(label_output_dir, exist_ok=True)

#         # Construct filenames using combo names
#         suffix = "_".join(combo_names)
#         new_img_path = os.path.join(image_output_dir, f"{filename}_{suffix}.jpg")
#         new_bbox_path = os.path.join(label_output_dir, f"{filename}_{suffix}.txt")

#         # Clip bboxes and save both image and label
#         cv2.imwrite(new_img_path, aug_image)
#         clipped_bboxes = [[class_labels[i]] + [max(0, min(1, coord)) for coord in augmented["bboxes"][i]] for i in range(len(class_labels))]
#         save_yolo_bbox(new_bbox_path, clipped_bboxes)

def augment_image(image_path, label_path, output_dir):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Skipping {image_path}: Unable to read image.")
        return

    # Load bounding boxes and class labels
    bboxes = read_yolo_bbox(label_path)
    if len(bboxes) == 0:
        return

    bbox_list = [bbox[1:] for bbox in bboxes]
    class_labels = [int(bbox[0]) for bbox in bboxes]
    filename = os.path.splitext(os.path.basename(image_path))[0]

    # ✅ Map augmentations for lookup
    aug_dict = {name: aug for name, aug in augmentations}

    # ✅ Define specific augment combos (6 singles + 4 combos)
    combo_list = [
        ["brightness"],
        ["motion_blur"],
        ["gauss_noise"],
        ["rotate"],
        ["scale"],
        ["rain"],
        ["brightness", "rain"],
        ["scale", "rain"],
        ["rotate", "scale"],
        ["brightness", "scale"],
    ]

    for combo_names in tqdm(combo_list, desc=f"↳ {filename}", leave=False):
        combo_transforms = [aug_dict[name] for name in combo_names]

        try:
            augmented = apply_combination(image, bbox_list, class_labels, combo_transforms)
        except Exception as e:
            print(f"❌ Skipped {filename} with {combo_names} due to error: {e}")
            continue

        # Convert to NumPy array if needed
        aug_image = augmented["image"]
        if isinstance(aug_image, torch.Tensor):
            aug_image = aug_image.permute(1, 2, 0).cpu().numpy()

        # Prepare output directories
        image_output_dir = os.path.join(output_dir, "images")
        label_output_dir = os.path.join(output_dir, "labels")
        os.makedirs(image_output_dir, exist_ok=True)
        os.makedirs(label_output_dir, exist_ok=True)

        # Save files
        suffix = "_".join(combo_names)
        new_img_path = os.path.join(image_output_dir, f"{filename}_{suffix}.jpg")
        new_bbox_path = os.path.join(label_output_dir, f"{filename}_{suffix}.txt")

        clipped_bboxes = [[class_labels[i]] + [max(0, min(1, coord)) for coord in augmented["bboxes"][i]] for i in range(len(class_labels))]
        cv2.imwrite(new_img_path, aug_image)
        save_yolo_bbox(new_bbox_path, clipped_bboxes)


# ✅ Batch process all images in a dataset folder
def process_all_images(image_dir, label_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

    for filename in tqdm(image_files, desc="Processing Images"):
        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, filename.replace(".jpg", ".txt"))

        if os.path.exists(label_path):
            augment_image(image_path, label_path, output_dir)

# ✅ Run the batch processor
# Example:
process_all_images("dataset/break/train/images", "dataset/break/train/labels", "dataset_break_aug")
process_all_images("dataset/whole/train/images", "dataset/whole/train/labels", "dataset_whole_aug")
