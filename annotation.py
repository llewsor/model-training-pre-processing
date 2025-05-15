import cv2
import numpy as np
import os
import random
import concurrent.futures
import xml.etree.ElementTree as ET
from pathlib import Path
import datetime

# --- CONFIGURATIONS ---
VIDEO_PATH = "videos_batch_1/mobile-front-center-forward-miss.mp4"  
# VIDEO_PATH = "videos_batch_1/center-top-Miss.mp4"  
# VIDEO_PATH = "videos_batch_1/center-top-highMiss.mp4"  
# VIDEO_PATH = "videos_batch_1/center-facing-highHouse-doubleMiss.mp4"  

XML_TRAJECTORY_PATH = "trajectories.xml"  # Path to the XML trajectory file
OUTPUT_DIR = "datasets/whole"
FRAME_SKIP = 1  # Adjust (5 = ~24 FPS, 10 = ~12 FPS)
MOTION_THRESHOLD = 0.30  # Adjust for more/fewer frames
ROI_MARGIN = 20  # Extra padding for ROI

# Adjust dataset splitting ratios
TRAIN_SPLIT = 0.8  # 70% training
VAL_SPLIT = 0.2  # 15% validation
# TEST_SPLIT = 0  # 15% test

# Ensure the splits sum to 1
# assert TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT == 1.0, "Splits must sum to 1!"
assert TRAIN_SPLIT + VAL_SPLIT == 1.0, "Splits must sum to 1!"

AREA_MIN = 300  # Minimum contour area for detection
AREA_MAX = 5000  # Maximum contour area for filtering noise

# YOLO Class ID (Adjust if needed)
CLASS_ID = 0  # 0 = whole clay target (can be changed for multiple classes)

# Background Subtractor (MOG2)
object_detector = cv2.createBackgroundSubtractorMOG2(history=60, varThreshold=100, detectShadows=False)

# Create output directories with images and labels subfolders
def create_dirs(base):
  os.makedirs(os.path.join(base, "images"), exist_ok=True)
  os.makedirs(os.path.join(base, "labels"), exist_ok=True)
    
# Create output directories
train_dir = os.path.join(OUTPUT_DIR, "train")
val_dir = os.path.join(OUTPUT_DIR, "val")
# test_dir = os.path.join(OUTPUT_DIR, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
# os.makedirs(test_dir, exist_ok=True)
create_dirs(train_dir)
create_dirs(val_dir)
# create_dirs(test_dir)

# Load video
cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Loading video {fps}fps, width {width}px, height {height}px")

# --- Load Trajectories from XML ---
def load_trajectories_from_xml(filename):
  if not Path(filename).exists():
    print("Error: XML file not found.")
    return None

  tree = ET.parse(filename)
  root = tree.getroot()
  trajectories = {}

  for track in root.findall("Track"):
    track_id = int(track.get("id", -1))
    if track_id == -1:
      continue
    
    trajectories[track_id] = []
    for point in track.findall("Point"):
      x, y = int(point.get("x")), int(point.get("y"))
      trajectories[track_id].append((x, y))

  print(f"✅ Loaded {len(trajectories)} targets from XML")
  return trajectories

# --- Compute ROI from Trajectories ---
def get_roi_from_trajectories(trajectories, margin=ROI_MARGIN):
  if not trajectories:
      return None  # No valid trajectory data

  all_points = [point for track in trajectories.values() for point in track]
  if not all_points:
      return None

  x_values = [p[0] for p in all_points]
  y_values = [p[1] for p in all_points]

  x_min, x_max = max(0, min(x_values) - margin), min(width, max(x_values) + margin)
  y_min, y_max = max(0, min(y_values) - margin), min(height, max(y_values) + margin)

  return x_min, y_min, x_max, y_max

# Load trajectories and compute ROI
trajectories = load_trajectories_from_xml(XML_TRAJECTORY_PATH)
region_of_interest = get_roi_from_trajectories(trajectories) if trajectories else (0, 0, width, height)
x_min, y_min, x_max, y_max = region_of_interest
print(f"Computed ROI: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")

# Extract video name (without extension)
video_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]

# Get current date and time in YYYYMMDD_HHMMSS format
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Function to save frames and YOLO annotations
def save_frame_and_annotation(frame, frame_num, bboxes, dataset_type):
  """
  Saves the frame and corresponding YOLO annotation file.

  Args:
  - frame (numpy array): The image frame to save.
  - frame_num (int): The frame number.
  - bboxes (list): List of bounding boxes in (x, y, w, h) format.
  - dataset_type (str): One of "train", "val", or "test" to determine storage directory.
  """

  # Select the appropriate base folder
  if dataset_type == "train":
    base_folder = train_dir
  elif dataset_type == "val":
    base_folder = val_dir
  # elif dataset_type == "test":
  #   base_folder = test_dir
  else:
    raise ValueError("Invalid dataset_type. Use 'train', 'val', or 'test'.")

  image_folder = os.path.join(base_folder, "images")
  label_folder = os.path.join(base_folder, "labels")

  # Format the frame number as a 5-digit number
  frame_str = f"{frame_num:05d}"

  # Construct new file name format (timestamp + video + frame number)
  file_base_name = f"{timestamp}_{video_name}_frame_{frame_str}"

  # Ensure at least one bounding box exists
  if not bboxes:
    return

  for i, (x, y, w, h) in enumerate(bboxes):
    # Compute the center of the bounding box
    x_center = x + w // 2
    y_center = y + h // 2

    # Define cropping boundaries (640x640) centered on the target
    crop_x_min = max(0, x_center - 320)
    crop_x_max = min(width, crop_x_min + 640)
    crop_y_min = max(0, y_center - 320)
    crop_y_max = min(height, crop_y_min + 640)

    # Adjust crop if it exceeds image boundaries
    if crop_x_max - crop_x_min < 640:
      crop_x_min = max(0, crop_x_max - 640)
    if crop_y_max - crop_y_min < 640:
      crop_y_min = max(0, crop_y_max - 640)

    # Crop the image around the primary target
    cropped_frame = frame[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

    # Ensure the cropped image is exactly 640x640
    if cropped_frame.shape[0] != 640 or cropped_frame.shape[1] != 640:
      cropped_frame = cv2.resize(cropped_frame, (640, 640))

    # Generate filenames
    image_filename = os.path.join(image_folder, f"{file_base_name}_{i}.jpg")
    label_filename = os.path.join(label_folder, f"{file_base_name}_{i}.txt")

    # Save the cropped image
    cv2.imwrite(image_filename, cropped_frame)

    # Find all bounding boxes that fall inside this 640x640 frame
    with open(label_filename, "w") as f:
      for x_other, y_other, w_other, h_other in bboxes:
        # Check if this bounding box is inside the cropped frame
        x_other_center = x_other + w_other // 2
        y_other_center = y_other + h_other // 2

        if (crop_x_min <= x_other_center <= crop_x_max) and (crop_y_min <= y_other_center <= crop_y_max):
          # Normalize the bounding box to the cropped 640x640 frame
          norm_x_center = (x_other_center - crop_x_min) / 640
          norm_y_center = (y_other_center - crop_y_min) / 640
          norm_w = w_other / 640
          norm_h = h_other / 640

          # Ensure values are between 0 and 1
          norm_x_center = max(0, min(1, norm_x_center))
          norm_y_center = max(0, min(1, norm_y_center))
          norm_w = max(0, min(1, norm_w))
          norm_h = max(0, min(1, norm_h))

          # Write the annotation (all targets inside this cropped image)
          f.write(f"{CLASS_ID} {norm_x_center:.6f} {norm_y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

frame_count = 0
saved_frames = []
ret, prev_frame = cap.read()

if not ret:
  print("Error: Could not read video.")
  cap.release()
  exit()

# Apply background subtraction for motion detection
fg_mask = object_detector.apply(prev_frame[y_min:y_max, x_min:x_max])

while True:
  ret, frame = cap.read()
  if not ret:
    break

  output = frame.copy()

  frame_count += 15
  
  # Skip frames based on the interval
  if frame_count % FRAME_SKIP != 0:
      continue

  roi = frame[y_min:y_max, x_min:x_max]  # Crop frame to ROI

  cv2.rectangle(output, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)  # Draw ROI
  
  # Apply background subtraction within ROI
  fg_mask = object_detector.apply(roi)

  # Find contours in the ROI
  contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Detect clay target & create bounding box if motion is significant
  bboxes = []
  for cnt in contours:
    area = cv2.contourArea(cnt)

    # Apply area filtering
    if AREA_MIN < area < AREA_MAX:
      x, y, w, h = cv2.boundingRect(cnt)

      # Convert ROI bounding box to full-frame coordinates
      x_full = x + x_min
      y_full = y + y_min

      # Ensure bounding box is within full frame dimensions
      if x_full + w <= width and y_full + h <= height:
        bboxes.append((x_full, y_full, w, h))
        cv2.rectangle(output, (x_full, y_full), (x_full + w, y_full + h), (0, 255, 0), 2)

  # Save all detected objects if at least one is found
  if bboxes:
    saved_frames.append((frame.copy(), frame_count, bboxes))

  cv2.imshow("Detection", cv2.resize(output, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()

# # Shuffle and split frames into train, val, and test
# random.shuffle(saved_frames)
# train_end = int(len(saved_frames) * TRAIN_SPLIT)
# val_end = train_end + int(len(saved_frames) * VAL_SPLIT)

# train_frames = saved_frames[:train_end]
# val_frames = saved_frames[train_end:val_end]
# test_frames = saved_frames[val_end:]
train_frames = saved_frames[int(len(saved_frames))]


# Save frames using multi-threading
with concurrent.futures.ThreadPoolExecutor() as executor:
  futures = []

  # Train frames
  for frame, frame_num, bboxes in train_frames:
    futures.append(executor.submit(save_frame_and_annotation, frame, frame_num, bboxes, "train"))

  # # Validation frames
  # for frame, frame_num, bboxes in val_frames:
  #   futures.append(executor.submit(save_frame_and_annotation, frame, frame_num, bboxes, "val"))

  # # Test frames (saved separately)
  # for frame, frame_num, bboxes in test_frames:
  #   futures.append(executor.submit(save_frame_and_annotation, frame, frame_num, bboxes, "test"))

# Wait for all threads to complete
concurrent.futures.wait(futures)

print(f"✅ Extraction completed. {len(saved_frames)} frames saved.")
# print(f"📂 Train frames: {len(train_frames)}, Validation frames: {len(val_frames)}, Test frames: {len(test_frames)}")
print(f"📂 Train frames: {len(train_frames)}")