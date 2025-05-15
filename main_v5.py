import os
import cv2
import numpy as np
import time
import datetime
import xml.etree.ElementTree as ET
from pathlib import Path
from tracker import EuclideanDistTracker
from scipy.spatial import distance
import torch
# from ultralytics import YOLO

# OUTPUT_DIR = "annotations"
OUTPUT_DIR = "whole"

# Define class names
CLASS_NAMES = {
    0: "Whole Target",
    1: "Broken Target",
}

# Define fixed colors for each class (BGR format for OpenCV)
CLASS_COLORS = {
    0: (0, 0, 255),   # Whole Target (Red)
    1: (0, 255, 0),   # Broken Target (Green)
}

def load_trajectories_from_xml(video_filename, filename="trajectories.xml"):
    if not Path(filename).exists():
        print(f"❌ File does not exist: {filename}")
        return None
    
    tree = ET.parse(filename)
    root = tree.getroot()
    
    video_elem = next((video for video in root.findall("Video") if video.get("name") == video_filename), None)
    if not video_elem:
        print(f"❌ Video '{video_filename}' not found in XML.")
        return None
    
    frame_info = video_elem.find("FrameInfo")
    frame_width, frame_height = map(int, (frame_info.get("width", 0), frame_info.get("height", 0)))
    area_info = video_elem.find("TargetArea")
    area_min, area_max = map(int, (area_info.get("min", 0), area_info.get("max", 0)))
    
    trajectories = {
        int(track.get("id")): [(int(point.get("x")), int(point.get("y"))) for point in track.findall("Point")]
        for track in video_elem.findall("Track") if track.get("id")
    }
    
    print(f"✅ Loaded {len(trajectories)} targets for '{video_filename}'; Resolution: {frame_width}x{frame_height}; Area min: {area_min}, max: {area_max}")
    return trajectories, frame_width, frame_height, area_min, area_max

def get_roi_from_trajectories(trajectories, margin=20):
    if not trajectories:
        return None
    
    all_points = [point for track in trajectories.values() for point in track]
    if not all_points:
        return None
    
    x_values, y_values = zip(*all_points)
    return max(0, min(x_values) - margin), max(0, min(y_values) - margin), max(x_values) + margin, max(y_values) + margin

def ensure_directories():
    os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/labels", exist_ok=True)

# Function to save frames and YOLO annotations
def save_frame_and_annotation(frame, width, height, frame_num, bboxes):
    """
    Saves the frame and corresponding YOLO annotation file.

    Args:
    - frame (numpy array): The image frame to save.
    - frame_num (int): The frame number.
    - bboxes (list): List of bounding boxes in (x, y, w, h) format.
    """
    ensure_directories()
    image_folder = os.path.join(OUTPUT_DIR, "images")
    label_folder = os.path.join(OUTPUT_DIR, "labels")

    # Format the frame number as a 5-digit number
    frame_str = f"{frame_num:05d}"

    # Get current date and time in YYYYMMDD_HHMMSS format
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Construct new file name format (timestamp + video + frame number)
    file_base_name = f"{timestamp}_frame_{frame_str}"

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
                    # f.write(f"{1} {norm_x_center:.6f} {norm_y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n") # broken
                    f.write(f"{0} {norm_x_center:.6f} {norm_y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n") # whole
        
        print(f"✅ Annotation saved: {image_filename}, {label_filename}")

def get_video_list(directory="videos_batch_1"):
    video_extensions = (".mp4", ".MP4")
    return sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(video_extensions)])

def load_video(video_index, video_list):
    if 0 <= video_index < len(video_list):
        cap = cv2.VideoCapture(video_list[video_index])
        if cap.isOpened():
            print(f"Loaded video {video_list[video_index]}")
            return cap
    return None

def save_frame(frame, frame_num):
    image_folder = "screenshots"
    os.makedirs(image_folder, exist_ok=True)
    
    # Format the frame number as a 5-digit number
    frame_str = f"{frame_num:05d}"

    # Get current date and time in YYYYMMDD_HHMMSS format
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Construct new file name format (timestamp + video + frame number)
    file_base_name = f"{timestamp}_frame_{frame_str}"

    image_filename = os.path.join(image_folder, f"{file_base_name}.jpg")

    # Save the cropped image
    cv2.imwrite(image_filename, frame)

def merge_boxes(boxes, threshold):
    merged = []
    used = set()

    for i, (x1, y1, w1, h1) in enumerate(boxes):
        if i in used:
            continue
        x2, y2 = x1 + w1, y1 + h1  # Bottom-right corner
        to_merge = [(x1, y1, x2, y2)]  # Start a new merged box

        for j, (xj, yj, wj, hj) in enumerate(boxes):
            if i != j and j not in used:
                xj2, yj2 = xj + wj, yj + hj

                # Compute distance between centers of bounding boxes
                center1 = ((x1 + x2) / 2, (y1 + y2) / 2)
                center2 = ((xj + xj2) / 2, (yj + yj2) / 2)
                dist = distance.euclidean(center1, center2)

                if dist < threshold:  # Merge if within distance
                    to_merge.append((xj, yj, xj2, yj2))
                    used.add(j)

        # Compute merged bounding box
        x_min = min([b[0] for b in to_merge])
        y_min = min([b[1] for b in to_merge])
        x_max = max([b[2] for b in to_merge])
        y_max = max([b[3] for b in to_merge])
        merged.append((x_min, y_min, x_max - x_min, y_max - y_min))

        used.add(i)
        
    return merged

def classify_target(frame, x, y, w, h, model, width, height):
    """
    Extracts and classifies a detected target using the trained model.

    Args:
    - frame (numpy array): The video frame.
    - x, y, w, h (int): Bounding box coordinates.
    - model: Trained YOLO or other ML model.

    Returns:
    - Label and confidence score.
    """
     # Compute the center of the bounding box
    x_center = x + w // 2
    y_center = y + h // 2

    CROPPING_BOUNDARIES = 640 #(640X640)
    
    # Define cropping boundaries centered on the target
    crop_x_min = max(0, x_center - int(CROPPING_BOUNDARIES/2))
    crop_x_max = min(width, crop_x_min + CROPPING_BOUNDARIES)
    crop_y_min = max(0, y_center - int(CROPPING_BOUNDARIES/2))
    crop_y_max = min(height, crop_y_min + CROPPING_BOUNDARIES)

    # Adjust crop if it exceeds image boundaries
    if crop_x_max - crop_x_min < CROPPING_BOUNDARIES:
        crop_x_min = max(0, crop_x_max - CROPPING_BOUNDARIES)
    if crop_y_max - crop_y_min < CROPPING_BOUNDARIES:
        crop_y_min = max(0, crop_y_max - CROPPING_BOUNDARIES)

    # Crop the image around the primary target
    cropped_frame = frame[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

    # Ensure the cropped image is exactly 640x640
    if cropped_frame.shape[0] != CROPPING_BOUNDARIES or cropped_frame.shape[1] != CROPPING_BOUNDARIES:
        cropped_frame = cv2.resize(cropped_frame, (CROPPING_BOUNDARIES, CROPPING_BOUNDARIES))
    
    # Run the model
    # img_tensor = torch.tensor(cropped_frame).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0 # Convert to tensor and move to GPU

    # results = model(img_tensor)
    results = model(cropped_frame, imgsz=CROPPING_BOUNDARIES)
    predictions = results[0]

    # Extract class and confidence
    if len(predictions.boxes) > 0:
        pred_box = predictions.boxes[0]  # Get the first prediction
        class_id = int(pred_box.cls.item())  # Class index
        confidence = float(pred_box.conf.item())  # Confidence score
        return class_id, confidence
    
    return None, None

device = "cuda" if torch.cuda.is_available() else "cpu"
# Load the model
MODEL_PATH = "yolo_v11_x_run_1.pt"
# MODEL_PATH = "yolo_v11_m_run_1.pt"
# MODEL_PATH = "yolo_v11_n_run_2.pt"
# model = YOLO(MODEL_PATH)  # Load YOLO model
# model = YOLO(MODEL_PATH).to(device)  # Load YOLO model

# model.to(device)
# print(f"Using device: {device}")

def process_video(video_index, video_list):
    cap = load_video(video_index, video_list)
    if not cap:
        return
    
    tracker = EuclideanDistTracker()
    object_detector = cv2.createBackgroundSubtractorMOG2(history=60, varThreshold=100, detectShadows=False)
    
    paused = False  # Variable to track if the video is paused
    frame_index = 0  # Track current frame index
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frames in video


    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video Info: {fps} FPS, {width}x{height}")
    
    
    global actual, predicted, frame_width, frame_height, area_min, area_max
    x_min, y_min, x_max, y_max = 0, 0, width, height
    region_of_interest = None
    area_min, area_max = None, None
    
    # Load trajectories
    storage = load_trajectories_from_xml(os.path.basename(video_list[video_index]))
    if (storage != None):
        trajectories, frame_width, frame_height, area_min, area_max = storage

        # Compute ROI
        region_of_interest = get_roi_from_trajectories(trajectories) if trajectories else None
        x_min, y_min, x_max, y_max = region_of_interest
    else:
        # Default ROI to top half of the frame
        region_of_interest = 0, 0, width, height // 2
        x_min, y_min, x_max, y_max = region_of_interest
        
    # Ensure ROI is within valid frame size
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(width, x_max)  # frame.shape[1] = width
    y_max = min(height, y_max)  # frame.shape[0] = height

    print(f"Computed ROI: x_min{x_min} x_max{x_max} y_min{y_min} y_max{y_max}")
    
    frame_times = []
    
    while True:
        if not paused or (paused):
            if paused:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                
            ret, frame = cap.read()
            
            # Restart video when it ends
            if not ret:
                print("Video finished. Press 'r' to restart. N for next and P for previous video")
                key = cv2.waitKey(0)  # Wait indefinitely for user input
                if key == ord('r'):
                    frame_index = 0
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index) # Restart video
                    tracker.reset()  # Clear trajectories and objects
                    object_detector = cv2.createBackgroundSubtractorMOG2(history=60, varThreshold=100, detectShadows=False)
                    continue
                elif key == ord('n'):  # Next video
                    video_index = (video_index + 1) % len(video_list)  # Loop around if at last video
                    if load_video(video_index, video_list):
                        cap.release()
                        cv2.destroyAllWindows()
                        process_video(video_index, video_list)
                elif key == ord('p'):  # Previous video
                    video_index = (video_index - 1) % len(video_list)  # Loop around if at first video
                    if load_video(video_index, video_list):
                        cap.release()
                        cv2.destroyAllWindows()
                        process_video(video_index, video_list)
                else:
                    break  # Exit if any other key is pressed
            
            start = time.perf_counter()        
            
            roi = frame.copy()  # If no valid ROI, use the full frame
            output = frame.copy()
            
            if region_of_interest:
                roi = frame[y_min:y_max, x_min:x_max]
                cv2.rectangle(output, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)  # Draw ROI

            # 1. Object Detection 
            mask = object_detector.apply(roi)
            # _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            detections = []
            
            alpha = 0.2  # Adjust transparency level (0.0 = fully transparent, 1.0 = solid)
            
            # Get bounding boxes for detected contours
            bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]

            # Distance threshold for merging (40 pixels)
            MERGE_DISTANCE = 100
            
            # Merge bounding boxes based on distance
            merged_bboxes = merge_boxes(bounding_boxes, MERGE_DISTANCE)

            for cnt in merged_bboxes:
                #Calculate area and remove small elements
                if isinstance(cnt, (tuple, list)) and len(cnt) == 4:
                    x, y, w, h = cnt
                    area = w * h  # Compute bounding box area
                else:
                    area = cv2.contourArea(cnt)  # Only for actual contours
                    
                if (area_min != None and (area > area_min and area < area_max) or (area_min == None and (fps < 31 and area > 100) or (fps < 121 and area > 50))):
                    if isinstance(cnt, (tuple, list)) and len(cnt) == 4:
                        x, y, w, h = cnt  # cnt is already a bounding box (x, y, w, h)
                    else:
                        x, y, w, h = cv2.boundingRect(cnt)  # Use boundingRect only for contours

                    detections.append([x + x_min, y + y_min, w, h, area])
                                
                    # aspect_ratio = float(w) / h
                    # if 0.2 <= aspect_ratio <= 1.8:  # Filter for nearly circular shapes
                    #   detections.append([x, y, w, h, area])
                    #   print(detections)
            
            # Update tracker
            boxes_ids = tracker.update(detections, fps) if len(detections) > 0 else []

            # Draw existing trajectories (always show them)
            for id, points in tracker.trajectories.items():
                for i in range(1, len(points)):
                    cv2.line(output, points[i - 1], points[i], (0, 0, 255), 2)  # Draw trajectory in red

            # Draw bounding boxes & object IDs
            zoom_boxes = []
            for box_id in boxes_ids:
                x, y, w, h, area, id = box_id
                x1 = x
                x2 = x + w
                y1 = y
                y2 = y + h
                
                # class_id, confidence = classify_target(frame, x, y, w, h, model, width, height)
                # label = ""
                color = (255, 0, 0)
                # if class_id is not None:
                #     class_name = CLASS_NAMES.get(class_id, "Unknown")
                #     color = CLASS_COLORS.get(class_id, (255, 255, 255))  # Default to white if class not found

                #     label = f"{class_name} ({confidence:.2f}); "
                
                # cv2.putText(output, f"{id}; {label}{str(w)} {str(h)} {area}", (x, y - 45), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 1)    
                cv2.rectangle(output, (x, y), (x + w, y + h), color, 1)
                
                 # Prepare text info
                # text = f"{id}; {label}| W: {w}px H: {h}px Area: {area}px2"
                text = f"{id};| W: {w}px H: {h}px Area: {area}px2"

                # Get text size
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_width, text_height = text_size

                # Determine if the bounding box is on the right side of the image
                if (x1 + x2) // 2 > output.shape[1] // 2:
                    # If the target is on the right side, shift the text to the left
                    text_x = max(5, x1 - text_width - 10)  # Ensure text stays inside the frame
                else:
                    # Otherwise, place the text on the right side
                    text_x = min(output.shape[1] - text_width - 5, x2 + 5)

                text_y = max(20, y1 - 10)  # Ensure text stays inside the frame vertically

                # Draw text background for readability
                cv2.rectangle(output, (text_x, text_y - text_height - 5), (text_x + text_width + 5, text_y + 5), color, -1)

                # Draw text
                cv2.putText(output, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                               
                tracker.capture(x, y, h, w, id, area)
                zoom_boxes.append((x, y, w, h, id, area))

            # Position for zoomed images at the bottom center
            snapshot_width = 400
            snapshot_height = 400
            max_snapshots = 2  # Number of snapshots displayed in a row
                    
            # Calculate the starting X position for centering
            zoom_x_start = (width - (snapshot_width * (max_snapshots * 2))) // 2  
            zoom_y_start = height - snapshot_height - 20  # 20px padding from bottom
                    
            for idx, (x, y, w, h, id, area) in enumerate(zoom_boxes):
                if idx >= max_snapshots:
                    break  # Limit the number of displayed snapshots
                
                # Calculate the center of the detected object
                center_x, center_y = x + w // 2, y + h // 2  

                # Extract from center to top and center to left
                x1 = max(0, center_x - snapshot_width // 2)
                y1 = max(0, center_y - snapshot_height // 2)
                x2 = min(width, center_x + snapshot_width // 2)
                y2 = min(height, center_y + snapshot_height // 2)

                # Extract the centered region from the frame
                zoomed1 = output[y1:y2, x1:x2]

                # Resize if necessary (only if the extracted region is smaller than the snapshot size)
                zoomed1 = cv2.resize(zoomed1, (snapshot_width, snapshot_height), interpolation=cv2.INTER_LINEAR)

                # Place the actual snapshot in the designated region
                output[zoom_y_start:zoom_y_start + snapshot_height, zoom_x_start:zoom_x_start + snapshot_width] = zoomed1

                # Add text inside the zoomed image
                cv2.putText(output, f"Target: {id}", (zoom_x_start + 10, zoom_y_start + (snapshot_height- 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                cv2.putText(output, f"W:{str(w)} H:{str(h)} Area:{area}", (zoom_x_start + 10, zoom_y_start + (snapshot_height- 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

                zoom_x_start += snapshot_width + 20
                
                padding = 20  # Extra pixels to include more context
                
                x1, y1 = max(0, x - padding), max(0, y - padding)
                x2, y2 = min(width, x + w + padding), min(height, y + h + padding)

                zoomed = frame[y1:y2, x1:x2]
                zoomed = cv2.resize(zoomed, (snapshot_width, snapshot_height), interpolation=cv2.INTER_LINEAR)
                output[zoom_y_start:zoom_y_start + snapshot_height, zoom_x_start:zoom_x_start + snapshot_width] = zoomed
                
                # Add text inside the zoomed image
                cv2.putText(output, f"Target: {id}", (zoom_x_start + 10, zoom_y_start + (snapshot_height- 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                cv2.putText(output, f"W:{str(w)} H:{str(h)} Area:{area}", (zoom_x_start + 10, zoom_y_start + (snapshot_height- 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                
                # Move x-coordinate for the next snapshot
                zoom_x_start += snapshot_width + 20  

            # Track elapsed time
            end = time.perf_counter()
            elapsed_time = end - start

            # Maintain a rolling average of frame times
            frame_times.append(elapsed_time)
            if len(frame_times) > 10:
                frame_times.pop(0)

            # Ensure we have enough frames to calculate FPS
            if len(frame_times) > 1:
                avg_time_per_frame = sum(frame_times) / len(frame_times)
                fps_ = 1.0 / avg_time_per_frame
            else:
                fps_ = 0  # Avoid division by zero if no frames are processed
            
            # Overlay navigation instructions on the frame
            
            filename = os.path.basename(video_list[video_index])
            total_videos = len(video_list)
            index_text = f"{video_index + 1}/{total_videos} - {filename}"
            overlay_texts = [
                f"Video: {index_text}",
                f"Resolution: {width}x{height}, FPS: {fps_:.2f}/{fps}",
                f"Frame: {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}/{total_frames}",
                "Controls:",
                "ESC - Exit  | SPACE - Pause/Resume",
                "R - Restart | S - Save Frame | #0-9 Annotate Broken 640x640",
                "D - Next Frame | A - Previous Frame (while paused)",
                "N - Next Video | P - Previous Video (while paused)",
                "L - ROI Landscape 0.5 | I - RIO Landscape 0.8 | F - RIO Full | O - ROI Portrait 0.5 | K - ROI Portrait 0.4 (while paused)"
            ]
            
            y_offset = height - (len(overlay_texts) * 30) - 10  # Positioning at the bottom

            for text in overlay_texts:
                cv2.putText(output, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30
                
            # Resize the frame
            resized_frame = cv2.resize(output, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            
            # Display the resized frame
            cv2.imshow("Frame", resized_frame)
        
        # Handle key events
        key = cv2.waitKeyEx(0 if paused else 1)

        if key == 27:  # ESC key to exit
            break
        elif key == 32:  # Spacebar to pause/resume
            paused = not paused
            frame_index = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if not paused: 
                frame_times = []  # Reset frame times when pausing/resuming
        elif key == ord("r"): 
            frame_index = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index) # Restart video
            tracker.reset()  # Clear trajectories and objects
            object_detector = cv2.createBackgroundSubtractorMOG2(history=60, varThreshold=100, detectShadows=False)
            paused = False
        elif key == ord("t"): 
            tracker.reset()  # Clear trajectories and objects
        elif paused and key == ord("d"): 
            frame_index = min(frame_index + 1, total_frames - 1)
            continue
        elif paused and key == ord("a"): 
            frame_index = max(frame_index - 1, 0)
        elif paused and key == ord('n'):  # Next video
            video_index = (video_index + 1) % len(video_list)  # Loop around if at last video
            if load_video(video_index, video_list):
                cap.release()
                cv2.destroyAllWindows()
                process_video(video_index, video_list)
        elif paused and key == ord('p'):  # Previous video
            video_index = (video_index - 1) % len(video_list)  # Loop around if at first video
            if load_video(video_index, video_list):
                cap.release()
                cv2.destroyAllWindows()
                process_video(video_index, video_list)
        elif paused and key == ord("s"): # Save frame
            save_frame(frame=frame, frame_num=int(frame_index))
            
        elif paused and key == ord("l"): # ROI landscape /2 height
            region_of_interest = 0, 0, width, height // 2
            x_min, y_min, x_max, y_max = region_of_interest
            
        elif paused and key == ord("i"): # ROI landscape 0.8 height
            region_of_interest = 0, 0, width, int(height * 0.8)
            x_min, y_min, x_max, y_max = region_of_interest
            
        elif paused and key == ord("o"): # ROI portrait /2 width
            region_of_interest = 0, 0, width // 2, height
            x_min, y_min, x_max, y_max = region_of_interest
            
        elif paused and key == ord("k"): # ROI portrait 0.4 width
            region_of_interest = 0, 0, int(width * 0.4), height
            x_min, y_min, x_max, y_max = region_of_interest
            
        elif paused and key == ord("f"): # ROI full
            region_of_interest = 0, 0, width, height
            x_min, y_min, x_max, y_max = region_of_interest
            
        elif paused and key in range(48, 58): # Save frame and annotation based on the number inputted buy the user that translates to the target id. 640x640
            target_id = key - 48
            for box_id in boxes_ids:
                x, y, w, h, area, tid = box_id
                if tid == target_id:
                    save_frame_and_annotation(frame=frame, width=width, height=height, frame_num=int(frame_index), bboxes=[(x, y, w, h)])
                    break

    cap.release()
    cv2.destroyAllWindows()

def main():
    # video_list = get_video_list("videos_run_1")
    # video_list = get_video_list("../videos break")
    # video_list = get_video_list("../videos_break_1")
    # video_list = get_video_list("../videos_whole_1")
    video_list = get_video_list("../videos_whole_2")
    if not video_list:
        print("No videos found.")
        return
    
    video_index = 0
    process_video(video_index, video_list)

if __name__ == "__main__":
    main()
