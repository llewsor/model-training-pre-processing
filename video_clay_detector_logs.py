import cv2
import numpy as np
import os
from datetime import timedelta

# Background Subtraction for object detection
object_detector = cv2.createBackgroundSubtractorMOG2(history=60, varThreshold=100, detectShadows=False)

def detect_targets(frame):
    mask = object_detector.apply(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for c in contours:
        area = cv2.contourArea(c)
        if 100 < area < 50000:  # Filter out small and excessively large objects
            detections.append(True)
            break

    return len(detections) > 0


def process_video(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_file = os.path.join(output_folder, f"{video_name}.txt")

    detected_times = set()
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_time = frame_number / fps  # Time in seconds
        if detect_targets(frame):
            minutes, seconds = divmod(int(frame_time), 60)
            detected_times.add(f"{minutes:02d}:{seconds:02d}")

        frame_number += 1

    cap.release()

    # Save detections to file
    with open(output_file, "w") as f:
        f.write("\n".join(sorted(detected_times)))

    print(f"Processed: {video_name} - Detections saved to {output_file}")


def process_videos_in_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.MP4','.mp4'))]

    for video_file in video_files:
        video_path = os.path.join(input_folder, video_file)
        process_video(video_path, output_folder)


if __name__ == "__main__":
    input_folder = "../videos"  # Folder containing videos
    output_folder = "../detections"  # Folder where output .txt files will be saved
    process_videos_in_folder(input_folder, output_folder)
