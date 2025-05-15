import cv2
import os
from datetime import datetime, timedelta

def parse_timestamps(timestamp_lines):
    times = [datetime.strptime(t.strip(), "%M:%S") for t in timestamp_lines]
    times.sort()

    # Group consecutive seconds into ranges
    clip_ranges = []
    start_time = times[0]
    end_time = times[0]

    for i in range(1, len(times)):
        if (times[i] - end_time) <= timedelta(seconds=1):
            end_time = times[i]
        else:
            clip_ranges.append((start_time, end_time))
            start_time = end_time = times[i]

    clip_ranges.append((start_time, end_time))
    return clip_ranges

def extract_clip(video_path, output_path, start_time, end_time, fps):
    cap = cv2.VideoCapture(video_path)
    start_sec = start_time.minute * 60 + start_time.second
    end_sec = end_time.minute * 60 + end_time.second
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for f in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    print(f"Saved clip: {output_path}")

def create_clips_from_detections(video_folder, detection_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for video_file in os.listdir(video_folder):
        if not video_file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            continue

        video_name = os.path.splitext(video_file)[0]
        detection_file = os.path.join(detection_folder, f"{video_name}.txt")

        if not os.path.exists(detection_file):
            continue

        with open(detection_file, "r") as f:
            timestamp_lines = f.readlines()

        if not timestamp_lines:
            continue

        clip_ranges = parse_timestamps(timestamp_lines)

        video_path = os.path.join(video_folder, video_file)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        for idx, (start, end) in enumerate(clip_ranges):
            output_path = os.path.join(output_folder, f"{video_name}_clip_{idx+1:02d}.mp4")
            extract_clip(video_path, output_path, start, end, fps)

if __name__ == "__main__":
    video_folder = "../videos"
    detection_folder = "../detections"
    output_folder = "../clips"
    create_clips_from_detections(video_folder, detection_folder, output_folder)
