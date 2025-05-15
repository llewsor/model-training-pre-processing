import cv2
import os
import glob

# --- CONFIGURATION ---
BASE_DIR = "dataset_break_aug"
# BASE_DIR = "dataset/break/"
# BASE_DIR = "dataset/whole/"
TRAIN_DIR = os.path.join(BASE_DIR, "train/")
VAL_DIR = os.path.join(BASE_DIR, "val/")
TEST_DIR = os.path.join(BASE_DIR, "test/")
AUG_DIR = os.path.join(BASE_DIR, "aug/")
MANUAL_DIR = "annotations/"
MANUAL_AUG_DIR = "annotations_aug/"


# Global variables
current_folder = TRAIN_DIR  # Default folder
image_files = []
current_index = 0

def load_images():
    """Load image file paths from the selected folder."""
    global image_files, current_index
    image_files = sorted(glob.glob(os.path.join(current_folder, "images", "*.jpg")))
    current_index = 0 if image_files else -1  # Reset index

def get_label_path(image_path):
    """Get the corresponding YOLO label file path for an image."""
    label_path = image_path.replace("images", "labels").replace(".jpg", ".txt")
    return label_path if os.path.exists(label_path) else None

def read_yolo_labels(label_path, img_width, img_height):
    """Read bounding boxes from a YOLO label file and convert to pixel coordinates."""
    boxes = []
    if label_path:
        with open(label_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    id, x_center, y_center, width, height = map(float, parts)
                    x1 = int((x_center - width / 2) * img_width)
                    y1 = int((y_center - height / 2) * img_height)
                    x2 = int((x_center + width / 2) * img_width)
                    y2 = int((y_center + height / 2) * img_height)
                    boxes.append((id, x1, y1, x2, y2))
    return boxes

def delete_current_image_and_label():
    """Delete the current image and its associated label file."""
    global image_files, current_index

    if current_index == -1 or not image_files:
        print("No image to delete.")
        return
      
    temp_index = current_index
    
    image_path = image_files[current_index]
    label_path = get_label_path(image_path)

    # Delete image file
    try:
        os.remove(image_path)
        print(f"Deleted image: {image_path}")
    except Exception as e:
        print(f"Error deleting image: {e}")

    # Delete label file if it exists
    if label_path and os.path.exists(label_path):
        try:
            os.remove(label_path)
            print(f"Deleted label: {label_path}")
        except Exception as e:
            print(f"Error deleting label: {e}")

    # Refresh image list and update index
    load_images()
    if current_index >= len(image_files):  # Adjust index if at the end
        current_index = max(0, len(image_files) - 1)

    current_index = temp_index
    display_image()

# Define class names and colors
CLASS_NAMES = {
    0: "Whole Clay",
    1: "Broken Clay"
}

CLASS_COLORS = {
    0: (0, 255, 0),   # Green for Whole Clay
    1: (0, 0, 255)    # Red for Broken Clay
}

def display_image():
    """Display the current image with bounding boxes."""
    if current_index == -1 or not image_files:
        print("No images found in", current_folder)
        return

    image_path = image_files[current_index]
    label_path = get_label_path(image_path)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return

    img_height, img_width = image.shape[:2]

    # Read bounding boxes
    boxes = read_yolo_labels(label_path, img_width, img_height)

    # Draw bounding boxes with class names, dimensions, and area
    for bbox in boxes:
        class_id, x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        area = w * h

        # Get class name and color
        class_name = CLASS_NAMES.get(class_id, "Unknown")
        color = CLASS_COLORS.get(class_id, (255, 255, 255))  # White if class not found

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Prepare text info
        text = f"{class_name} | W: {w}px H: {h}px Area: {area}px2"

        # Get text size
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        text_width, text_height = text_size

        # Determine if the bounding box is on the right side of the image
        if (x1 + x2) // 2 > image.shape[1] // 2:
            # If the target is on the right side, shift the text to the left
            text_x = max(5, x1 - text_width - 10)  # Ensure text stays inside the frame
        else:
            # Otherwise, place the text on the right side
            text_x = min(image.shape[1] - text_width - 5, x2 + 5)

        text_y = max(20, y1 - 10)  # Ensure text stays inside the frame vertically

        # Draw text background for readability
        cv2.rectangle(image, (text_x, text_y - text_height - 5),
                      (text_x + text_width + 5, text_y + 5), color, -1)

        # Draw text
        cv2.putText(image, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # Display filename with image index
    filename = os.path.basename(image_path)
    total_images = len(image_files)
    index_text = f"{current_index + 1}/{total_images} - {filename}"
    cv2.putText(image, index_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    # Add navigation instructions (bottom-left corner)
    instructions = [
        "T - Train | V - Val | E - Test | G - Aug | M - Man | F - M & A",
        "A - Next | D - Previous | X - Delete",
        "Q - Quit"
    ]

    y_pos = img_height - 70  # Position near bottom
    for i, text in enumerate(instructions):
        cv2.putText(image, text, (20, y_pos + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, text, (20, y_pos + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)  # Black shadow for contrast

    # Show image
    # cv2.imshow("Image Viewer", cv2.resize(image, (img_width // 2, img_height // 2))) # Resize for display
    cv2.imshow("Image Viewer", image)

def main():
    """Main function to handle user input and display images."""
    global current_folder, current_index

    load_images()
    display_image()

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):  # Quit on 'q'
            break
        elif key == ord('t'):  # Switch to training images
            current_folder = TRAIN_DIR
            load_images()
            display_image()
        elif key == ord('v'):  # Switch to validation images
            current_folder = VAL_DIR
            load_images()
            display_image()
        elif key == ord('e'):  # Switch to test images
            current_folder = TEST_DIR
            load_images()
            display_image()
        elif key == ord('g'):  # Switch to aug images
            current_folder = AUG_DIR
            load_images()
            display_image()
        elif key == ord('m'):  # Switch to manual images
            current_folder = MANUAL_DIR
            load_images()
            display_image()
        elif key == ord('f'):  # Switch to manual augmented images
            current_folder = MANUAL_AUG_DIR
            load_images()
            display_image()
        elif key == ord('x'):  # Delete image and label
          current_index = delete_current_image_and_label()
        elif key == ord('a'):  # Next image
            if image_files and current_index < len(image_files) - 1:
                current_index += 1
                display_image()
            elif image_files and current_index == len(image_files) - 1:
                current_index = 0
                display_image()
        elif key == ord('d'):  # Previous image
            if image_files and current_index > 0:
                current_index -= 1
                display_image()
            elif image_files and current_index == 0:
                current_index = len(image_files) - 1
                display_image()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
