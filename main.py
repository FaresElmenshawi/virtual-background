from ultralytics import YOLO
import numpy as np
import cv2
import os
import argparse

# Initialize YOLO model with a pre-trained segmentation model
model = YOLO('yolov8x-seg.pt')

# Load the virtual background image
background = cv2.imread("backgrounds/background0.jpg")


def apply_virtual_background(frame, mask, background):
    """
    Apply a virtual background to the input frame based on the provided mask.

    Args:
        frame (numpy.ndarray): Input frame (image).
        mask (numpy.ndarray): Mask defining the region to keep from the frame.
        background (numpy.ndarray): Virtual background image.

    Returns:
        numpy.ndarray: Resulting frame with the virtual background applied.
    """
    img_roi = cv2.bitwise_and(frame, mask)
    background = cv2.resize(background, (frame.shape[1], frame.shape[0]))
    inverted_mask = cv2.bitwise_not(mask)
    background_roi = cv2.bitwise_and(background, inverted_mask)
    result_frame = cv2.add(background_roi, img_roi)
    return result_frame


def create_mask(frame, results):
    """
    Create a mask based on segmentation results.

    Args:
        frame (numpy.ndarray): Input frame (image).
        results (YOLOResults): YOLO segmentation results.

    Returns:
        numpy.ndarray: Binary mask representing the segmented object.
    """
    mask = np.zeros_like(frame, dtype=np.uint8)
    if results.masks is not None:
        mask_coords = results.masks.xy[0].astype(int)
        mask_coords = mask_coords.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [mask_coords], (255, 255, 255))
    return mask


def post_process_mask(mask):
    # Convert the mask to grayscale
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Apply dilation to fill in gaps in the mask
    kernel_dilate = np.ones((5, 5), np.uint8)
    mask_dilated = cv2.dilate(mask_gray, kernel_dilate, iterations=1)

    # Apply erosion to refine the mask
    kernel_erode = np.ones((5, 5), np.uint8)
    mask_eroded = cv2.erode(mask_dilated, kernel_erode, iterations=1)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a new mask with filled contours
    refined_mask = np.zeros_like(mask)
    cv2.drawContours(refined_mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    return refined_mask


# Open a video capture from the default camera (camera index 0)
def process_video(source=0):
    """
    Captures video from a specified source, performs object segmentation, refines the object mask,
    and applies a virtual background based on the refined mask.

    Parameters:
    - source (int or str): The index of the video capture device or the path to a video file.
    """
    global background
    current_background_index = 0
    cap = cv2.VideoCapture(source)
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        if not ret:
            break

        # Perform segmentation using the YOLO model
        results = model(frame, verbose=False, conf=0.4, classes=0)
        results = results[0]

        # Create a mask for the segmented object
        mask = create_mask(frame, results)

        # Refine mask
        # mask = post_process_mask(mask)

        # Apply the virtual background to the frame
        frame = apply_virtual_background(frame, mask, background)

        cv2.imshow("Segmentation Results", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            # Calculate the next background index with a loop back to 0 if it's the last background
            background_path = os.path.join("backgrounds",
                                           f"background{(current_background_index + 1) % len(os.listdir('backgrounds'))}.jpg")
            new_background = cv2.imread(background_path)
            # Check if the new background image was successfully loaded
            if new_background is not None:
                background = new_background
                current_background_index = (current_background_index + 1) % len(os.listdir("backgrounds"))
            else:
                print(f"Failed to load: {background_path}")

    # Release the video capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


def main():
    """
    Main function to parse command line arguments and start the video processing.
    """
    parser = argparse.ArgumentParser(description="Process video for object segmentation and background application.")
    parser.add_argument("--source", type=str, default=0, help="Index of the video capture device. Default is 0.")
    args = parser.parse_args()
    source = args.source
    if source.isdigit():
        source = int(source)
    process_video(source=source)


if __name__ == "__main__":
    main()
