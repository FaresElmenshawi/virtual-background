# Virtual Background with YOLO Segmentation

This project applies a virtual background to a video stream or images using YOLO segmentation. It segments the foreground object in the frame and replaces the background with a virtual one.

## Requirements

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy
- Ultralytics YOLO

You can install the required packages via pip:

```bash
pip install opencv-python numpy ultralytics
```

## Usage

### Running the Script

You can run the script with the following command:

```bash
python main.py --mode <mode> --path <path_to_video_or_image_directory>
```

- `<mode>`: Mode of operation. Options are `video`, `images`, or `realtime` (default: `realtime`).
- `<path_to_video_or_image_directory>`: Path to the video file or directory containing images.

### Modes of Operation

1. **Video Mode (`video`)**:
    - Process a video file.
    - Specify the path to the video file using the `--path` argument.
    - Example: `python main.py --mode video --path path/to/video.mp4`

2. **Images Mode (`images`)**:
    - Process multiple images from a directory.
    - Specify the path to the directory containing images using the `--path` argument.
    - Example: `python main.py --mode images --path path/to/image_directory`

3. **Real-time Mode (`realtime`)**:
    - Process video stream from the default camera in real-time.
    - No need to specify the path.

### Keyboard Controls

- Press `q` to quit the application.
- Press `n` (in real-time mode) to switch to the next background image.

## Customization

You can customize the project by adding your own background images. Place your background images in the `backgrounds` directory.

## License

This project is licensed under the Apache License 2.0..

