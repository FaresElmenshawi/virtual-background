# Virtual Background Application using YOLO and OpenCV

This Python application utilizes the YOLO (You Only Look Once) model and OpenCV to segment objects in real-time video streams and apply virtual backgrounds. The script processes video input, segments objects using a pre-trained YOLO model, and seamlessly integrates a virtual background of choice behind the segmented object.

## Features

- Real-time object segmentation using YOLO
- Dynamic virtual background application
- Support for video input from files or camera devices
- Easy-to-use command-line interface

## Prerequisites

Before running this application, ensure you have the following installed:

- Python 3.6 or higher
- OpenCV-Python
- NumPy
- The ultralytics YOLO library

## Setup

1. Clone this repository to your local machine.

    ```bash
    git clone <repository-url>
    ```

2. Navigate into the cloned directory.

    ```bash
    cd <cloned-directory>
    ```

3. Install the required Python dependencies.

    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the application, use the following command:

```bash
python <script-name>.py --source <source>
```

`<script-name>.py`: Name of the main Python script.

`<source>`: (Optional) Index of the video capture device (e.g., `0` for the default webcam) or the path to a video file. Defaults to `0` if not specified.

## Command Line Arguments

- `--source`: Specifies the video source. It can be the device index of the camera (e.g., `0`) or the path to a video file.

## How It Works

The application performs the following steps:

1. Captures video from the specified source.
2. Uses the YOLO model to segment objects in each frame.
3. Applies a virtual background behind the segmented objects.
4. Displays the processed video stream in real-time.

You can switch between different virtual backgrounds by pressing the 'n' key during execution. Press 'q' to quit the application.

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for more details.

## Contributing

Contributions are welcome! Please feel free to submit pull requests, report bugs, or suggest features.
