# Traffic Counter

A traffic counting application using YOLO and OpenCV to detect and track objects crossing a defined line in a video.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Clone the Repository](#clone-the-repository)
  - [Set Up a Virtual Environment](#set-up-a-virtual-environment)
  - [Install Dependencies](#install-dependencies)
  - [Install the Package](#install-the-package)
- [Usage](#usage)
  - [Configuration](#configuration)
  - [Run the Application](#run-the-application)
  - [Run with Docker](#run-with-docker)
- [Testing](#testing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Features

- **Object Detection and Tracking:** Utilizes YOLOv8 and BoT-SORT for robust object detection and tracking.
- **Configurable Region of Interest (ROI):** Focuses detection within a specified area around the counting line to optimize performance.
- **Counting Mechanism:** Accurately counts inbound and outbound traffic for defined object classes.
- **Performance Monitoring:** Displays real-time FPS and processing progress.
- **Result Logging:** Saves crossing events to a CSV file and optionally annotates and saves the output video.
- **Flexible Configuration:** Easily adjust settings via a YAML configuration file.

---

## Installation

Follow these steps to set up the `TrafficCounter` application on your local machine.

### Prerequisites

Before you begin, ensure you have met the following requirements:

- **Operating System:** macOS, Linux, or Windows
- **Python Version:** Python 3.7 or higher
- **CUDA-enabled GPU (Optional):** For faster processing using GPU acceleration
- **Docker (Optional):** For containerized deployment

### Clone the Repository

Start by cloning the repository to your local machine using Git:

```bash
git clone https://github.com/finnmo/traffic-counter.git
cd traffic-counter
```

### Set Up a Virtual Environment

It's recommended to use a virtual environment to manage dependencies and avoid conflicts with other Python projects.

1. **Create a Virtual Environment:**

   ```bash
   python3 -m venv env
   ```

   This command creates a virtual environment named `env` in your project directory.

2. **Activate the Virtual Environment:**

   - **On macOS and Linux:**

     ```bash
     source env/bin/activate
     ```

   - **On Windows:**

     ```bash
     env\Scripts\activate
     ```

   After activation, your terminal prompt will be prefixed with `(env)` indicating that the virtual environment is active.

### Install Dependencies

Install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

> **Note:** Ensure that `requirements.txt` contains all necessary dependencies with the correct versions.

### Install the Package

Install the `TrafficCounter` package in editable mode. This allows you to make changes to the source code without reinstalling the package.

```bash
pip install -e .
```

> **Explanation:**
>
> - The `-e` flag stands for "editable," meaning any changes made to the source code will be immediately reflected without needing to reinstall.
> - The `.` indicates that the `setup.py` file is in the current directory.

---

## Usage

After successfully installing the package and its dependencies, you're ready to use the `TrafficCounter` application.

### Configuration

Before running the application, configure its settings via the `config.yaml` file.

1. **Locate the `config.yaml` File:**

   The `config.yaml` file should be in the root directory of the project. If it's not present, create one based on the provided sample.

2. **Customize Configuration Parameters:**

   Open `config.yaml` and adjust the parameters as needed. Below is a sample configuration:

   ```yaml
   # config.yaml

   # -----------------------------
   # Model Configuration
   # -----------------------------
   model:
     path: "yolov11n.pt"                 # Path to the YOLO model weights file
     detection_threshold: 0.45          # Confidence threshold for object detections (0.0 - 1.0)
     tracking_threshold: 0.8            # IoU threshold for tracking (0.0 - 1.0)
     tracker: "botsort.yaml"            # Tracker configuration file ("bytetrack.yaml" or "botsort.yaml")

   # -----------------------------
   # Path and Crossing Configuration
   # -----------------------------
   path:
     max_length: 30                     # Maximum number of points to keep in each object's tracking path
     min_points_for_crossing: 3         # Minimum number of points required to validate a path for crossing detection

   # -----------------------------
   # Frame Processing Configuration
   # -----------------------------
   frame_processing:
     frame_skip: 3                      # Process every Nth frame to improve processing speed
     roi_padding: 200                   # Padding in pixels around the counting line to define the Region of Interest (ROI)

   # -----------------------------
   # Class Mapping Configuration
   # -----------------------------
   classes:
     mapping:
       0: "person"                       # YOLO class ID 0 mapped to "person"
       2: "car"                          # YOLO class ID 2 mapped to "car"
       7: "truck"                        # YOLO class ID 7 mapped to "truck"

   # -----------------------------
   # Output Configuration
   # -----------------------------
   output:
     save_csv: true                     # Whether to save the crossing events to a CSV file
     csv_path: "crossings.csv"          # Path to save the CSV file with crossing events
     save_video: true                   # Whether to save the output video with annotations
     video_path: "output.mp4"           # Path to save the annotated output video

   # -----------------------------
   # Logging Configuration
   # -----------------------------
   logging:
     level: "INFO"                       # Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
     file: "traffic_counter.log"         # Path to save the log file
   ```

   > **Tips:**
   >
   > - **Model Path:** Ensure that the YOLO model weights file (`yolov8n.pt`) is in the specified path or provide the correct path to your model.
   > - **Video Paths:** The `video_path` and `csv_path` should be writable locations where you want the output to be saved.
   > - **Logging Level:** Adjust the logging level based on your debugging needs (`DEBUG` for more verbose logs).

### Run the Application

You can run the `TrafficCounter` application using either the console script or directly via Python.

#### Option 1: Using the Console Script

After installation, you can use the `traffic-counter` command directly.

```bash
traffic-counter ./media/0001.avi --config config.yaml
```

> **Explanation:**
>
> - `./media/0001.avi`: Path to the input video file.
> - `--config config.yaml`: Specifies the configuration file to use.

#### Option 2: Using Python Directly

Alternatively, you can run the application by executing the `run.py` script with Python.

```bash
python traffic_counter/scripts/run.py ./media/0001.avi --config config.yaml
```

> **Note:** Ensure you're in the project root directory when running this command.

### Interactive Line Drawing

Upon running the application, a window titled "Draw Line" will appear:

1. **Define the Counting Line:**
   - Click two points on the first frame of the video to define the counting line.
   - This line is used to detect objects crossing it.

2. **Confirm the Line:**
   - Press `ESC` to finalize the line drawing.
   - Alternatively, wait for the line to be drawn automatically if your implementation supports it.

### Output and Logs

- **Annotated Video:** Saved at the path specified in `config.yaml` (e.g., `output.mp4`).
- **CSV File:** Contains crossing events saved at the path specified in `config.yaml` (e.g., `crossings.csv`).
- **Log File:** Detailed logs saved at the path specified in `config.yaml` (e.g., `traffic_counter.log`).

---

## Run with Docker

For containerized deployment, you can build and run the application using Docker. This is especially useful for ensuring consistency across different environments.

### Build Docker Image

Navigate to the project root directory and build the Docker image:

```bash
docker build -t traffic-counter:1.0 .
```

> **Explanation:**
>
> - `-t traffic-counter:1.0`: Tags the image with the name `traffic-counter` and version `1.0`.
> - `.`: Specifies the build context as the current directory.

### Run Docker Container

Run the Docker container with GPU support (if available) and mount the media directory:

```bash
docker run --gpus all -v /path/to/media:/app/media traffic-counter:1.0 ./media/0001.avi --config config.yaml
```

> **Explanation:**
>
> - `--gpus all`: Grants the container access to all available GPUs. Omit or adjust if not using GPU.
> - `-v /path/to/media:/app/media`: Mounts your local `media` directory to the container's `/app/media` directory.
> - `traffic-counter:1.0`: Specifies the Docker image to use.
> - `./media/0001.avi --config config.yaml`: Arguments passed to the application inside the container.

> **Note:** Replace `/path/to/media` with the actual path to your media files on the host machine.

---

## Testing

Ensure that all components of your application work as expected by running unit tests.

### Run Unit Tests Using `pytest`

Execute the following command from the project root directory:

```bash
pytest
```

> **Explanation:**
>
> - `pytest`: A testing framework that discovers and runs tests in your project.

> **Note:** Ensure that your `tests/` directory contains test modules named `test_*.py` or `*_test.py`.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/yolov5)
- [BoT-SORT](https://github.com/zzh8829/yolov3-spp/tree/master)
- [OpenCV](https://opencv.org/)
- [PyTorch](https://pytorch.org/)
- [Docker](https://www.docker.com/)

---

## **Final Checklist**

Before deploying your application to production, ensure that:

1. **All Dependencies are Listed:**
   - Verify that `requirements.txt` includes all necessary packages with correct versions.

2. **Package is Installed Correctly:**
   - Install the package in editable mode using `pip install -e .` from the project root.

3. **Configuration is Correct:**
   - Ensure `config.yaml` is properly formatted and placed in the project root or specified correctly via the `--config` argument.

4. **Entry Points are Defined:**
   - The `entry_points` in `setup.py` should correctly point to your main script's `main` function.

5. **Testing Passes:**
   - Run `pytest` to ensure all tests pass without errors.

6. **Documentation is Up-to-Date:**
   - Your `README.md` should reflect the current state of your project, including installation and usage instructions.

7. **Docker Builds Successfully:**
   - If using Docker, ensure the `Dockerfile` builds and runs without issues, especially with GPU support.

8. **Logging is Functional:**
   - Verify that logs are being written to both the console and the specified log file.

9. **Error Handling is Robust:**
   - Test scenarios where inputs might be invalid or missing to ensure the application handles them gracefully.

10. **Performance is Optimized:**
    - Profile the application to identify and optimize any bottlenecks, ensuring it meets your performance requirements.

---