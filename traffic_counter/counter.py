# traffic_counter/counter.py

import numpy as np
import pandas as pd
from ultralytics import YOLO
import cv2
import torch
import time
from collections import deque
from typing import List, Tuple, Dict, Optional
import datetime
from dataclasses import dataclass
import logging
from pathlib import Path

from .config import setup_logging

# Initialize Logging
# The log file and level will be set based on the configuration
logger = logging.getLogger(__name__)

@dataclass
class CountLine:
    """Represents a counting line with start and end points"""
    start: Tuple[int, int]
    end: Tuple[int, int]
    original_size: Tuple[int, int]
    
    def get_line_equation(self) -> Tuple[float, float, float]:
        """Returns the line equation coefficients (a, b, c) where ax + by + c = 0"""
        x1, y1 = self.start
        x2, y2 = self.end
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        return a, b, c
    
    def get_scaled_points(self, current_size: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Scale the line points based on current frame size"""
        if not self.original_size:
            return self.start, self.end
        
        orig_w, orig_h = self.original_size
        cur_w, cur_h = current_size
        scale_x = cur_w / orig_w
        scale_y = cur_h / orig_h
        
        scaled_start = (int(self.start[0] * scale_x), int(self.start[1] * scale_y))
        scaled_end = (int(self.end[0] * scale_x), int(self.end[1] * scale_y))
        return scaled_start, scaled_end

class Direction:
    INBOUND = "inbound"
    OUTBOUND = "outbound"

@dataclass
class CrossingEvent:
    """Represents a crossing event"""
    relative_time: float  # Time in seconds relative to video start
    object_class: str
    direction: str
    count: int

class TrafficCounter:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        detection_threshold: float = 0.45,
        tracking_threshold: float = 0.8,
        tracker: str = "botsort.yaml",
        max_path_length: int = 30,
        min_points_for_path: int = 3,
        frame_skip: int = 3,
        roi_padding: int = 200,
        class_mapping: Dict[int, str] = None,
        fps: float = 30.0,  # Add fps parameter
        start_time: str = "00:00"  # Add start_time parameter
    ):
        """
        Initialize the TrafficCounter with configuration parameters.
        
        Args:
            model_path (str): Path to the YOLO model weights file.
            detection_threshold (float): Confidence threshold for detections.
            tracking_threshold (float): IoU threshold for tracking.
            tracker (str): Tracker configuration file.
            max_path_length (int): Maximum number of points in the tracking path.
            min_points_for_path (int): Minimum points required to validate a path crossing.
            frame_skip (int): Process every Nth frame.
            roi_padding (int): Padding around the counting line in pixels.
            class_mapping (Dict[int, str], optional): Mapping from class IDs to class names.
        """
        # Initialize logging inside the TrafficCounter if needed
        # Assuming logging is already set up externally
        
        # Determine device
        if torch.cuda.is_available():
            self.device = 'cuda'
            logger.info("Using CUDA for inference.")
        elif torch.backends.mps.is_available():
            self.device = 'mps'
            logger.info("Using MPS (Metal) for inference.")
        else:
            self.device = 'cpu'
            logger.info("Using CPU for inference.")

        # Load YOLO model
        self.model = YOLO(model_path).to(self.device)
        # Enable half precision if using CUDA
        if self.device == 'cuda':
            # self.model.model.half()  # Removed to prevent dtype mismatch
            logger.info("Using CUDA for inference without manual half precision.")

        self.model.overrides['batch'] = 1
        self.count_line: Optional[CountLine] = None
        self.crossings: List[CrossingEvent] = []
        self.detection_threshold = detection_threshold
        self.tracking_threshold = tracking_threshold
        self.tracker = tracker
        self.previous_centers: Dict[int, Tuple[float, float]] = {}
        self.paths: Dict[int, List[Tuple[float, float]]] = {}
        self.max_path_length = max_path_length
        self.min_points_for_path = min_points_for_path
        self.frame_skip = frame_skip        # Store the parameter
        self.roi_padding = roi_padding      # Store the parameter
        self.counted_tracks = set()
        self.fps = fps
        self.start_time_seconds = self._parse_start_time(start_time)

        # Enhanced tracking settings for YOLO
        self.model.overrides['conf'] = detection_threshold  # Detection confidence threshold
        self.model.overrides['iou'] = tracking_threshold    # NMS IoU threshold
        self.model.overrides['verbose'] = False            # Reduce console output
        self.model.overrides['retina_masks'] = True        # Better mask prediction
        self.model.overrides['agnostic_nms'] = True        # Class-agnostic NMS

        # Initialize counts
        self.counts = {}
        for obj_class in class_mapping.values():
            self.counts[obj_class] = {Direction.INBOUND: 0, Direction.OUTBOUND: 0}

        self.class_mapping = class_mapping if class_mapping else {
            0: "person",
            2: "car",
            7: "truck"
        }

        self.frame_count = 0  # Initialize frame_count here

    def _is_valid_crossing(
        self,
        previous: Tuple[float, float],
        current: Tuple[float, float],
        min_distance: float = 10.0
    ) -> bool:
        """Validate if a crossing detection is legitimate"""
        dx = current[0] - previous[0]
        dy = current[1] - previous[1]
        distance = np.sqrt(dx*dx + dy*dy)

        if distance < min_distance:
            return False

        if not self.count_line:
            return False

        line_dx = self.count_line.end[0] - self.count_line.start[0]
        line_dy = self.count_line.end[1] - self.count_line.start[1]
        line_length = np.sqrt(line_dx*line_dx + line_dy*line_dy)

        dot_product = (dx*line_dx + dy*line_dy) / (distance * line_length)
        angle = np.abs(np.arccos(dot_product))

        return 0.785 <= angle <= 2.356  # 45 to 135 degrees in radians

    def _update_path(self, track_id: int, center: Tuple[float, float]) -> None:
        """Update the path for a given track"""
        if track_id not in self.paths:
            self.paths[track_id] = []

        self.paths[track_id].append(center)
        if len(self.paths[track_id]) > self.max_path_length:
            self.paths[track_id].pop(0)

    def _check_path_crossing(self, path: List[Tuple[float, float]]) -> Tuple[bool, str]:
        """
        Check if a path crosses the counting line by analyzing multiple points
        Returns: (crossed, direction)
        """
        if not self.count_line or len(path) < self.min_points_for_path:
            return False, ""

        a, b, c = self.count_line.get_line_equation()

        # Calculate signs for all points in the path
        signs = [np.sign(a * x + b * y + c) for x, y in path]

        # Look for sign changes in consecutive points
        for i in range(len(signs) - 1):
            if signs[i] != signs[i + 1]:
                # Get the points where crossing occurred
                p1, p2 = path[i], path[i + 1]

                # Calculate movement direction
                dx = self.count_line.end[0] - self.count_line.start[0]
                dy = self.count_line.end[1] - self.count_line.start[1]

                movement_dx = p2[0] - p1[0]
                movement_dy = p2[1] - p1[1]

                cross_product = dx * movement_dy - dy * movement_dx

                return True, Direction.INBOUND if cross_product > 0 else Direction.OUTBOUND

        return False, ""

    def _determine_direction(
        self,
        track_id: int,
        current_center: Tuple[float, float]
    ) -> str:
        """Determine direction of movement using path analysis"""
        self._update_path(track_id, current_center)
        crossed, direction = self._check_path_crossing(self.paths[track_id])
        return direction if crossed else ""

    def _parse_start_time(self, start_time: str) -> float:
        """Parse start time string into seconds."""
        try:
            hours, minutes = map(int, start_time.split(":"))
            return hours * 3600 + minutes * 60
        except ValueError:
            logger.warning(f"Invalid start_time format '{start_time}'. Defaulting to 0 seconds.")
            return 0.0
        
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for better detection quality while maintaining brightness
        """
        # Maintain aspect ratio while resizing
        max_dimension = 1280  # Increased for better detection
        height, width = frame.shape[:2]
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Enhanced preprocessing pipeline
        # 1. Denoise
        #frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

        # 2. Adjust contrast and brightness
        alpha = 1.1  # Contrast control (1.0-3.0)
        beta = 5    # Brightness control (0-100)
        #frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

        # 3. Sharpen
        #kernel = np.array([[-1,-1,-1], 
        #                 [-1, 9,-1],
        #                 [-1,-1,-1]])
        #frame = cv2.filter2D(frame, -1, kernel)

        return frame

    def process_frame(self, frame: np.ndarray, status_text: str = "") -> np.ndarray:
        """
        Process a single frame with YOLO tracking, count line crossings once per track,
        and restrict detection to a region of interest (ROI) around the line.

        Args:
            frame: The current video frame (BGR)
            status_text: Optional text to display (e.g., FPS, progress)

        Returns:
            Processed frame (with counts, line, bounding boxes, paths, and status text drawn)
        """
        self.frame_count += 1

        if not self.count_line:
            raise ValueError("Count line not set")

        # Preprocess frame
        frame = self._preprocess_frame(frame)
        frame_h, frame_w = frame.shape[:2]

        # Compute scaled line points for current frame size
        line_start, line_end = self.count_line.get_scaled_points((frame_w, frame_h))

        # Compute ROI around the line
        # Determine ROI boundaries
        x_min = max(0, min(line_start[0], line_end[0]) - self.roi_padding)
        x_max = min(frame_w, max(line_start[0], line_end[0]) + self.roi_padding)
        y_min = max(0, min(line_start[1], line_end[1]) - self.roi_padding)
        y_max = min(frame_h, max(line_start[1], line_end[1]) + self.roi_padding)

        # Draw ROI for visualization (semi-transparent rectangle)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 255, 255), -1)  # Cyan color
        alpha = 0.2  # Transparency factor
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Crop ROI
        roi_frame = frame[y_min:y_max, x_min:x_max]

        # **Remove Manual Half Precision Conversion**
        # if self.device == 'cuda':
        #     roi_frame = roi_frame.astype(np.float16)

        # Run YOLO tracking on ROI only
        results = self.model.track(
            roi_frame,
            conf=self.detection_threshold,
            iou=self.tracking_threshold,
            persist=True,
            tracker=self.tracker
        )

        # Note: results is a list of Results objects; take the first one
        if isinstance(results, list) and len(results) > 0:
            results = results[0]
        else:
            results = results  # Handle if not list

        # Draw counting line on full frame
        cv2.line(frame, line_start, line_end, (0, 255, 0), 2)

        # Process detections
        if results.boxes and results.boxes.id is not None:
            # Convert tensors to numpy arrays
            boxes = results.boxes.xywh.cpu().numpy()  # [x_center, y_center, width, height]
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            classes = results.boxes.cls.cpu().numpy().astype(int)
            confidences = results.boxes.conf.cpu().numpy()

            # Process valid detections
            for box, track_id, class_id, conf in zip(boxes, track_ids, classes, confidences):
                if class_id not in self.class_mapping:
                    continue

                # XYWH format in ROI coordinates: convert to full frame coordinates
                x_center, y_center, w, h = box
                x_center_full = x_center + x_min
                y_center_full = y_center + y_min

                current_center = (x_center_full, y_center_full)

                # Determine direction of movement via path analysis
                direction = self._determine_direction(track_id, current_center)

                # Draw path if we have one
                if track_id in self.paths and len(self.paths[track_id]) > 1:
                    path_points = np.array(self.paths[track_id], dtype=np.int32)
                    cv2.polylines(frame, [path_points.reshape((-1, 1, 2))],
                                  False, (255, 255, 0), 2)

                    # If direction is detected and this track hasn't been counted yet, count it
                    if direction and track_id not in self.counted_tracks:
                        object_class = self.class_mapping[class_id]
                        self.counts[object_class][direction] += 1
                        
                        # Calculate relative time
                        relative_time = (self.frame_count / self.fps) + self.start_time_seconds
                        # Round to two decimal places
                        relative_time = round(relative_time, 2)
    
                        # Append to crossings with relative time
                        event = CrossingEvent(
                            relative_time=relative_time,
                            object_class=object_class,
                            direction=direction,
                            count=self.counts[object_class][direction]
                        )
                        
                        self.crossings.append(event)

                        # Mark this track as counted
                        self.counted_tracks.add(track_id)

                # Update tracking with current center
                self.previous_centers[track_id] = current_center

                # Convert xywh (center) to bounding box in full frame coordinates
                x1 = int(x_center_full - w / 2)
                y1 = int(y_center_full - h / 2)
                x2 = int(x_center_full + w / 2)
                y2 = int(y_center_full + h / 2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw label
                label = f"{self.class_mapping[class_id]} {conf:.2f}"
                # Black outline
                cv2.putText(frame, label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 0), 4)
                # Green text
                cv2.putText(frame, label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 1)

        # Draw counts
        counts_text = []
        for obj_class, directions in self.counts.items():
            counts_text.append(f"{obj_class}: IN={directions[Direction.INBOUND]} OUT={directions[Direction.OUTBOUND]}")

        for i, text in enumerate(counts_text):
            # Black outline
            cv2.putText(frame, text, (10, 30 + i*30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 4)
            # Green text
            cv2.putText(frame, text, (10, 30 + i*30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)

        # Always draw status text (FPS, progress, etc.)
        if status_text:
            # Black outline
            cv2.putText(frame, status_text,
                        (10, frame_h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 4)
            # Green text
            cv2.putText(frame, status_text,
                        (10, frame_h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)

        return frame

    def draw_line(self, frame: np.ndarray) -> None:
        """Allow user to draw counting line on frame"""
        points = []
        original_size = (frame.shape[1], frame.shape[0])

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                if len(points) == 2:
                    self.count_line = CountLine(
                        start=points[0],
                        end=points[1],
                        original_size=original_size
                    )
                    cv2.destroyWindow("Draw Line")

        cv2.namedWindow("Draw Line")
        cv2.setMouseCallback("Draw Line", mouse_callback)

        while True:
            display_frame = frame.copy()
            if len(points) == 1:
                cv2.circle(display_frame, points[0], 5, (0, 255, 0), -1)

            cv2.imshow("Draw Line", display_frame)
            key = cv2.waitKey(1)

            if key == 27 or self.count_line:  # ESC key or line drawn
                break

        cv2.destroyAllWindows()

    def save_results(self, output_path: str) -> None:
        """Save crossing events to CSV"""
        if not self.crossings:
            logger.warning("No crossing events to save")
            return
        
        # Prepare data for DataFrame
        data = []
        for event in self.crossings:
            # Convert relative_time (seconds) to "HH:MM:SS.SS" format
            hours = int(event.relative_time // 3600)
            minutes = int((event.relative_time % 3600) // 60)
            seconds = event.relative_time % 60
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"
    
            data.append({
                "date time": time_str,
                "category": event.object_class,
                "direction": event.direction,
                "count": event.count
            })

        df = pd.DataFrame([vars(event) for event in self.crossings])
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
