# scripts/run.py

import argparse
import cv2
import time
import datetime
from pathlib import Path
from collections import deque
import logging

from traffic_counter import TrafficCounter, load_config, setup_logging, draw_semi_transparent_rectangle

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Traffic Counter")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set up logging based on config
    log_file = config.get('logging', {}).get('file', 'traffic_counter.log')
    log_level = config.get('logging', {}).get('level', 'INFO')
    setup_logging(log_file=log_file, level=log_level)

    # Initialize TrafficCounter with config
    traffic_config = config.get('model', {})
    path_config = config.get('path', {})
    frame_config = config.get('frame_processing', {})
    class_mapping = config.get('classes', {}).get('mapping', {
        0: "person",
        2: "car",
        7: "truck"
    })
    output_config = config.get('output', {})

    counter = TrafficCounter(
        model_path=traffic_config.get('path', 'yolov8n.pt'),
        detection_threshold=traffic_config.get('detection_threshold', 0.45),
        tracking_threshold=traffic_config.get('tracking_threshold', 0.8),
        tracker=traffic_config.get('tracker', 'botsort.yaml'),
        max_path_length=path_config.get('max_length', 30),
        min_points_for_path=path_config.get('min_points_for_crossing', 3),
        frame_skip=frame_config.get('frame_skip', 3),
        roi_padding=frame_config.get('roi_padding', 200),
        class_mapping=class_mapping
    )

    # Open video with optimized buffer size
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        logging.error(f"Error opening video file: {args.video_path}")
        return

    # Set OpenCV buffer size
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

    # Read first frame for line drawing
    ret, frame = cap.read()
    if not ret:
        logging.error("Error reading video frame")
        return

    # Let user draw counting line
    counter.draw_line(frame)
    if not counter.count_line:
        logging.error("No counting line drawn")
        return

    try:
        # Create video writer with optimized settings
        output_video_path = output_config.get('video_path', f"{Path(args.video_path).stem}_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter(
            output_video_path,
            fourcc,
            fps,
            (width, height),
            isColor=True
        )

        # Performance monitoring variables
        frame_times = deque(maxlen=20)  # Keep last 20 frame times for averaging
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed_frames = 0
        last_update_time = time.time()
        update_interval = 1.0  # Status update interval in seconds

        status_text = ""  # Initialize outside the loop
        while cap.isOpened():
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            # Update FPS and progress every second
            current_time = time.time()
            if current_time - last_update_time >= update_interval and len(frame_times) == frame_times.maxlen:
                avg_fps = len(frame_times) / sum(frame_times)
                progress = (processed_frames / total_frames) * 100
                if counter.frame_count % counter.frame_skip == 0:
                    mode_text = "Processing"
                else:
                    mode_text = "Skipping"
                status_text = f"FPS: {avg_fps:.1f} | Progress: {progress:.1f}% | {mode_text}"
                last_update_time = current_time

            # Process frame and get the processed frame
            processed_frame = counter.process_frame(frame, status_text=status_text)
            out.write(processed_frame)

            # Performance monitoring
            frame_time = time.time() - start_time
            frame_times.append(frame_time)
            processed_frames += 1

            # Display frame
            cv2.imshow("Traffic Counter", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Limit display rate to ~30 FPS if desired
            elapsed = time.time() - start_time
            if elapsed < 1.0 / 30.0:
                time.sleep(1.0 / 30.0 - elapsed)

    except Exception as e:
        logging.error(f"Error processing video: {str(e)}")
        raise
    finally:
        # Clean up
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Save results
        csv_path = output_config.get('csv_path', 'crossings.csv')
        if output_config.get('save_csv', True):
            counter.save_results(csv_path)
            logging.info(f"Crossing events saved to {csv_path}")

        if output_config.get('save_video', True):
            logging.info(f"Processed video saved to {output_video_path}")

if __name__ == "__main__":
    main()
