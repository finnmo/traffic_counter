# scripts/run.py

import argparse
import cv2
import logging
import threading
import time
from pathlib import Path
from collections import deque

from traffic_counter import TrafficCounter, load_config, setup_logging


class RTSPStream:
    """
    Reads an RTSP stream in a background thread so the main loop always
    gets the latest frame rather than a buffered one. Auto-reconnects on
    stream loss.
    """

    def __init__(self, url: str):
        self.url = url
        self._lock = threading.Lock()
        self._frame = None
        self._running = True
        self._cap = self._open()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _read_loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
            else:
                logging.warning("RTSP stream lost — reconnecting in 2 s…")
                self._cap.release()
                time.sleep(2)
                self._cap = self._open()

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def get(self, prop):
        return self._cap.get(prop)

    def isOpened(self) -> bool:
        return self._cap.isOpened()

    def release(self):
        self._running = False
        self._thread.join(timeout=3)
        self._cap.release()


def resize_frame(frame, max_width: int = 1280):
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return frame


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Traffic Counter")
    parser.add_argument(
        "source",
        help='Video file path, camera index (0,1,...), or RTSP URL (rtsp://...)'
    )
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    setup_logging(
        log_file=config.get('logging', {}).get('file', 'traffic_counter.log'),
        level=config.get('logging', {}).get('level', 'INFO')
    )

    output_config = config.get('output', {})

    # Determine source type
    src_lower = args.source.lower()
    use_rtsp = src_lower.startswith(("rtsp://", "rtsps://"))

    # Open capture
    if use_rtsp:
        logging.info(f"Opening RTSP stream: {args.source}")
        cap = RTSPStream(args.source)
        # Wait up to 5 s for the first frame
        for _ in range(50):
            ret, _ = cap.read()
            if ret:
                break
            time.sleep(0.1)
        else:
            logging.error(f"Cannot read from RTSP stream: {args.source}")
            cap.release()
            return
    else:
        src = int(args.source) if args.source.isdigit() else args.source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            logging.error(f"Cannot open source: {args.source}")
            return
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

    # Get FPS for relative timestamp tracking
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    # Initialise TrafficCounter
    counter = TrafficCounter(
        model_path=config['model']['path'],
        detection_threshold=config['model']['detection_threshold'],
        tracking_threshold=config['model']['tracking_threshold'],
        tracker=config['model']['tracker'],
        max_path_length=config['path']['max_length'],
        min_points_for_path=config['path']['min_points_for_crossing'],
        frame_skip=config['frame_processing']['frame_skip'],
        roi_padding=config['frame_processing']['roi_padding'],
        class_mapping=config['classes']['mapping'],
        fps=fps,
        start_time=config['time']['start_time']
    )

    # Read first frame for line drawing
    ret, frame = cap.read()
    if not ret:
        logging.error("Could not read initial frame")
        cap.release()
        return
    frame = resize_frame(frame)

    counter.draw_line(frame)
    if not counter.count_line:
        logging.error("No counting line drawn")
        cap.release()
        return

    # Prepare video writer
    h, w = frame.shape[:2]
    if use_rtsp:
        output_path = output_config.get('video_path', 'rtsp_output.mp4')
    elif args.source.isdigit():
        output_path = output_config.get('video_path', 'webcam_output.mp4')
    else:
        output_path = output_config.get('video_path', f"{Path(args.source).stem}_output.mp4")

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h), isColor=True)

    # Processing loop
    is_live = use_rtsp or args.source.isdigit()
    total_frames = 0 if is_live else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_times = deque(maxlen=20)
    processed_frames = 0
    last_update = time.time()
    status_text = ""

    try:
        while True:
            start = time.time()

            ret, frame = cap.read()
            if not ret:
                if use_rtsp:
                    continue  # background thread is reconnecting
                break

            frame = resize_frame(frame)

            # Update status text once per second
            if len(frame_times) == frame_times.maxlen and (time.time() - last_update) >= 1.0:
                avg_fps = len(frame_times) / sum(frame_times)
                if total_frames > 0:
                    prog = (processed_frames / total_frames) * 100
                    status_text = f"FPS: {avg_fps:.1f} | Progress: {prog:.1f}%"
                else:
                    status_text = f"FPS: {avg_fps:.1f}"
                last_update = time.time()

            processed_frame = counter.process_frame(frame, status_text=status_text)

            if output_config.get('save_video', True):
                out.write(processed_frame)

            frame_times.append(time.time() - start)
            processed_frames += 1

            cv2.imshow("Traffic Counter", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Cap to ~30 FPS only for video files (live sources should run as fast as possible)
            elapsed = time.time() - start
            if not is_live and elapsed < 1.0 / 30.0:
                time.sleep(1.0 / 30.0 - elapsed)

    except Exception as e:
        logging.error(f"Error processing frames: {e}")
        raise
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        if output_config.get('save_csv', True):
            csv_path = output_config.get('csv_path', 'crossings.csv')
            counter.save_results(csv_path)
            logging.info(f"Crossing events saved to {csv_path}")
        if output_config.get('save_video', True):
            logging.info(f"Processed video saved to {output_path}")


if __name__ == "__main__":
    main()
