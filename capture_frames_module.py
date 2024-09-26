import cv2
import os
import traceback
import logging
import time
import streamlink
import argparse
from datetime import datetime

logger = logging.getLogger()

class FrameCapture:
    def __init__(self, url=None, frames_to_skip=20, duration=43200):
        # Parse command-line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--url', type=str, help='Stream URL')
        parser.add_argument('--frame_queue', type=str, default=None, help='Path to the folder where frames are stored')
        parser.add_argument('--frames_to_skip', type=int, default=9, help='Number of frames to skip during processing')
        parser.add_argument('--duration', type=int, default=43200, help='Duration for frame capture in seconds')
        args = parser.parse_args()

        # Use provided arguments or fallback to defaults
        self.url = url or args.url or os.getenv('CAPTURE_URL', 'default_stream_url')
        self.frame_queue = args.frame_queue or os.getenv('FRAME_QUEUE', '/app/Internal/FramesQueue')
        self.frames_to_skip = frames_to_skip or args.frames_to_skip
        self.duration = duration or args.duration

    def start_capture(self):
        if not os.path.exists(self.frame_queue):
            os.makedirs(self.frame_queue)

        # Access the stream using Streamlink
        streams = streamlink.streams(self.url)
        if not streams:
            print("No streams found, exiting...")
            return False
        stream_url = streams['best'].url

        self.cap = cv2.VideoCapture(stream_url)
        if not self.cap.isOpened():
            print("Failed to open the stream.")
            return False

        self.start_time = time.time()
        return True

    def capture_frames(self):
        if not self.cap or not self.start_time:
            raise ValueError("Capture has not been initialized or failed to start.")

        frame_count = 0

        while time.time() - self.start_time < self.duration:
            ret, frame = self.cap.read()
            if not ret:
                break

            try:
                frame_count += 1
                if frame_count % self.frames_to_skip == 0:
                    img_name = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                    frame_filename = os.path.join(self.frame_queue, f"{img_name}.png")
                    frame_count = 0
                    cv2.imwrite(frame_filename, frame)
            except Exception as e:
                logging.error("An error occurred: %s", e, exc_info=True)
                traceback.print_exc()

        self.cap.release()

if __name__ == "__main__":
    capture = FrameCapture()
    if capture.start_capture():
        capture.capture_frames()
