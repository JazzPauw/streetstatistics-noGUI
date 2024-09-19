import cv2
import os
import traceback
import logging
import time
import streamlink
from datetime import datetime
from model import FrameProcessor
# Import neccesary imports

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
save_directory = os.path.join(BASE_DIR, 'Video_Outputs')  
dest_folder = os.path.join(BASE_DIR, 'Internal', 'FramesQueue') 
# Define the localized paths 

logger = logging.getLogger()

# Timestamp for video output file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"{timestamp}.mp4"
output_path = os.path.join(save_directory, filename)

class FrameCapture:
    def __init__(self, url, points, model_path='yolov9e-seg.pt', frames_to_skip=9, duration=43200):
        self.url = url
        self.dest_folder = dest_folder  
        self.model_path = model_path
        self.frames_to_skip = frames_to_skip # The streamlink connection collects all the frames of the livestream, this is more than the program could ever handle (and many dupe frames), we skip some frames for this reason
        self.track_positions = {}
        self.duration = duration
        self.show_lines = True
        self.processing_active = False
        self.points = points
        self.video_write = False
        self.stop_capturing = False
        self.frame_processor = FrameProcessor(points=self.points, model_path=self.model_path)
        self.video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 5, (1920, 1080))

    def start_capture(self):
        if not os.path.exists(self.dest_folder):
            os.makedirs(self.dest_folder)
        else:
            for file in os.listdir(self.dest_folder):
                os.remove(os.path.join(self.dest_folder, file))

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
        last_time = self.start_time

        while time.time() - self.start_time < self.duration:
            ret, frame = self.cap.read()
            if not ret:
                break
            try:
                current_time = time.time()
                frame_interval = current_time - last_time
                last_time = current_time

                frame_count += 1

                if frame_count % self.frames_to_skip == 0:
                    if self.processing_active:
                        frame, track_positions, img_name = self.frame_processor.process_frame(frame, self.points, self.track_positions, self.show_lines)
                        time.sleep(0.05)
                        self.track_positions = track_positions
                    else:
                        img_name = datetime.now()
                        time.sleep(0.1)
                    img_name = f"{img_name.strftime('%Y%m%d_%H%M%S')}_{img_name.microsecond // 1000:03d}"
                    while len(os.listdir(self.dest_folder)) >= 6:
                        print("Waiting for directory to clear...")
                        time.sleep(0.1)  # Wait for 0.1 seconds before checking again
                    frame_filename = os.path.join(self.dest_folder, f"{img_name}.png")
                    cv2.imwrite(frame_filename, frame)
                    if self.video_write:
                        self.video_writer.write(frame)
                    if self.stop_capturing:
                        if self.video_write:
                            self.video_writer.release()
                            print("Saved video, probably")
                        break
            except Exception as e:
                logging.error("An error occured: %s", e, exc_info=True)        
                traceback.print_exc()
                print(f"ERROR: {e}")

        self.cap.release()

    def set_show_lines(self, show):
        self.show_lines = show

    def update_points(self, points):
        self.points = {key: [int(coord) for coord in value] for key, value in points.items()}
        print(f"New points: {self.points} ")
        
    def set_processing_active(self, active):
        self.processing_active = active

    def start_video_write(self, save):
        self.video_write = save
        print("Video writing")

    def stop_capture(self):
        self.stop_capturing = True
