import os
import logging
import docker
import time
import cv2
import sys
import shutil
import json
import pickle
from datetime import datetime
from short import Sort

# Define inputs
url = "https://www.youtube.com/watch?v=R8LU4PCZdgo"
frames_to_skip = 20
duration = 432000
points = {
    'DeReg1L': [605, 865], 'DeReg1R': [1256, 871],
    'DeRegL': [625,759], 'DeRegR': [1165, 775],
    'RegL': [640, 676], 'RegR': [1087, 700],
    'Reg1L': [588, 991], 'Reg1R': [1377, 976]
}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
frame_queue = os.path.join(BASE_DIR, 'Internal', 'FramesQueue')
model_path = os.path.join(BASE_DIR, "Internal", "yolov9e-seg.pt")

log_dir = os.path.join(BASE_DIR, 'logs')
data_transfer_dir = os.path.join(BASE_DIR, 'Internal', 'DataTransfer')
sort_object_file = os.path.join(data_transfer_dir, 'sort_object.pkl')

Video_outputs = os.path.join(BASE_DIR, "Video_outputs")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(os.path.join(Video_outputs, 'output_video.mp4'), fourcc, 10, (1920, 1080))
video_write = True
# Set up logging
logging.basicConfig(filename=os.path.join(log_dir, 'capture_frames.log'), level=logging.INFO)

# Initialize Docker client
client = docker.from_env()

# Function to stop the Docker container
def stop_docker_container(container):
    try:
        if container:
            print(f"Stopping container {container}")
            container.stop()
            print("Container stopped successfully.")
    except Exception as e:
        print("Stopped {e}")

# Function to load Docker image if not already loaded
def load_image_if_not_exists(image_name, tar_file):
    try:
        image = client.images.get(image_name)
        print(f"Docker image {image_name} is already loaded.")
    except docker.errors.ImageNotFound:
        print(f"Loading Docker image from {tar_file}...")
        with open(tar_file, 'rb') as file:
            client.images.load(file.read())

# Function to initialize SORT object and save it to file
def initialize_sort():
    print("Initializing SORT object and saving to file...")
    tracker = Sort()
    with open(sort_object_file, 'wb') as f:
        pickle.dump(tracker, f)
        
def initialize_track_positions():
    track_positions = {}  # Initialize empty dictionary for track positions
    track_positions_path = os.path.join(data_transfer_dir, 'track_positions.pkl')
    
    # Write the initialized track_positions to a pickle file
    with open(track_positions_path, 'wb') as pickle_file:
        pickle.dump(track_positions, pickle_file)

    print(f"Initialized track_positions and saved to {track_positions_path}")

# Function to clear the frame queue folder
def clear_frame_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logging.error(f"Failed to delete {file_path}. Reason: {e}", exc_info=True)

if __name__ == '__main__':
    try:
        # Clear the frame queue and load the capture Docker image
        
        clear_frame_folder(frame_queue)
        
        capture_frame_tar = 'capture_frames_image.tar'
        capture_frame_image = 'capture_frames_image'
        load_image_if_not_exists(capture_frame_image, capture_frame_tar)

        # Initialize SORT and save to file
        initialize_sort()
        initialize_track_positions()
        container = client.containers.run(
            'capture_frames_image',
            detach=True,
            remove=True,
            environment={
                "CAPTURE_URL": url,
                "FRAMES_TO_SKIP": frames_to_skip,
                "DURATION": duration
            },
            volumes={frame_queue: {'bind': '/app/Internal/FramesQueue', 'mode': 'rw'}},
            labels={"module": "capture_frames"}
        )

        collected_frames = False
        stop_initiated = False 
        direction_1 = 0
        direction_2 = 0

        # Monitor frames in the frame_queue
        while True:
            time.sleep(1)
            frames = sorted(os.listdir(frame_queue))
            
            # Stop capture once we have collected enough frames
            if len(frames) >= 5 and not stop_initiated:
                stop_docker_container(container)
                stop_initiated = True
                collected_frames = True
                time.sleep(10)

            if collected_frames:
                # Start the model Docker container to process the frames
                frames = sorted(os.listdir(frame_queue))
                for frame in frames:
                    frame_path = os.path.join(frame_queue, frame)
                    print(f"Processing frame: {frame_path}")

                    # Run the model container
                    model_container_id = client.containers.run(
                        'model_image',
                        detach=True,
                        remove=True,
                        environment={
                            "FRAME_PATH": f'/app/Internal/FramesQueue/{frame}',
                            "POINTS": json.dumps(points),
                            "MODEL_PATH": model_path,
                            "YOLO_CONFIG_DIR": "/app/UltralyticsConfig"  
                        },
                        volumes={
                            frame_queue: {'bind': '/app/Internal/FramesQueue', 'mode': 'ro'},
                            data_transfer_dir: {'bind': '/app/DataTransfer', 'mode': 'rw'},
                            '/path/to/local/UltralyticsConfig': {'bind': '/app/UltralyticsConfig', 'mode': 'rw'}
                        },
                        device_requests=[{
                            'Capabilities': [['gpu']]}],
                        labels={"module": "model_image"}
                    )
                    model_container_id.wait()
                    try:
                        with open(os.path.join(data_transfer_dir, 'stats.json'), 'r') as stats_file:
                            stats = json.load(stats_file)
                    except Exception as e:
                        print(e)
                    if video_write:
                        processed_frame_path = os.path.join(data_transfer_dir, 'processed_frame.png')
                        processed_frame = cv2.imread(processed_frame_path)
                        video_writer.write(processed_frame)

                    print(f"Frame processed")
                    if stats['direction_1'] > 0:
                        direction_1 += stats['direction_1']
                    if stats['direction_2'] > 0:
                        direction_1 += stats['direction_2']
                    print(direction_1)
                    print(direction_2)
                break
        video_writer.release()
        print(datetime.now())
            
    except Exception as e:
        print(e)
        logging.error(f"Unhandled exception: {e}", exc_info=True)