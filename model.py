import cv2
import numpy as np
from short import Sort
import torch
import os
import logging
from collections import deque
from datetime import datetime
import random
import json
import pickle
from ultralytics import YOLO
import argparse
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Torch: {torch.cuda.is_available()}")
logger = logging.getLogger()

def process_frame(frame, points, track_positions, model, tracker, showlines):
    points = points
    if points is None:
        points = {
            'DeReg1L': [626, 726], 'DeReg1R': [1426, 653],
            'DeRegL': [650, 624], 'DeRegR': [1323, 575],
            'RegL': [675, 548], 'RegR': [1228, 506],
            'Reg1L': [592, 856], 'Reg1R': [1573, 762]}

    frame_width, frame_height = frame.shape[:2]
    crop_x1, crop_y1, crop_x2, crop_y2, adjusted_points, frame = calculate_crop_and_adjust_points(points, frame, frame_width, frame_height)
    original = frame.copy()        
    cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]

    overlay = cropped_frame.copy()   
    results = model(cropped_frame, classes=[0, 2, 3, 5, 7], line_width=1, show_boxes=True, retina_masks=True, verbose=False)

    current_ids = set()
    total = 0
    direction_1, direction_2 = 0, 0

    try:
        for res in results:  
            boxes = res.boxes.xyxy.cpu().numpy().astype(int)
            masks = res.masks.xy if hasattr(res, 'masks') else []
            labels = res.boxes.cls.cpu().numpy().astype(int)
            tracks = tracker.update(boxes)

            for box, mask in zip(tracks.astype(int), masks):
                class_id = labels[total]
                total += 1
                xmin, ymin, xmax, ymax, track_id = box
                x_center = (xmin + xmax) // 2
                y_center = (ymin + ymax) // 2
                current_ids.add(track_id)
                new_position = (x_center, y_center)

                if track_id not in track_positions:
                    track_positions[track_id] = {'positions': deque(maxlen=15), 'color': (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 'registrations': [], 'counted': False, 'cls_id': class_id}
                track_positions[track_id]['positions'].append(new_position)

                pt = (int(round(x_center)), int(round(y_center)))
                track_positions[track_id]['positions'].append(pt)

                if mask.size > 0:
                    mask_poly = np.array(mask, dtype=np.int32).reshape((-1, 1, 2))
                    mask_poly = np.append(mask_poly, [mask_poly[0]], axis=0)
                    cv2.fillPoly(overlay, [mask_poly], (0, 0, 255))
                    if cv2.pointPolygonTest(mask_poly, pt, False) == 0:
                        track_positions[track_id]['color'] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    cv2.fillPoly(overlay, [mask_poly], (0,0,255))
    
                if len(track_positions[track_id]['positions']) > 1:
                    last_position = track_positions[track_id]['positions'][-2]
                    intersections = line_intersects(last_position, new_position, adjusted_points)
                    for tag in intersections:
                        track_positions[track_id]['registrations'].append(tag)
                    if "Registered_Reg1" in track_positions[track_id]['registrations'] and "Deregistered_DeReg" in track_positions[track_id]['registrations'] and not track_positions[track_id]['counted']:
                        track_positions[track_id]['counted'] = True
                        direction_1 += 1
                    elif "Registered_Reg" in track_positions[track_id]['registrations'] and "Deregistered_DeReg1" in track_positions[track_id]['registrations'] and not track_positions[track_id]['counted']:
                        track_positions[track_id]['counted'] = True
                        direction_2 += 1

        for track_id, data in track_positions.items():
            for i in range(1, len(data['positions'])):
                cv2.line(cropped_frame, data['positions'][i - 1], data['positions'][i], data['color'], 2)

        alpha = 0.6
        cropped_frame = cv2.addWeighted(cropped_frame, alpha, overlay, 1 - alpha, 0)

        for track_id in current_ids:
            positions = track_positions[track_id]['positions'][-1]
            cv2.circle(cropped_frame, positions, radius=4, color=(255, 105, 180), thickness=-1)
            cv2.putText(cropped_frame, f"Id:{track_id}", (positions[0], positions[1] - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        
        original[crop_y1:crop_y2, crop_x1:crop_x2] = cropped_frame

        if showlines:
            original = draw_all_regions(original, points)
        
        processed_frame = original
        stats = {'direction_1': direction_1,
                    'direction_2': direction_2,
                    'timestamp': datetime.now().isoformat()  # Convert timestamp to a string format
}
        save_outputs(track_positions, processed_frame, stats, tracker)
    except Exception as e:
        logging.error("An error occurred: %s", e, exc_info=True)
        stats = {'direction_1': direction_1,
                    'direction_2': direction_2,
                    'timestamp': datetime.now().isoformat()  # Convert timestamp to a string format
                    }
        save_outputs(track_positions, processed_frame, stats, tracker)




def save_outputs(track_positions, processed_frame, stats, tracker):
    data_transfer_dir = "/app/DataTransfer"

    # Save processed frame (image)
    processed_frame_path = os.path.join(data_transfer_dir, 'processed_frame.png')
    cv2.imwrite(processed_frame_path, processed_frame)
    
    # Save SORT tracker object
    sort_object_file = os.path.join("/app/DataTransfer", 'sort_object.pkl')
    with open(sort_object_file, 'wb') as f:
        pickle.dump(tracker, f)

    # Save track_positions using pickle
    track_positions_path = os.path.join(data_transfer_dir, 'track_positions.pkl')
    with open(track_positions_path, 'wb') as pickle_file:
        pickle.dump(track_positions, pickle_file)

    # Save stats using JSON (if needed)
    stats_path = os.path.join(data_transfer_dir, 'stats.json')
    with open(stats_path, 'w') as stats_file:
        json.dump(stats, stats_file, indent=4)

    print("Outputs saved successfully.")



# Function based on a mathematical formula to find out if one line crosses with another line 
def line_intersects(p1, p2, adjusted_points):
    def intersects(q1, q2):
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)

    result = []
    lines = [('RegL', 'RegR'), ('Reg1L', 'Reg1R'), ('DeRegL', 'DeRegR'), ('DeReg1L', 'DeReg1R')]
    for line in lines:
        if intersects(adjusted_points[line[0]], adjusted_points[line[1]]):
            if line[0] == "RegL":
                result.append("Registered_Reg")
            if line[0] == "Reg1L":
                result.append("Registered_Reg1")
            if line[0] == "DeRegL":
                result.append("Deregistered_DeReg")
            if line[0] == "DeReg1L":
                result.append("Deregistered_DeReg1")
    return result

def draw_all_regions(img, points):
    # Define regions using points
    regions = {
        'Region1': ['RegL', 'RegR', 'DeRegR', 'DeRegL'],
        'Region3': ['DeReg1L', 'DeReg1R', 'Reg1R', 'Reg1L'],
        'Region2': ['DeRegL', 'DeRegR', 'DeReg1R', 'DeReg1L']
    }
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  
    for color, region in zip(colors, regions.values()):
        pts = np.array([points[pt] for pt in region], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)

    label_b_pos = ((points['RegL'][0] + points['RegR'][0]) // 2,
                (points['RegL'][1] + points['RegR'][1]) // 2)
    label_a_pos = ((points['Reg1L'][0] + points['Reg1R'][0]) // 2,
                (points['Reg1L'][1] + points['Reg1R'][1]) // 2)

    cv2.putText(img, "A", label_a_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "B", label_b_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    return img  # Return the modified image

def draw_all_regions_on_cropped_frame(img, points, crop_x1, crop_y1):

    adjusted_points = {k: (v[0] - crop_x1, v[1] - crop_y1) for k, v in points.items()}
    
    # Define regions using adjusted points
    regions = {
        'Region1': ['RegL', 'RegR', 'DeRegR', 'DeRegL'],
        'Region3': ['DeReg1L', 'DeReg1R', 'Reg1R', 'Reg1L'],
        'Region2': ['DeRegL', 'DeRegR', 'DeReg1R', 'DeReg1L']
    }
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Adjusted for visibility
    for color, region in zip(colors, regions.values()):
        pts = np.array([adjusted_points[pt] for pt in region], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
    
    return img  # Return the modified cropped image


def calculate_crop_and_adjust_points(points, frame, frame_height, frame_width, padding=100):
    # The structure can be dynamic, we need to make sure we find out which point is where before we use them to calculate the coordinates for the cropped frame. 
    crop_x1 = min(points['RegL'][0], points['RegR'][0], points['Reg1L'][0], points['Reg1R'][0])
    crop_x2 = max(points['RegL'][0], points['RegR'][0], points['Reg1L'][0], points['Reg1R'][0])
    
    crop_y1 = min(points['RegL'][1], points['RegR'][1], points['Reg1L'][1], points['Reg1R'][1])
    crop_y2 = max(points['RegL'][1], points['RegR'][1], points['Reg1L'][1], points['Reg1R'][1])

    left_x = crop_x1  
    right_x = crop_x2  

    if crop_y1 < crop_y2:  
        top_left_x = left_x
        top_left_y = crop_y1
        bottom_right_x = right_x
        bottom_right_y = crop_y2
    else:  
        top_left_x = right_x
        top_left_y = crop_y2
        bottom_right_x = left_x
        bottom_right_y = crop_y1

    padded_top_left_x = max(top_left_x - padding, 0) 
    padded_top_left_y = max(top_left_y - padding, 0)  
    padded_bottom_right_x = min(bottom_right_x + padding, frame_width) 
    padded_bottom_right_y = min(bottom_right_y + padding, frame_height)  

    adjusted_points = {k: (v[0] - padded_top_left_x, v[1] - padded_top_left_y) for k, v in points.items()}

    return padded_top_left_x, padded_top_left_y, padded_bottom_right_x, padded_bottom_right_y, adjusted_points, frame

def load_sort_tracker():
    sort_object_file = os.path.join("/app/DataTransfer", 'sort_object.pkl')
    with open(sort_object_file, 'rb') as f:
        tracker = pickle.load(f)
        return tracker

def load_track_positions():
    track_positions_path = os.path.join("/app/DataTransfer", 'track_positions.pkl')
    try:
        with open(track_positions_path, 'rb') as pickle_file:
            track_positions = pickle.load(pickle_file)
        return track_positions
    except (FileNotFoundError, pickle.PickleError) as e:
        print(f"Error reading pickle file: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_path', type=str, help='Path to the frame image')
    parser.add_argument('--points', type=str, help='Points data (JSON format expected)')
    parser.add_argument('--model_path', type=str, default='yolov9e-seg.pt', help='Path to the YOLO model')
    args = parser.parse_args()

    # Use provided or default values
    frame_path = args.frame_path if args.frame_path else os.getenv('FRAME_PATH')
    points = json.loads(args.points) if args.points else json.loads(os.getenv('POINTS', '{}'))
    model_path = args.model_path if args.model_path else os.getenv('MODEL_PATH', 'yolov9e-seg.pt')

    if not points:
        print("Warning! Error loading points.")

    # Load the YOLO model
    model = YOLO(model_path).to(device)

    # Load the SORT tracker and track positions
    tracker = load_sort_tracker()
    track_positions = load_track_positions()
    
    direction_1, direction_2 = 0, 0

    # Read the frame and process it
    frame = cv2.imread(frame_path)
    process_frame(frame, points, track_positions, model, tracker, True)

if __name__ == '__main__':
    main()