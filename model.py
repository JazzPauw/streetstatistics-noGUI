import cv2
import numpy as np
import os
import random
import json
import logging
from collections import deque
from datetime import datetime
from short import Sort  
from ultralytics import YOLO  
# Neccesary imports

selected_point = None
enable_editing = True  

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
live_stats_path = os.path.join(BASE_DIR, 'Internal', 'live_stats.json')
overall_stats_path = os.path.join(BASE_DIR, 'Internal', 'overall_stats.json')
user_settings_path = os.path.join(BASE_DIR, 'Internal', 'user_settings.json')
log_dir = os.path.join(BASE_DIR, 'logs')

# Define the dynamic paths 
log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
log_filepath = os.path.join(log_dir, log_filename)
logging.basicConfig(filename=log_filepath,
                    level=logging.ERROR,  # Log only errors and above
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


class FrameProcessor:
    def __init__(self, points, model_path='yolov9e-seg.pt'):
        self.model = YOLO(model_path)
        self.tracker = Sort()
        self.points = points
        self.track_positions = {}  # To maintain state across frames
        self.direction_1 = 0
        self.direction_2 = 0
        self.json_file_path = live_stats_path
        self.overall_stats_path = overall_stats_path
    def process_frame(self, frame, points, track_positions, showlines):
        self.points = self.load_points_from_file()
        self.track_positions = track_positions
        if self.points is None:
            # Revert to predefined points if none can be found (user can edit this in the program)
            self.points = {
                'DeReg1L': [626, 726], 'DeReg1R': [1426, 653],
                'DeRegL': [650, 624], 'DeRegR': [1323, 575],
                'RegL': [675, 548], 'RegR': [1228, 506],
                'Reg1L': [592, 856], 'Reg1R': [1573, 762]}

        frame_width, frame_height = frame.shape[:2]
        crop_x1, crop_y1, crop_x2, crop_y2, adjusted_points, frame = self.calculate_crop_and_adjust_points(frame, frame_width, frame_height)
        original = frame.copy()        
        cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        # cropped_frame = self.draw_all_regions_on_cropped_frame(cropped_frame, self.points, crop_x1, crop_y1)

        # Crop frame based on the points + some padding
        # Save original frame as well, to overlay the processed cropped frame over

        overlay = cropped_frame.copy()   

        # Run the yolov9 segmentation model, using classes [0,2,3,5,7] which makes up humans and various vehicles. 
        results = self.model(cropped_frame, classes=[0,2,3,5,7], line_width=1, show_boxes=True, retina_masks=True, verbose=False)
        
        current_ids = set()
        total = 0
        try:
            # Start going through each result one by one
            for res in results:  
                boxes = res.boxes.xyxy.cpu().numpy().astype(int)
                masks = res.masks.xy if hasattr(res, 'masks') else []
                labels = res.boxes.cls.cpu().numpy().astype(int)  
                tracks = self.tracker.update(boxes)
                iterative = 0
                for box, mask in zip(tracks.astype(int), masks):
                    class_id = labels[iterative]
                    iterative += 1
                    total += 1
                    xmin, ymin, xmax, ymax, track_id = box
                    # Calculate the center of the mask, this is our anchor position
                    x_center = (xmin + xmax) // 2
                    y_center = (ymin + ymax) // 2
                    current_ids.add(track_id)
                    new_position = (x_center, y_center)
                    # Give the target an entry in track_positions, which stores data about the target. 
                    if track_id not in self.track_positions:
                        random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                        self.track_positions[track_id] = {'positions': deque(maxlen=15), 'color': random_color, 'registrations' : [], 'counted': False, 'cls_id' : class_id}
                    
                    # Add the last found position to the track_positions, to draw tracks later on 
                    pt = (int(round(x_center)), int(round(y_center)))
                    self.track_positions[track_id]['positions'].append(pt)

                    # Draw mask 
                    if mask.size > 0:
                        mask_poly = np.array(mask, dtype=np.int32).reshape((-1, 1, 2))
                        mask_poly = np.append(mask_poly, [mask_poly[0]], axis=0)
                        if cv2.pointPolygonTest(mask_poly, pt, False) == 0:
                            self.track_positions[track_id]['color'] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                        cv2.fillPoly(overlay, [mask_poly], (0,0,255))
        
        
                    # Register targets if they have crossed a line in the structure
                    if len(self.track_positions[track_id]['positions']) > 1:
                        last_position = self.track_positions[track_id]['positions'][-2]
                        intersections = self.line_intersects(last_position, new_position, adjusted_points)
                        for tag in intersections:
                            self.track_positions[track_id]['registrations'].append(tag)
                        if "Registered_Reg1" in self.track_positions[track_id]['registrations'] and "Deregistered_DeReg" in self.track_positions[track_id]['registrations'] and self.track_positions[track_id]['counted'] == False:
                            self.track_positions[track_id]['counted'] = True
                            self.direction_1 += 1
                            self.log_target(1, self.track_positions[track_id]['cls_id'])
                        elif "Registered_Reg" in self.track_positions[track_id]['registrations'] and "Deregistered_DeReg1" in self.track_positions[track_id]['registrations'] and self.track_positions[track_id]['counted'] == False:
                            self.track_positions[track_id]['counted'] = True
                            self.direction_2 += 1
                            self.log_target(2, self.track_positions[track_id]['cls_id'])
            # Draw tracks 
            for track_id, data in self.track_positions.items():
                for i in range(1, len(data['positions'])):
                    cv2.line(cropped_frame, data['positions'][i - 1], data['positions'][i], data['color'], 2)

            # Add the overlay onto the cropped frame with 0.6 transparency 
            alpha = 0.6
            cropped_frame = cv2.addWeighted(cropped_frame, alpha, overlay, 1 - alpha, 0)

            for track_id in current_ids:
                positions = self.track_positions[track_id]['positions'][-1]
                cv2.circle(cropped_frame, positions, radius=4, color=(255, 105, 180), thickness=-1)
                cv2.putText(cropped_frame, f"Id:{track_id}", (positions[0], positions[1] - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            
            # Add the cropped frame back onto the original image. 
            original[crop_y1:crop_y2, crop_x1:crop_x2] = cropped_frame
            
            if showlines:
                original = self.draw_all_regions(original, self.points)
            original = cv2.putText(original, f"Count: {self.direction_1+self.direction_2}", (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA) 
            
            # Cleanup: remove tracks not updated in the current frame
            self.track_positions = {key: val for key, val in self.track_positions.items() if key in current_ids}
            self.stats = { 'direction_1' : self.direction_1,
                          'direction_2' : self.direction_2,
                         'total' : total}
            try:
                with open(self.json_file_path, 'r') as json_file:
                    data = json.load(json_file)
            except (json.JSONDecodeError, FileNotFoundError):
                data = {}
            timestamp = datetime.now()
            img_name = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{timestamp.microsecond // 1000:03d}"
            img_name = f"{img_name}"
            data[img_name] = self.stats
            with open(self.json_file_path, 'w') as json_file:
                json.dump(data, json_file, indent=4)
            # write stats to json file, with key being img file name, when reading this will be the access and it will be removed by this
            
            return original, self.track_positions, timestamp
        except Exception as e:
            if 'NoneType' not in str(e):
                logging.error("An error occured: %s", e, exc_info=True)        
                print(f"LOG: {e}")
            if not self.track_positions:
                self.track_positions = {}
                self.stats = {'direction_1' : self.direction_1,
                              'direction_2' : self.direction_2,
                              'total' : total}
            try:
                with open(self.json_file_path, 'r') as json_file:
                    data = json.load(json_file)
            except (json.JSONDecodeError, FileNotFoundError):
                data = {}
            timestamp = datetime.now()
            img_name = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{timestamp.microsecond // 1000:03d}"
            img_name = f"{img_name}"
            data[img_name] = self.stats
            with open(self.json_file_path, 'w') as json_file:
                json.dump(data, json_file, indent=4)
            return original, self.track_positions, timestamp





    # Writes final data to overall_stats.json 
    def log_target(self, direction, class_id):
        try:
            with open(self.overall_stats_path, 'r') as file:
                stats = json.load(file)
        except (json.JSONDecodeError):
            stats = []
        current_time = datetime.now()
        key = [current_time.isoformat(), direction, str(class_id)]
        stats.append(key)
        with open(self.overall_stats_path, 'w') as file:
            json.dump(stats, file, indent=4)

    # Function based on a mathematical formula to find out if one line crosses with another line 
    def line_intersects(self, p1, p2, adjusted_points):
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
    
    def clear_track_positions(self):
        self.track_positions.clear()  # Clear tracking data if/when needed
    
    def draw_all_regions(self, img, points):
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

    def draw_all_regions_on_cropped_frame(self, img, points, crop_x1, crop_y1):

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


    def calculate_crop_and_adjust_points(self, frame, frame_height, frame_width, padding=100):
        # The structure can be dynamic, we need to make sure we find out which point is where before we use them to calculate the coordinates for the cropped frame. 
        crop_x1 = min(self.points['RegL'][0], self.points['RegR'][0], self.points['Reg1L'][0], self.points['Reg1R'][0])
        crop_x2 = max(self.points['RegL'][0], self.points['RegR'][0], self.points['Reg1L'][0], self.points['Reg1R'][0])
        
        crop_y1 = min(self.points['RegL'][1], self.points['RegR'][1], self.points['Reg1L'][1], self.points['Reg1R'][1])
        crop_y2 = max(self.points['RegL'][1], self.points['RegR'][1], self.points['Reg1L'][1], self.points['Reg1R'][1])

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

        adjusted_points = {k: (v[0] - padded_top_left_x, v[1] - padded_top_left_y) for k, v in self.points.items()}

        # Used to draw a circle for debugging
        # cv2.circle(frame, (padded_top_left_x, padded_top_left_y), 5, (255,255,255), -1) 
        # cv2.circle(frame, (padded_bottom_right_x, padded_bottom_right_y), 5, (255,255,255), -1) 

        return padded_top_left_x, padded_top_left_y, padded_bottom_right_x, padded_bottom_right_y, adjusted_points, frame
    
    def load_points_from_file(self):
        try:
            with open(user_settings_path, 'r') as f:
                data = json.load(f)
                return data["points"]
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print("Failed to load settings:", e)
            return None
