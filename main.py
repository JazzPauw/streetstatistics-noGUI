import sys
import os
import shutil
import cv2
import time
import json
import pandas as pd
import webbrowser
import numpy as np
import logging
import traceback
import torch
import subprocess
import requests
from collections import deque
from datetime import datetime, timedelta
from PyQt5 import uic
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QRect
from PyQt5.QtGui import QPixmap, QImage, QClipboard
from PyQt5.QtWidgets import QGridLayout, QLabel, QMainWindow, QCheckBox, QApplication, QDialog, QVBoxLayout, QPushButton, QLineEdit, QMessageBox, QFileDialog, QGraphicsScene, QGraphicsView
from capture_frames_module import FrameCapture
# Import neccesary imports

current_version = "v0.2"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
frame_queue_dir = os.path.join(BASE_DIR, 'Internal', 'FramesQueue')
frame_holding_dir = os.path.join(BASE_DIR, 'Internal', 'FrameHolding')
live_stats_path = os.path.join(BASE_DIR, 'Internal', 'live_stats.json')
overall_stats_path = os.path.join(BASE_DIR, 'Internal', 'overall_stats.json')
user_settings_path = os.path.join(BASE_DIR, 'Internal', 'user_settings.json')
video_output_dir = os.path.join(BASE_DIR, 'Video_Outputs')
data_output_dir = os.path.join(BASE_DIR, 'Data_outputs')
log_dir = os.path.join(BASE_DIR, 'logs')

# Custom or new model? Define the path here: (Must be in Internal folder) 
model_path = os.path.join(BASE_DIR, 'Internal', "yolov9e-seg.pt")


# Define the dynamic paths 
log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
log_filepath = os.path.join(log_dir, log_filename)
logging.basicConfig(filename=log_filepath,
                    level=logging.ERROR,  # Log only errors and above
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


print(torch.cuda.is_available())
print("Available GPU: ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

# Widget to input livestream link
class InputDialog(QDialog):
    def __init__(self, parent=None):
        super(InputDialog, self).__init__(parent)
        self.setWindowTitle("Input YouTube Live Stream Link")
        self.layout = QVBoxLayout(self)
        self.lineEdit = QLineEdit(self)
        self.layout.addWidget(self.lineEdit)
        self.submitButton = QPushButton("Submit", self)
        self.layout.addWidget(self.submitButton)
        self.setLayout(self.layout)
        self.submitButton.clicked.connect(self.accept)

    def get_link(self):
        return self.lineEdit.text()


# Thread that communicates to capture_frames_module 
class CaptureThread(QThread):
    def __init__(self, url, dest_folder, points):
        super().__init__()
        self.frame_capture = FrameCapture(url, points, model_path)

    def run(self):
        if self.frame_capture.start_capture():
            self.frame_capture.capture_frames()

    # The following functions are all called somewhere else in main.py, calling these changes a value inside of the capture_frames_module and sometimes it even gets sent through to model.py
    def stop_capture(self):
        self.frame_capture.stop_capture()
    def set_show_lines(self, show):
        self.frame_capture.set_show_lines(show)
    def start_video_write(self, save):
        self.frame_capture.start_video_write(save)
    def update_points(self, points):
        self.frame_capture.update_points(points)
    def toggle_processing(self):
        active = not self.frame_capture.processing_active
        self.frame_capture.set_processing_active(active)

# Display Thread deals mainly with the updating of the PyQT display once data comes in. 
class FrameDisplayThread(QThread):
    update_display = pyqtSignal(str)

    def __init__(self, folder, avg_people_label, label_2, label_3, label_4, label_5, label_6, label_7, label_8, label_9, label_10):
        super().__init__()
        self.folder = folder
        self.running = True
        self.avg_people_label = avg_people_label
        self.label_2 = label_2
        self.label_3 = label_3
        self.label_4 = label_4
        self.label_5 = label_5
        self.label_6 = label_6
        self.label_7 = label_7
        self.label_8 = label_8
        self.label_9 = label_9
        self.label_10 = label_10
        self.processing_active = False
        self.fps = None
        self.last_time = datetime.now()

    def processing_active_flag(self, enable):
        self.processing_active = enable

    def run(self):
        self.order = 0
        self.vehicle_class_count = 0
        self.human_class_count = 0 
        self.ppl_ph = 0
        self.dir_1_ph = 0  
        self.dir_2_ph = 0  
        queue = deque()
        # Frames captured come in to the folder rapidly, sometimes the connection to the youtube link is not as steady or the model takes longer to process. In any case, this can cause issues
        # For that reason, failsafes were created to do easy processing instead of the advanced processing using the model. 
        # Its important to note that the program starts at processing until the user starts the advanced processing. 
        while self.running:
            try:
                frames = sorted(os.listdir(self.folder))
                if len(frames) > 5:
                    if len(frames) > 6:
                        self.easy_processing(frames, queue)
                    else:
                        if not self.processing_active == True:
                            self.easy_processing(frames, queue)
                        else:
                            self.advanced_processing(frames, queue)
            except Exception as e:
                logging.error("An error occured: %s", e, exc_info=True)
                traceback.print_exc()
                time.sleep(0.1)
    def stop(self):
        self.running = False
        self.wait()

    def easy_processing(self, frames, queue):
        frame_to_show = os.path.join(self.folder, frames[-5])
        frame_name = frames[-5]
        frame_name, _ = os.path.splitext(frame_name)
        with open(live_stats_path, 'r') as json_file:
            try:
                data = json.load(json_file)
            except json.JSONDecodeError:
                data = {}
        with open(live_stats_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        self.update_display.emit(frame_to_show)
        os.remove(os.path.join(self.folder, frames[0])) 
        frame_data = data.get(frame_name, {})
        parts = frame_name.split('_')
        date_part = parts[0]
        time_part = parts[1]
        ms_part = parts[2]
        current_datetime_str = f"{date_part}{time_part}{int(ms_part):03d}000"
        current_datetime = datetime.strptime(current_datetime_str, '%Y%m%d%H%M%S%f')
        delta = current_datetime - self.last_time
        seconds = delta.total_seconds()
        fps = 1 / seconds if seconds > 0 else float('inf')
        self.fps = fps
        self.label_10.setText(f'Frames per Second: {self.fps}')



    def advanced_processing(self, frames, queue):
        frame_to_show = os.path.join(self.folder, frames[-5])
        frame_name = frames[-5]
        frame_name, _ = os.path.splitext(frame_name)
        with open(live_stats_path, 'r') as json_file:
            try:
                data = json.load(json_file)
            except json.JSONDecodeError:
                data = {}
        with open(live_stats_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        self.update_display.emit(frame_to_show)
        os.remove(os.path.join(self.folder, frames[0]))
        frame_data = data.get(frame_name, {})
        parts = frame_name.split('_')
        date_part = parts[0]
        time_part = parts[1]
        ms_part = parts[2]
        current_datetime_str = f"{date_part}{time_part}{int(ms_part):03d}000"
        current_datetime = datetime.strptime(current_datetime_str, '%Y%m%d%H%M%S%f')
        delta = current_datetime - self.last_time
        seconds = delta.total_seconds()
        fps = 1 / seconds if seconds > 0 else float('inf')
        self.fps = fps
        self.last_time = current_datetime
        one_hour_ago = current_datetime - timedelta(hours=1)
        with open(overall_stats_path, 'r') as json_file:
            try:
                overall_stats = json.load(json_file)
            except json.JSONDecodeError:
                overall_stats = []

        if overall_stats:
            for i in overall_stats[self.order:]:
                timestamp = i[0]
                direction = i[1]
                class_id = i[2]
                print(class_id)
                if int(class_id) in [0]:
                    self.human_class_count += 1
                elif int(class_id) in [2,3,5,7]:
                    self.vehicle_class_count += 1
                self.order += 1
                self.ppl_ph += 1
                print(direction)
                if direction == 1: 
                    self.dir_1_ph += 1
                elif direction == 2:  
                    self.dir_2_ph += 1
                queue.append([timestamp, direction, class_id]) 
            while True:
                if len(queue) > 1:
                    timestamp_dt = datetime.fromisoformat(queue[0][0])
                    if timestamp_dt <= one_hour_ago:
                        self.ppl_ph -= 1
                        if queue[0][1] == 1:
                            self.dir_1_ph -= 1
                        else:
                            self.dir_2_ph -= 1
                        queue.popleft()                        
                    else:
                        break
                else:
                    break
        total_all_time = frame_data.get('total', 0)
        direction_1 = frame_data.get('direction_1', 0)
        direction_2 = frame_data.get('direction_2', 0)
        total = direction_1 + direction_2 
        self.avg_people_label.setText(f'Targets per hour: {self.ppl_ph}')
        self.label_2.setText(f'Target p/h direction A: {self.dir_1_ph}')
        self.label_3.setText(f'Target p/h direction B: {self.dir_2_ph} ')
        self.label_4.setText(f'Total targets counted: {total}')
        self.label_5.setText(f'Total from direction A: {direction_1}')
        self.label_6.setText(f'Total from direction B: {direction_2}')
        self.label_7.setText(f'Total Vehicles: {self.vehicle_class_count}')
        self.label_8.setText(f'Total Humans: {self.human_class_count}')
        self.label_9.setText(f'Total detections: {total_all_time}')
        self.label_10.setText(f'Frames per Second: {self.fps}')
        frame_data = data.pop(frame_name, {})
       


# The main PyQt window
class Schema(QMainWindow):
    def __init__(self):
        super(Schema, self).__init__()
        uic.loadUi(os.path.join(BASE_DIR, 'Internal', 'schema.ui'), self) 
        self.setup_ui()
        self.setup_threads()
        self.points = self.load_initial_points()
        
        
    def load_initial_points(self):
        points = load_points_from_file()
        if points is None:
            points = {
                'DeReg1L': [626, 726], 'DeReg1R': [1426, 653],
                'DeRegL': [650, 624], 'DeRegR': [1323, 575],
                'RegL': [675, 548], 'RegR': [1228, 506],
                'Reg1L': [592, 856], 'Reg1R': [1573, 762]}
            save_points_to_file(points)  
        return points
    
    def setup_ui(self):
        # Initialize UI objects and connect functions to buttons
        self.graphics_scene = QGraphicsScene(self)
        self.graphics_view.setScene(self.graphics_scene)
        self.graphics_view.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.finish_button = self.findChild(QPushButton, 'finish_button')
        self.start_button = self.findChild(QPushButton, 'start_button')
        self.show_box = self.findChild(QCheckBox, 'show_box')
        self.configure = self.findChild(QPushButton, 'configure')
        self.misc_button = self.findChild(QPushButton, 'pushButton_4')
        self.misc_button.setText(f'Export Stats')
        self.save_video = self.findChild(QCheckBox, 'save_video')
        self.misc_button.clicked.connect(self.misc)
        self.start_button.clicked.connect(self.start_ML)
        self.show_box.stateChanged.connect(self.update_show_lines)
        self.configure.clicked.connect(self.configure_box)
        self.save_video.stateChanged.connect(self.update_videowriter)
        self.finish_button.clicked.connect(self.finish_capture)
        self.select_media.setText('Enter Livestream Link (YT)')
        self.select_media.clicked.connect(self.open_file_dialog)
        self.avg_people_label = self.findChild(QLabel, 'label_1')
        self.label_2 = self.findChild(QLabel, 'label_2')
        self.label_3 = self.findChild(QLabel, 'label_3')
        self.label_4 = self.findChild(QLabel, 'label_4')
        self.label_5 = self.findChild(QLabel, 'label_5')
        self.label_6 = self.findChild(QLabel, 'label_6')
        self.label_7 = self.findChild(QLabel, 'label_7')
        self.label_8 = self.findChild(QLabel, 'label_8')
        self.label_9 = self.findChild(QLabel, 'label_9')
        self.label_10 = self.findChild(QLabel, 'label_10')
        self.avg_people_label.setText(f'Targets per hour: ???')
        self.label_2.setText(f'Target p/h direction A: ??? ')
        self.label_3.setText(f'Target p/h direction B: ??? ')
        self.label_4.setText(f'Total targets counted: ???')
        self.label_5.setText(f'Total from direction A: ???')
        self.label_6.setText(f'Total from direction B: ???')
        self.label_7.setText(f'Total Vehicles: ???')
        self.label_8.setText(f'Total Humans: ???')
        self.label_9.setText(f'Total detections: ???')
        self.label_10.setText(f'Frames per Second: ???')
        self.version_label = self.findChild(QLabel, 'version_label')
        self.version_label_2 = self.findChild(QLabel, 'version_label_2')
        latest_version = version_control(current_version)
        print(latest_version)
        if current_version == latest_version:
            self.version_label.setText(f"{current_version}")
        else:
            self.version_label.setText(f"A new version is available!")
            self.version_label_2.setText(f"{current_version} -> {latest_version}")
    def setup_threads(self):
        self.frame_display_thread = FrameDisplayThread(frame_queue_dir, self.avg_people_label, self.label_2, self.label_3, self.label_4, self.label_5, self.label_6, self.label_7, self.label_8, self.label_9, self.label_10)
        self.frame_display_thread.update_display.connect(self.display_frame)
        self.frame_display_thread.start()

    def closeEvent(self, event):
        self.finish_capture()
        event.accept()
    
    def misc(self):
        export_stats_path = os.path.join(BASE_DIR, 'export_stats.py')
        subprocess.Popen(["python", export_stats_path])
        webbrowser.open("http://127.0.0.1:5000")


        
    def update_show_lines(self):
        if hasattr(self, 'capture_thread') and self.capture_thread.isRunning():
            self.capture_thread.set_show_lines(self.show_box.isChecked())
    def update_videowriter(self):
        self.save_video.setEnabled(False)
        if hasattr(self, 'capture_thread') and self.capture_thread.isRunning():
            self.capture_thread.start_video_write(self.save_video.isChecked())

    def start_ML(self):
        print("starting process")
        self.configure.setEnabled(True)
        self.start_button.setEnabled(False)
        self.show_box.setEnabled(True)
        self.show_lines = True
        if hasattr(self, 'capture_thread') and self.capture_thread.isRunning():
            self.capture_thread.toggle_processing()
            self.start_button.setText("Start") 
            self.frame_display_thread.processing_active_flag(True)
        else:
            print("No active capture thread. Please select a source first.")
            QMessageBox.warning(self, "No Source", "Please select a video source before starting processing.")

    def configure_box(self):
        # Determine the latest frame in the FrameQueue
        frames = sorted(os.listdir(frame_queue_dir))
        if frames:
            latest_frame_path = os.path.join(frame_queue_dir, frames[-1])
            holding_path = frame_holding_dir
            if not os.path.exists(holding_path):
                os.makedirs(holding_path)
            # Copy the latest frame to the Holding folder
            holding_frame_path = os.path.join(holding_path, frames[-1])
            shutil.copy(latest_frame_path, holding_frame_path)
            dialog = PointConfigDialog(self.points, holding_frame_path, self)
            if dialog.exec_():
                self.points = dialog.get_points()
                print("Updated points:", self.points)
        else:
            QMessageBox.warning(self, "Error", "No frames available to configure.")
    def finish_capture(self):
        if hasattr(self, 'capture_thread') and self.capture_thread.isRunning():
            self.capture_thread.stop_capture()
            self.capture_thread.wait()
        self.frame_display_thread.stop()
        clear_live_stats(live_stats_path)
        self.clear_folder(frame_queue_dir)
        self.graphics_scene.clear()  

    def open_file_dialog(self):
        self.open_youtube_link_dialog()
        # Code for allowing file as source, not currently supported.
        # msgBox = QMessageBox()
        # msgBox.setWindowTitle("Select Source")
        # msgBox.setText("Choose the video source:")
        # file_button = msgBox.addButton("Video File", QMessageBox.AcceptRole)
        # link_button = msgBox.addButton("Live Stream Link", QMessageBox.ActionRole)
        # msgBox.exec_()

        # if msgBox.clickedButton() == file_button:
        #     self.handle_file_input()
        # elif msgBox.clickedButton() == link_button:
        #     self.open_youtube_link_dialog()
    def handle_file_input(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        video_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4)", options=options)
        if video_path:
            self.configure.setEnabled(True)
            self.start_button.setEnabled(True)
            self.load_video(video_path)

    def open_youtube_link_dialog(self):
        dialog = InputDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            youtube_link = dialog.get_link()
            self.configure.setEnabled(True)
            self.start_button.setEnabled(True)
            self.save_video.setEnabled(True)

            self.start_capture(youtube_link)

    def start_capture(self, url):
        dest_folder = frame_queue_dir
        if hasattr(self, 'capture_thread') and self.capture_thread.isRunning():
            self.capture_thread.terminate()
        self.capture_thread = CaptureThread(url, dest_folder, self.points)
        self.capture_thread.start()

    @pyqtSlot(str)
    def display_frame(self, frame_path):
        pixmap = QPixmap(frame_path)
        if not pixmap.isNull():
            self.graphics_scene.clear()
            self.graphics_scene.addPixmap(pixmap)
            self.graphics_view.fitInView(self.graphics_scene.sceneRect(), Qt.KeepAspectRatio)
        else:
            print("Failed to load image at {}".format(frame_path))

    def clear_folder(self, folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logging.error(f"Failed to delete {file_path}. Reason: {e} ", exc_info=True)
                print(f'Failed to delete {file_path}. Reason: {e}')

class DraggableGraphicsView(QGraphicsView):
    pointMoved = pyqtSignal(str, tuple)  # Signal to update point coordinates

    def __init__(self, scene, points, parent=None):
        super(DraggableGraphicsView, self).__init__(scene, parent)
        self.points = points
        self.scale_x = 1920 / 1280 
        self.scale_y = 1080 / 720
        self.dragging_point = None
        self.setMouseTracking(True)
    
    def mousePressEvent(self, event):
        pos_scene = self.mapToScene(event.pos())
        adjusted_x = pos_scene.x() * self.scale_x
        adjusted_y = pos_scene.y() * self.scale_y
    
        min_distance = float('inf')
        selected_point = None
        for key, (x, y) in self.points.items():
            distance = ((adjusted_x - x) ** 2 + (adjusted_y - y) ** 2)
            if distance < min_distance:
                min_distance = distance
                selected_point = key
    
        if min_distance < 80:  # Variable is treshold for mouse distance from point 
            self.dragging_point = selected_point
            self.pointMoved.emit(selected_point, (adjusted_x, adjusted_y))
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.dragging_point:
            pos = self.mapToScene(event.pos()).toPoint()
            self.points[self.dragging_point] = (pos.x()*self.scale_x, pos.y()*self.scale_y)
            self.pointMoved.emit(self.dragging_point, (pos.x()*self.scale_x, pos.y()*self.scale_y))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.dragging_point = None
        super().mouseReleaseEvent(event)

        
class PointConfigDialog(QDialog):
    def __init__(self, points, image_path, parent=None):
        super(PointConfigDialog, self).__init__(parent)
        self.points = points
        self.image_path = image_path
        self.original_image = cv2.imread(self.image_path)
        self.image = self.original_image.copy()
        self.init_ui()
        self.update_image()

    def init_ui(self):
        self.setWindowTitle("Configure Points")
        layout = QVBoxLayout(self)
        self.graphics_scene = QGraphicsScene(self)
        self.graphics_view = DraggableGraphicsView(self.graphics_scene, self.points, self)
        self.graphics_view.pointMoved.connect(self.update_point)
        layout.addWidget(self.graphics_view)

        grid_layout = QGridLayout()
        self.fields = {}
        row = 0
        for key, (x, y) in self.points.items():
            grid_layout.addWidget(QLabel(f"{key} x:"), row, 0)
            grid_layout.addWidget(QLabel(f"{key} y:"), row, 2)
            x_field = QLineEdit(str(x))
            y_field = QLineEdit(str(y))
            self.fields[key] = (x_field, y_field)
            grid_layout.addWidget(x_field, row, 1)
            grid_layout.addWidget(y_field, row, 3)
            btn_inc_x = QPushButton("+")
            btn_dec_x = QPushButton("-")
            btn_inc_y = QPushButton("+")
            btn_dec_y = QPushButton("-")
            btn_inc_x.clicked.connect(lambda checked, key=key: self.adjust_point(key, 'x', 1))
            btn_dec_x.clicked.connect(lambda checked, key=key: self.adjust_point(key, 'x', -1))
            btn_inc_y.clicked.connect(lambda checked, key=key: self.adjust_point(key, 'y', 1))
            btn_dec_y.clicked.connect(lambda checked, key=key: self.adjust_point(key, 'y', -1))
            grid_layout.addWidget(btn_inc_x, row, 4)
            grid_layout.addWidget(btn_dec_x, row, 5)
            grid_layout.addWidget(btn_inc_y, row, 6)
            grid_layout.addWidget(btn_dec_y, row, 7)
            row += 1
        layout.addLayout(grid_layout)

        self.save_button = QPushButton("Save and Close")
        self.save_button.clicked.connect(self.save_and_close)
        layout.addWidget(self.save_button)

    def closeEvent(self, event):
        if os.path.exists(self.image_path):
            os.remove(self.image_path)  # Safely remove the image 
        event.accept()  
    
    def get_points(self):
        return self.points
    
    def adjust_point(self, key, axis, delta):
        x_field, y_field = self.fields[key]
        if axis == 'x':
            new_value = int(x_field.text()) + delta
            x_field.setText(str(new_value))
        elif axis == 'y':
            new_value = int(y_field.text()) + delta
            y_field.setText(str(new_value))
        self.points[key] = (int(x_field.text()), int(y_field.text()))
        self.update_image()

    def update_point(self, key, pos):
        self.points[key] = pos
        x_field, y_field = self.fields[key]
        x_field.setText(str(pos[0]))
        y_field.setText(str(pos[1]))
        self.update_image()

    def update_image(self):
        updated_img = self.draw_all_regions(self.image.copy(), self.points)
        self.display_image(updated_img)


    def display_image(self, img):
        self.graphics_scene.clear()
        q_img = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_img)

        custom_width = 1280  
        custom_height = 720      
        scaled_pixmap = pixmap.scaled(custom_width, custom_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.graphics_scene.addPixmap(scaled_pixmap)
    
        rect = scaled_pixmap.rect()
        self.graphics_view.setSceneRect(rect.x(), rect.y(), rect.width(), rect.height())
        self.graphics_view.setFixedSize(custom_width, custom_height)

    
        
    def save_and_close(self):
        self.points = {key: [int(coord) for coord in value] for key, value in self.points.items()}
        if hasattr(self, 'capture_thread') and self.capture_thread.isRunning():
            self.capture_thread.update_points(self.points)
        save_points_to_file(self.points)
        os.remove(self.image_path)  # Clean up the image in the holding folder
        self.accept()
        
    def draw_all_regions(self, img, points):
        regions = {
            'Region1': ['RegL', 'RegR', 'DeRegR', 'DeRegL'],
            'Region3': ['DeReg1L', 'DeReg1R', 'Reg1R', 'Reg1L'],
            'Region2': ['DeRegL', 'DeRegR', 'DeReg1R', 'DeReg1L']
        }
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # RGB colors for lines
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1  
        font_thickness = 2  
        circle_radius = 15  
        circle_color = (0, 255, 255)
        circle_opacity = 0.5  
        overlay = img.copy()
        for color, region in zip(colors, regions.values()):
            pts = np.array([points[pt] for pt in region], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
            for pt_name in region:
                pt_x, pt_y = points[pt_name]
                cv2.circle(overlay, (int(pt_x), int(pt_y)), circle_radius, circle_color, -1)  # Draw circle on overlay
    
        # Blend the original image and the overlay with the circle
        cv2.addWeighted(overlay, circle_opacity, img, 1 - circle_opacity, 0, img)
    
        # Draw text after blending to ensure it is not semi-transparent
        for region in regions.values():
            for pt_name in region:
                pt_x, pt_y = points[pt_name]
                cv2.putText(img, pt_name, (int(pt_x + 15), int(pt_y + 15)), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    
        return img

def save_points_to_file(points):
    data = {"points": points}
    with open(user_settings_path, 'w') as f:
        json.dump(data, f, indent=4)
    print("Points saved to file.")

def load_points_from_file():
    try:
        with open(user_settings_path, 'r') as f:
            data = json.load(f)
            return data["points"]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print("Failed to load settings:", e)
        return None

def clear_live_stats(json_file_path):
    if os.path.exists(json_file_path):
        with open(json_file_path, 'w') as json_file:
            json.dump({}, json_file, indent=4)
        print(f"Cleared contents of {json_file_path}")
    else:
        print(f"{json_file_path} does not exist.")

def clear_overall_stats(json_file_path):
    if os.path.exists(json_file_path):
        with open(json_file_path, 'w') as json_file:
            json.dump([], json_file, indent=4)
        print(f"Cleared contents of {json_file_path}")
    else:
        print(f"{json_file_path} does not exist.")
        
# Used to clear up video writer files when video writer isnt initiated by user. Only deletes small files
def cleanup_small_files(directory):
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            if os.path.getsize(filepath) < 1_024_000:
                print(f"Deleting file: {filepath}, size: {os.path.getsize(filepath)} bytes")
                os.remove(filepath)

def clear_frame_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logging.error(f"Failed to delete {file_path}. Reason: {e} ", exc_info=True)
            print(f'Failed to delete {file_path}. Reason: {e}')

def version_control(current_version):
    url = "https://github.com/JazzPauw/streetstatistics/releases/latest"
    try:
        response = requests.get(url)
        response.raise_for_status()
        latest_version = response.json()['tag_name']
        if latest_version > current_version:
            return latest_version
        else:
            return current_version
    except requests.RequestException as e:
        logging.error(f"Failed to check updat: {e}")
        return current_version
    
def main():
    try:
        if torch.cuda.is_available():
            print("using torch")
            device = torch.device('cuda')
        else:
            print("not using torch")
        clear_frame_folder(frame_queue_dir)
        clear_live_stats(live_stats_path)
        clear_overall_stats(overall_stats_path)
        cleanup_small_files(video_output_dir)
        app = QApplication(sys.argv)
        window = Schema()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        logging.error("An error occured: %s", e, exc_info=True)        
        print(e)
        traceback.print_exc()


if __name__ == "__main__":
    main()