# **StreetStatistics**

**StreetStatistics** is a machine learning-powered application designed to easily gather and analyze statistics from live video feeds. It’s intended for anyone to use, providing an intuitive and user-friendly interface through **PyQt5**, allowing for quick insights from YouTube live streams.

## **Features**
- **Real-Time Analysis**: Process live video feeds to gather data on objects and movement.
- **Machine Learning Integration**: Powered by YOLO models for object detection and tracking.
- **Visual Insights**: Analyze hourly activity, object distributions, and directional data using built-in graphs.
- **Simple UI**: A clean and straightforward **PyQt5** interface that’s easy to use.
- **Data Export**: Export gathered statistics in a readable format or visualize them in an embedded dashboard.

## **How It Works**
StreetStatistics processes video feeds in real time, detecting objects like people and vehicles, and tracking their movement direction. It provides visualized insights such as:
- **Hourly activity**: Breakdown of detections per hour.
- **Class distribution**: Ratio of humans vs. vehicles.
- **Movement direction**: Tracks which direction objects are moving (e.g., Direction A vs. Direction B).

## **How to Install**

### **Prerequisites**
Before running the application, ensure you have the following installed:
- Python 3.7+
- **PyQt5**
- **OpenCV**
- **Streamlink**
- **Matplotlib**
- **Flask** 
- **Ultralytics YOLO**

### **Installation Steps**

1. **Clone the repository**:
    ```bash
    git clone https://github.com/JazzPauw/streetstatistics.git
    cd StreetStatistics
    ```

2. **Install required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### **Running the Application**

1. **Start the application**:
    To run the program, execute the `main.py` file.
    ```bash
    python main.py
    ```

2. **Using the Interface**:
    A **PyQt5** user interface will open, allowing you to:
    - Input your **YouTube Live Stream URL**.
    - Configure detection settings.
    - Start and stop real-time detection and analysis.

3. **Visualize Your Data**:
    After or during the running of the program, you can view visualized insights into the data you collected using the export button or viewing them on the sidebar. 

