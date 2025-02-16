# 🚗 Self-Driving Car Simulation using YOLO and OpenCV 🛣️

This project is a simulation of a self-driving car system that uses **YOLO (You Only Look Once)** for object detection and **OpenCV** for lane detection and decision-making. The system processes video input, detects lanes and objects, and makes driving decisions (e.g., turn left, turn right, stop) based on the detected information.


## 🎯 Introduction
This project simulates a self-driving car system by combining **computer vision** and **machine learning**. It uses:
- **YOLOv8** for real-time object detection (e.g., cars, pedestrians, traffic signs).
- **OpenCV** for lane detection, edge detection, and image processing.
- A decision-making algorithm to determine the car's actions based on detected lanes and objects.

The system is designed to process video input, detect lanes and objects, and display the results in a user-friendly interface.

---

## ✨ Features
- **🚦 Lane Detection**: Detects lanes using edge detection and Hough Transform.
- **📦 Object Detection**: Detects objects like cars, pedestrians, and traffic signs using YOLOv8.
- **🤖 Decision Making**: Makes driving decisions (e.g., turn left, turn right, stop) based on detected lanes and objects.
- **🖼️ Visualization**: Displays intermediate processing steps (e.g., edges, masked edges, lane mask) and the final output with bounding boxes and decisions.
- **📊 User Interface**: Shows car speed, location, and driving decisions on the screen.

---

## 🛠️ Technologies Used
- **Python**: The primary programming language.
- **OpenCV**: For image processing, lane detection, and visualization.
- **YOLOv8**: For object detection.
- **NumPy**: For numerical computations and array manipulations.

---

## 🧠 How It Works
1. **📹 Input Video**: The system takes a video file as input.
2. **🛣️ Lane Detection**:
   - Converts the frame to grayscale and applies Gaussian blur.
   - Uses Canny edge detection to detect edges.
   - Applies a region of interest (ROI) mask to focus on the road.
   - Detects lanes using Hough Transform.
3. **📦 Object Detection**:
   - Uses YOLOv8 to detect objects in the frame.
   - Draws bounding boxes and labels for detected objects.
4. **🤖 Decision Making**:
   - Analyzes detected lanes and objects to make driving decisions.
   - Decisions include "Turn Left", "Turn Right", "Go Straight", "Stop", "Slow Down", and "Avoid".
5. **🖼️ Visualization**:
   - Displays the processed frame with lanes, objects, and decisions.
   - Shows intermediate steps (edges, masked edges, lane mask) in a grid layout.

---

## 🛠️ Installation
1. Clone the repository:
   ```bash
   https://github.com/Syam-1133/Self-Driving-Car-Simulation-using-YOLO-and-OpenCV
   
2. Run the python script
    ```bash
   python drive.py

