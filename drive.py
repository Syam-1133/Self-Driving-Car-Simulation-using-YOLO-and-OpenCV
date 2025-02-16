import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")  # Replace with your YOLO model path

def detect_lanes(image):
    """Detect lanes and return intermediate processing steps"""
    # Preprocess image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Region of Interest (ROI)
    height, width = edges.shape
    roi_vertices = np.array([[(0, height), (width // 2, height // 2), (width, height)]], dtype=np.int32)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Detect lines
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=50)

    return lines, edges, masked_edges


def draw_lanes(image, lines):
    """Draw lanes and return lane mask"""
    lane_mask = np.zeros_like(image)
    if lines is not None:
        lane_points = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            lane_points.append((x1, y1))
            lane_points.append((x2, y2))
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if len(lane_points) >= 10:
            lane_points = np.array(lane_points, dtype=np.int32)
            hull = cv2.convexHull(lane_points)
            cv2.fillPoly(lane_mask, [hull], (0, 255, 0))

    return cv2.addWeighted(image, 1, lane_mask, 0.3, 0), lane_mask


# Define the missing detect_objects function
def detect_objects(image):
    """Detect objects using YOLO and return detected objects"""
    results = model.predict(image)  # Detect objects
    return results


# Rest of the functions remain the same (make_decision, draw_objects, etc.)

def make_decision(lanes, objects):
    """Make a decision (turn left, turn right, or go straight) based on detected lanes and objects."""
    decision = "Go Straight"  # Default decision

    # Lane-based decision
    if lanes is not None:
        left_lane_count = 0
        right_lane_count = 0
        for line in lanes:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
            if slope < -0.5:  # Left lane
                left_lane_count += 1
            elif slope > 0.5:  # Right lane
                right_lane_count += 1

        if left_lane_count > right_lane_count:
            decision = "Turn Left"
        elif right_lane_count > left_lane_count:
            decision = "Turn Right"

    # Object-based decision
    if objects is not None:
        for result in objects:
            for box in result.boxes:
                class_id = int(box.cls)
                confidence = float(box.conf)
                label = model.names[class_id]

                # Example: Stop if a pedestrian is detected
                if label == "person" and confidence > 0.5:
                    decision = "Stop"
                # Example: Turn left if a left-turn sign is detected
                elif label == "stop sign" and confidence > 0.5:
                    decision = "Stop"
                # Example: Slow down if a car is detected in front
                elif label == "car" and confidence > 0.5:
                    decision = "Slow Down"
                # Example: Avoid if an obstacle is detected
                elif label == "obstacle" and confidence > 0.5:
                    decision = "Avoid"

    return decision


def draw_objects(image, objects):
    """Draw detected objects on the image."""
    if objects is not None:
        for result in objects:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls)
                confidence = float(box.conf)
                label = model.names[class_id]

                # Draw bounding box and label
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
                cv2.putText(image, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


def display_car_info(image, speed, location):
    """Display car speed and location."""
    cv2.putText(image, f"Speed: {speed} km/h", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(image, f"Location: {location}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


def draw_square_arrow(image, decision):
    """Draw an arrow based on the decision."""
    height, width = image.shape[:2]
    if decision == "Turn Left":
        cv2.arrowedLine(image, (width // 2, height - 50), (width // 4, height - 50), (0, 0, 255), 5, tipLength=0.5)
    elif decision == "Turn Right":
        cv2.arrowedLine(image, (width // 2, height - 50), (3 * width // 4, height - 50), (0, 0, 255), 5, tipLength=0.5)
    elif decision == "Go Straight":
        cv2.arrowedLine(image, (width // 2, height - 50), (width // 2, height // 2), (0, 0, 255), 5, tipLength=0.5)


# Main loop with visualization
cap = cv2.VideoCapture(" test.mp4")
car_speed = 60  # Example speed in km/h
car_location = "Lane 1"  # Example location

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Pipeline processing
    lanes, edges, masked_edges = detect_lanes(frame)
    processed_frame, lane_mask = draw_lanes(frame.copy(), lanes)
    objects = detect_objects(processed_frame)
    decision = make_decision(lanes, objects)

    # Draw UI elements
    draw_objects(processed_frame, objects)
    draw_square_arrow(processed_frame, decision)
    display_car_info(processed_frame, car_speed, car_location)
    cv2.putText(processed_frame, f"Decision: {decision}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Create visualization grid
    h, w = processed_frame.shape[:2]
    size = (w // 2, h // 2)

    # Resize components
    main_view = cv2.resize(processed_frame, size)
    edges_view = cv2.cvtColor(cv2.resize(edges, size), cv2.COLOR_GRAY2BGR)
    masked_view = cv2.cvtColor(cv2.resize(masked_edges, size), cv2.COLOR_GRAY2BGR)
    lane_view = cv2.resize(lane_mask, size)

    # Build grid
    top_row = np.hstack((main_view, edges_view))
    bottom_row = np.hstack((masked_view, lane_view))
    grid = np.vstack((top_row, bottom_row))

    # Display
    cv2.imshow("Self Driving Car", grid)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


