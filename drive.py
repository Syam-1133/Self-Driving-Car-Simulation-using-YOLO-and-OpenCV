import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

def detect_lanes(image):

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

def calculate_curvature(lines, image_shape):

    if lines is None:
        return 0

    # Extract points from lines
    points = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        points.append((x1, y1))
        points.append((x2, y2))

    # Fit a second-order polynomial to the points
    y_points = np.array([p[1] for p in points])
    x_points = np.array([p[0] for p in points])
    fit = np.polyfit(y_points, x_points, 2)

    # Calculate the radius of curvature
    y_eval = np.max(y_points)
    A, B, C = fit
    curvature = ((1 + (2 * A * y_eval + B) ** 2) ** 1.5) / np.abs(2 * A)

    return curvature

def draw_lanes(image, lines):
    """Draw lanes and return lane mask."""
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

def detect_objects(image):
    """Detect objects using YOLO and return detected objects."""
    results = model.predict(image)
    return results

def make_decision(lanes, objects):
    """Make a decision (turn left, turn right, or go straight) based on detected lanes and objects."""
    decision = "Go Straight"

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

    if objects is not None:
        for result in objects:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls)
                confidence = float(box.conf)
                label = model.names[class_id]

                # Draw bounding box and label
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
                cv2.putText(image, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

def display_car_info(image, speed, location, curvature, offset):

    # Create a semi-transparent background for the text
    overlay = image.copy()
    cv2.rectangle(overlay, (20, 20), (500, 220), (0, 0, 0), -1)  # Background rectangle
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)  # Blend with the original image

    # Display text with larger font size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0  # Increased font size
    thickness = 2      # Increased thickness
    text_color = (0, 255, 0)  # Green text

    # Position and display each line of text
    y_start = 60
    y_increment = 50  # Space between lines

    cv2.putText(image, f"Speed: {speed} km/h", (30, y_start), font, font_scale, text_color, thickness)
    cv2.putText(image, f"Location: {location}", (30, y_start + y_increment), font, font_scale, text_color, thickness)
    cv2.putText(image, f"Curvature: {curvature:.2f} m", (30, y_start + 2 * y_increment), font, font_scale, text_color, thickness)
    cv2.putText(image, f"Offset: {offset:.2f} m", (30, y_start + 3 * y_increment), font, font_scale, text_color, thickness)

def draw_decision(image, decision):

    # Define text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2  # Increased font size
    thickness = 3      # Increased thickness
    bg_color = (0, 0, 0)  # Black background

    # Set text color based on decision
    if "Left" in decision:
        text_color = (255, 0, 0)  # Blue for Turn Left
        arrow_color = (255, 0, 0)  # Blue arrow
    elif "Right" in decision:
        text_color = (0, 255, 0)  # Green for Turn Right
        arrow_color = (0, 255, 0)  # Green arrow
    elif "Stop" in decision:
        text_color = (0, 0, 255)  # Red for Stop
        arrow_color = (0, 0, 255)  # Red arrow
    else:
        text_color = (255, 255, 255)  # White for other decisions
        arrow_color = (255, 255, 255)  # White arrow

    # Get the size of the text
    text_size = cv2.getTextSize(decision, font, font_scale, thickness)[0]

    # Define the position of the text (top-right corner)
    text_x = image.shape[1] - text_size[0] - 20  # 20 pixels from the right edge
    text_y = 50  # 50 pixels from the top

    # Draw a background rectangle for the text
    cv2.rectangle(
        image,
        (text_x - 10, text_y - text_size[1] - 10),  # Top-left corner of the box
        (text_x + text_size[0] + 10, text_y + 10),  # Bottom-right corner of the box
        bg_color,
        -1,
    )

    # Draw the decision text
    cv2.putText(
        image,
        decision,
        (text_x, text_y),
        font,
        font_scale,
        text_color,
        thickness,
        lineType=cv2.LINE_AA,
    )


    arrow_start = (image.shape[1] - 100, 100)
    if "Left" in decision:
        arrow_end = (arrow_start[0] - 50, arrow_start[1])
    elif "Right" in decision:
        arrow_end = (arrow_start[0] + 50, arrow_start[1])
    else:
        arrow_end = (arrow_start[0], arrow_start[1] - 50)

    cv2.arrowedLine(image, arrow_start, arrow_end, arrow_color, 5, tipLength=0.5)

    return image

def calculate_offset(lanes, image_shape):
    """Calculate the offset of the vehicle from the center of the lane."""
    if lanes is None:
        return 0

    # Calculate the center of the lane
    lane_center = np.mean([np.mean(line[0][::2]) for line in lanes])
    image_center = image_shape[1] / 2


    lane_width_pixels = image_shape[1]
    lane_width_meters = 3.7
    offset_pixels = lane_center - image_center
    offset_meters = (offset_pixels / lane_width_pixels) * lane_width_meters

    return offset_meters

# Main loop with visualization
cap = cv2.VideoCapture(" test.mp4")
car_speed = 60
car_location = "Lane 1"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Pipeline processing
    lanes, edges, masked_edges = detect_lanes(frame)
    curvature = calculate_curvature(lanes, frame.shape)
    offset = calculate_offset(lanes, frame.shape)
    processed_frame, lane_mask = draw_lanes(frame.copy(), lanes)
    objects = detect_objects(processed_frame)
    decision = make_decision(lanes, objects)

    # Draw UI elements
    draw_objects(processed_frame, objects)
    processed_frame = draw_decision(processed_frame, f"Decision: {decision}")
    display_car_info(processed_frame, car_speed, car_location, curvature, offset)

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