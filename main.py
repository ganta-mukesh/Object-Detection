import cv2
import numpy as np

# Load the pre-trained model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load the COCO class labels the model was trained on
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Access the camera
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Capture frame-by-frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    
    height, width, _ = frame.shape

    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)

    # Set the input for the model
    net.setInput(blob)

    # Run the model
    outputs = net.forward(net.getUnconnectedOutLayersNames())

    # Initialize lists for detected objects
    boxes = []
    confidences = []
    class_ids = []

    # Iterate over the outputs
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Get the bounding box coordinates
                x, y, w, h = detection[0:4] * np.array([width, height, width, height])
                x = int(x - w / 2)
                y = int(y - h / 2)
                
                # Store the bounding boxes, confidences, and class IDs
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    # Iterate over the remaining boxes after NMS
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"

            # Draw a rectangle and label on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with object detection
    cv2.imshow("Object Detection", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()