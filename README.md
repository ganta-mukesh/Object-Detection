# 🧠 Real-Time Object Detection using YOLOv3 and OpenCV and detect any object any safety and other objects.

This project is a real-time object detection application using the YOLOv3 (You Only Look Once) model integrated with OpenCV. It captures live webcam video, detects multiple object classes from the COCO dataset, and displays bounding boxes with class labels and confidence scores.

## 🚀 Features

- 📷 **Real-Time Detection** using webcam
- 🎯 **YOLOv3 Deep Learning Model** (pre-trained on COCO)
- ✅ **Confidence Threshold Filtering**
- 🧠 **Non-Maximum Suppression** for accurate bounding boxes
- 📦 **80 Object Classes Supported**
- 🛑 **Exit detection with single key press ('q')**

## 🛠 Technologies Used

- Python
- OpenCV (`cv2.dnn`)
- YOLOv3 (`.cfg` and `.weights`)
- COCO class labels (`coco.names`)

## 📁 Files

- `main.py` – Core logic for video capture and object detection
- `yolov3.cfg` – YOLOv3 model configuration
- `yolov3.weights` – Pre-trained weights for YOLOv3
- `coco.names` – List of class names supported by the model

## 🧪 How It Works

1. Loads the YOLOv3 model and class names
2. Captures frames from your webcam
3. Converts each frame to a blob and passes it to the model
4. Retrieves detections and applies Non-Maximum Suppression
5. Draws bounding boxes and displays live results

## 📸 Sample Detection Output
person: 0.88
bottle: 0.62
cell phone: 0.74

Live bounding boxes with labels appear over the webcam feed.

## 🧾 Requirements

- Python 3.x
- OpenCV (`pip install opencv-python`)
- Pre-trained YOLOv3 model files (download from official YOLO site or darknet repo)

## ▶️ Run the App

bash:
python main.py
📌 Notes
Ensure yolov3.weights, yolov3.cfg, and coco.names are in the same directory as main.py.

You can modify the confidence threshold and NMS threshold in the script as needed.
📂 License
This project is open-source and available under the MIT License.![Screenshot (86)](https://github.com/user-attachments/assets/4096c432-f3ba-4093-ac39-7ca520d8bf6e)
![Object Detection1](https://github.com/user-attachments/assets/297a80bc-d53b-4f22-86b7-d6c331de653a)
![Object Detection](https://github.com/user-attachments/assets/ffbe3f90-af22-4d7a-80e4-5da41b0591a6)
