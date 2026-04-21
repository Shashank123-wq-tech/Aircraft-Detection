# Aircraft Detection Project Using Computer Vision
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Flask](https://img.shields.io/badge/Backend-Flask-black)
![YOLO](https://img.shields.io/badge/Model-YOLOv8-red)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-MIT-yellow)
This project focuses on developing an automated aircraft detection system using deep learning and computer vision techniques. 
The goal is to accurately identify and localize aircraft in images or video streams, 
enabling applications in surveillance, defense, and air traffic monitoring.
dataset link: https://www.kaggle.com/datasets/rookieengg/military-aircraft-detection-dataset-yolo-format
## Objectives:
Detect aircraft in images with high accuracy
Minimize false detections in complex backgrounds
Build a scalable model suitable for real-time applications
Handle variations in aircraft size, orientation, and lighting.
### Number of Aircrafts in Dataset: 
There are 43 different aircrafts :
names:
  0: A10
  1: A400M
  2: AG600
  3: AV8B
  4: B1
  5: B2
  6: B52
  7: Be200
  8: C130
  9: C17
  10: C2
  11: C5
  12: E2
  13: E7
  14: EF2000
  15: F117
  16: F14
  17: F15
  18: F16
  19: F18
  20: F22
  21: F35
  22: F4
  23: J20
  24: JAS39
  25: MQ9
  26: Mig31
  27: Mirage2000
  28: P3
  29: RQ4
  30: Rafale
  31: SR71
  32: Su34
  33: Su57
  34: Tornado
  35: Tu160
  36: Tu95
  37: U2
  38: US2
  39: V22
  40: Vulcan
  41: XB70
  42: YF23

## Class Distribution:
![Distribution](https://github.com/Shashank123-wq-tech/Aircraft-Detection/blob/main/Screenshot%202026-04-21%20095147.png)

### My whole Project demonstration: 
(https://youtu.be/szVpaUCH3Nk)

## Key Features:
- Real-time aircraft detection.
- Image & video input support.
- Fast inference using optimized models.
- Confidence score visualization.
- Deployable via web interface (FastAPI)

## Tech Stack

#### Frontend

HTML, CSS, JavaScript

### Backend

- FastAPI (Python)

- Machine Learning / AI

- TensorFlow / PyTorch
- OpenCV
- YOLO / CNN-based architecture.
### Deployment
Local Server.

## System Architecture
        +----------------------+
        |   User Interface     |
        |  (Web / Upload UI)   |
        +----------+-----------+
                   |
                   v
        +----------------------+
        |   Flask Backend      |
        |  (API + Processing)  |
        +----------+-----------+
                   |
                   v
        +----------------------+
        |  ML Model (YOLO/CNN) |
        |  Detection Engine    |
        +----------+-----------+
                   |
                   v
        +----------------------+
        |   Output Results     |
        | Bounding Boxes +     |
        | Confidence Scores    |
        +----------------------+

## Workflow:
- User uploads image/video
- Backend receives request via Flask API
- Image is preprocessed (resize, normalization)
- Model performs detection
- Bounding boxes + labels generated
- Results displayed on UI

## Sample Output:
#### Aircraft detected with bounding boxes and Confidence Score:
![Output](https://github.com/Shashank123-wq-tech/Aircraft-Detection/blob/main/B52%20Aircraft.png)
![Output](https://github.com/Shashank123-wq-tech/Aircraft-Detection/blob/main/US2.png)
![Output](https://github.com/Shashank123-wq-tech/Aircraft-Detection/blob/main/Output.png)

##  Use Cases
- Defense surveillance systems
- Airspace monitoring
- Military intelligence
- Autonomous drone systems
## Future Improvements
- Real-time drone feed integration.
- Radar + AI fusion system.
- Improved model accuracy using larger datasets.
- Mobile app integration.

##  Author:
Shashank Dixit
AI/ML Enthusiat 
