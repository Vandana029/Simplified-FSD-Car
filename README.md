# OVERVIEW

This project simulates a **Simplified Self-Driving Car** that processes video frames (25 FPS) and performs **three key tasks**:
1. **Lane Segmentation** – Detecting lane boundaries using a YOLO segmentation model.
2. **Object Detection** – Detecting cars, pedestrians, etc., using a YOLO segmentation/detection model.
3. **Steering Angle Prediction** – Predicting the steering wheel rotation using a custom CNN regression model (in TensorFlow 1.x).

All these are integrated in real time using OpenCV to display:
- Original frame
- Segmented frame (lanes + objects)
- Steering wheel rotation visualization

# TECH STACK
## Programming Languages
- **Python 3.10+:** Core language for model training, inference, and simulation

## Deep Learning & Machine Learning
- **TensorFlow 1.x:** Used for regression model predicting steering angle.
- **Ultralytics YOLOv11:** Lane segmentation and object detection models.
- **CNNs:** For feature extraction from images.

## Computer Vision & Image Processing
- **OpenCV**: For image preprocessing, resizing, and visualization (frames, steering wheel rotation).
- **PIL (Python Imaging Library)**:  Loading and handling images.

## Data Handling
- **NumPy:** Numerical operations on image arrays
- **scipy:** Support for additional mathematical functions (in model.py)

## APIs
- **Roboflow API:** Dataset download and management for lane segmentation.

# RESULTS
[Steering Angle Prediction Video](assets/steering_angle_prediction.mp4)
[Object and Lane Prediction Video](assets/obj_n_lane_det.mp4)
[FSD inference Video](assets/fsd_inference.mp4)


# 📂 FOLDER STRUCTURE SUMMARY
| Folder/File                  | Purpose                                                                         |
| ---------------------------- | ------------------------------------------------------------------------------- |
| `data/`                      | Contains driving dataset (`45k+ images`, `data.txt`, steering wheel image).     |
| `model_training/`            | Training scripts for lane detection, object detection, and steering prediction. |
| `saved_models/`              | Trained weights for segmentation, object detection, and steering regression.    |
| `src/inference/`             | Python scripts that **run the models** and simulate inference.                  |
| `src/models/model.py`        | CNN architecture used for steering angle prediction.                            |
| `requirements.txt`           | Dependencies for setup.                                                         |
| `setup.py`, `pyproject.toml` | Installation and project metadata.                                              |
| `README.md`                  | Documentation for GitHub.                                                       |

# CODE EXPLANATION
## 1. **model_training/training_lane_detection.ipynb**
    - it’s used to **train YOLOv11 segmentation** for lane detection. Uses Ultralytics YOLOv11 with segmentation head (`yolo11n-seg.pt`).
    - Downloads dataset from Roboflow via API.
    - Trains with:
    ```
    !yolo task=segment mode=train model=yolo11m-seg.pt data={dataset.location}/data.yaml epochs=20 imgsz=640 plots=True
    ```
    - Saves best weights to `/runs/segment/train*/weights/best.pt`.

## 2. **src/models/model.py — Steering Angle Model (TensorFlow 1.x)**
This is a **custom CNN regression model** inspired by NVIDIA’s End-to-End Self-Driving model.
Model Flow:
- Input: `(66, 200, 3)` RGB image (cropped bottom road region).
- Output: **Predicted steering angle** (in radians, scaled via `atan`).
- Layers:
| Layer      | Type                           | Details                                    |
| ---------- | ------------------------------ | ------------------------------------------ |
| Conv1      | 5x5 conv, 24 filters, stride 2 | Extracts low-level features.               |
| Conv2      | 5x5 conv, 36 filters, stride 2 | Mid-level features.                        |
| Conv3      | 5x5 conv, 48 filters, stride 2 | Road geometry patterns.                    |
| Conv4      | 3x3 conv, 64 filters           | Higher-level lane edges.                   |
| Conv5      | 3x3 conv, 64 filters           | Refines features.                          |
| FC1        | 1152 → 1164                    | Dense layer                                |
| FC2        | 1164 → 100                     |                                            |
| FC3        | 100 → 50                       |                                            |
| FC4        | 50 → 10                        |                                            |
| FC5        | 10 → 1                         | Output (steering angle)                    |
| Activation | atan * 2                       | Ensures stable angle range (smooth output) |

- **Training goal**: Minimize mean squared error (MSE) between predicted vs. actual steering angle.

## 3. **src/inference/run_segmentation_obj_det.py**
Used to **visualize lane segmentation + object detection results** (without steering prediction).
**Main Class**: `SegmentationVisualizer`
- Loads **two YOLO models**:
    - `model_1`: Lane segmentation model.
    - `model_2`: Object detection/segmentation model.
- **Process**:
    1. **Predict masks and boxes** for both models.
    2. **Colorize masks**:
        - Lanes: Light green.
        - Objects: Unique HSV-based colors.
    3. **Overlay results** with transparency (`alpha` blending).
    4. **Display in OpenCV window** with frame-by-frame visualization.
- **Usage:**
```
input_folder = 'data/driving_dataset'
display_images_with_segmentation(input_folder, display_time=100)
```

## 4. **src/inference/run_steering_angle_prediction.py**
Used to test **only the steering prediction** model (no segmentation).
**Workflow:**
1. Loads the trained TensorFlow model (`model.ckpt`)
2. For each frame:
    - Crops bottom 150 pixels.
    - Resizes to (200×66).
    - Normalizes (0–1).
    - Predicts steering angle (degrees).
3. Smooths predictions to prevent jittering:
```
self.smoothed_angle += 0.2 * pow(abs(predicted_angle - self.smoothed_angle), 2/3) * ...
```
4. Displays:
    - Driving frame.
    - Steering wheel image rotated by the predicted angle.

## 5. **src/inference/run_fsd_inference.py**
This is the **final integrated simulation script**, combines **segmentation + object detection + steering angle prediction**.
**Components:**
- `SteeringAnglePredictor`: Loads and predicts from TensorFlow model.
- `ImageSegmentation`: Uses YOLO models for lane + object overlays.
- `SelfDrivingCarSimulator`: Runs synchronized inference + visualization.

**Steps**:
1. Read each frame from `data/driving_dataset`.
2. Run **YOLO lane & object detection** asynchronously using `ThreadPoolExecutor`.
3. Run **steering prediction** simultaneously.
4. Blend results into a single view:
    - Original frame
    - Segmented (lanes + objects)
    - Rotated steering wheel visualization
5. Display real-time simulation at 30 FPS.

**Final Output Windows**:
- Original Frame
- Segmented Frame
- Steering Wheel

**Run Example:**
```
python src/inference/run_fsd_inference.py
```

# HOW COMPONENTS CONNECT
                +-----------------------------+
                |    data/driving_dataset     |
                +-----------------------------+
                             |
                             ▼
         +--------------------------------------------+
         |  run_fsd_inference.py                      |
         |  (Full Self-Driving Simulation)            |
         +--------------------------------------------+
           |                   |                  |
           ▼                   ▼                  ▼
  SteeringAnglePredictor   ImageSegmentation   SelfDrivingCarSimulator
           |                   |
           ▼                   ▼
   model.py (TensorFlow)  YOLOv11 Seg + Det Models

# 💡 KEY IDEAS 
- **Parallel Inference**:
ThreadPoolExecutor enables running segmentation and steering prediction simultaneously for faster frame rates.

- **Angle Smoothing**:
Prevents jerky steering visualization.

- **Transfer Learning in YOLO**:
Pretrained YOLOv11 segmentation models are fine-tuned on a **lane dataset** via Roboflow.

- **Real-time Visualization**:
Using OpenCV `imshow()` with adjustable FPS timing (30 FPS default).

- **Cross-Platform Support**:
Works on Windows (minor CLI command differences handled).

