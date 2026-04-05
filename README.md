#AlertVision  – Driver Drowsiness Detection

A real-time driver monitoring system built with **OpenCV and deep learning (MobileNet)** that identifies signs of drowsiness and triggers an alert to improve road safety.
---

## Key Features

* Live webcam-based monitoring
* Face detection with focused eye-region extraction
* Deep learning model for classification (Awake / Sleepy)
* Time-based detection to avoid false alarms
* Audio alert system for prolonged drowsiness

---

## Working Principle

1. Capture frames continuously from the webcam
2. Detect the face using a Haar Cascade classifier
3. Estimate and extract the eye region from the face
4. Resize the image to 224×224 and normalize pixel values
5. Pass the processed image to the trained model
6. If the “sleepy” state persists for more than 5 seconds, an alert is triggered

---

##  Model Overview

* Architecture: MobileNet (pretrained on ImageNet)
* Task: Binary classification
* Output classes:

  * 0 → Awake
  * 1 → Sleepy

---

##  How to Run

### Train the model

```bash
python src/train_model.py
```

### Start detection

```bash
python src/detect_drowsiness.py
```

Press **q** to close the application.

---

##  Important Notes

* Performance is better in well-lit conditions
* Uses `winsound`, so it is compatible with Windows systems

---

## Output Indicators

* 🟢 Green → Driver is awake
* 🔴 Red → Drowsiness alert triggered
