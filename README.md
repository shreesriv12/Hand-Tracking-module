# 🖐️ Real-Time Hand Tracking using OpenCV & MediaPipe

## Overview
This project implements a real-time **hand tracking** system using **OpenCV** and **MediaPipe**. It detects hands in a video feed, extracts hand landmarks, and displays tracking information, including FPS.

https://github.com/user-attachments/assets/59ae3c36-78a5-47be-a898-5acaed00d25a


## Features
✅ **Real-time hand detection & tracking**  
✅ **Extracts hand landmark positions (x, y)**  
✅ **Draws hand landmarks & connections**  
✅ **Displays FPS for performance monitoring**  
✅ **Robust error handling for camera detection**  

## Installation
Make sure you have Python installed, then install the required dependencies:

```bash
pip install opencv-python mediapipe numpy
```

## Usage
Run the following command to start the hand tracking system:

```bash
python3 HandTrackingModel.py
```

Press `q` to exit the program.

## File Structure
```
/hand-tracking-project
│── HandTrachingModel.py  # Main hand tracking script
│── README.md         # Project documentation
```

## Code Breakdown
### 1️⃣ Import Dependencies
```python
import cv2
import mediapipe as mp
import time
import numpy as np
```


## Future Enhancements 🚀
- Implement **gesture recognition** for control interactions.
- Add **finger counting & tracking**.
- Improve **performance optimizations** for mobile devices.

## Author
👨‍💻 Developed by **[Shreeya Srivastava]**




