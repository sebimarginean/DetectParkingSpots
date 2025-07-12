# Parking Space Detection System

This system detects and visualizes available parking spaces from side camera footage of a moving vehicle.

## Overview

The system uses traditional computer vision techniques to detect potential parking spaces from a sequence of images captured by a side-mounted camera on a moving vehicle. It then determines whether each space is occupied or available and generates an annotated video showing the results.

## Features

- Detects parking spaces using edge detection and line fitting
- Classifies spaces as occupied or available
- Applies temporal smoothing to reduce flickering
- Displays confidence scores for each detection
- Generates an annotated video output

## Requirements

- Python 3.6+
- OpenCV
- NumPy
- Matplotlib
- scikit-image

Install the requirements using:

```bash
pip install -r requirements.txt
```

## Usage

1. Place your image sequence in the `dataset` folder
2. Run the detection script:

```bash
python detect_parking_spaces.py
```

3. The output video will be saved as `parking_output.mp4` in the current directory

## Technical Details

The system works in the following steps:

1. **Preprocessing**: Converts images to grayscale, applies Gaussian blur, and Canny edge detection
2. **Line Detection**: Uses Hough transform to detect lines in the edge image
3. **Parking Space Detection**: Groups horizontal lines to form potential parking spaces
4. **Occupancy Classification**: Analyzes texture and intensity to determine if a space is occupied
5. **Temporal Smoothing**: Tracks spaces across frames to reduce flickering
6. **Visualization**: Overlays bounding boxes and labels on the original images

## Camera Parameters

The system uses the following camera extrinsic parameters:

- Position X: 1910.04 mm
- Position Y: -1053.22 mm
- Position Z: 1180.47 mm
- Rotation X: -0.0078 rad
- Rotation Y: 0.7236 rad
- Rotation Z: -1.5759 rad

These parameters define the translation and rotation of the side camera with respect to the vehicle coordinate system.

This project was developed in collaboration with Robert Man.
