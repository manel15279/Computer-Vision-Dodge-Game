# Color-Based Object Detection & Dodge Game Project

## Overview

A university project consisting of two parts: the first part focuses on implementing various image processing filters and color-based object detection algorithms using Python and OpenCV. The second part involves the development of a dodge game - Chicken Escape 🐔🦊, enhanced with real-time object detection for player control.

<div style="width: 100%; display: flex;">
  <img src="interfaceFinal/screenshots/full_interface.png"/>
</div>

## Part 1: Filters and Object Detection 📸

### Filters Implemented
- Binarization and Thresholding
- Smoothing Filters: Mean, Median, Gaussian
- Edge Detection Filters: Laplacian, Gradient
- Morphological Filters: Erosion, Dilation, Closing, Opening
- Custom Filters: Prewitt, Sobel

### Object Detection
- Developed the "Object_Color_Detection" function for detecting objects based on color.
- Proposed improvements for the object detection function using the Kalman Filter for predicting the position of objects in real-time.
- Implemented two functionalities using object color detection: "Invisibility Cloak" and "Green Screen".
- Developed a graphical user interface (GUI) for applying filters and object detection.

## Part 2: Dodge Game - Chicken Escape 🐔🦊

The Dodge Game is an implementation of computer vision principles in the gaming domain. It offers an immersive gaming experience where players control a chicken 🐔 to navigate through obstacles 🌳 and avoid fox enemies 🦊 using real-time object detection techniques.

### Features
- **Object Detection Control**: Players can control the movement of the character using real-world objects, specifically by manipulating a colored object detected through the camera. The default color is green.
- **Dynamic Obstacle Avoidance**: The game environment presents dynamic obstacles that the player must navigate through by moving the character horizontally.
- **Scoring System**: Players accumulate points based on their performance, with increasing difficulty levels as the game progresses.
- **Enhancements**: Two additional enhancements, such as score tracking or speed variations, contribute to the gameplay experience.

### Gameplay
- **Objective**: Navigate the chicken character through obstacles (foxes and trees) by manipulating a colored object detected through the camera.
- **Controls**:
    - Keyboard Controls:
        - "SPACE" bar: Start or restart the game.
        - "Q": Move the character left.
        - "D": Move the character right.
        - "E": Quit the game.
        - "2": Horizontal movement.
        - "3": Horizontal and vertical movement (Kalman Filter).
    - Object Detection: Control the character's movement by shifting the position of a colored object detected through the camera.

## Installation
To run the project, follow these steps:

1. Ensure Python is installed on your system.
2. Install the necessary libraries: `pip install -r requirements.txt`.
3. Run the project: `python IHM.py`.
4. Interface Buttons:
    - Object Detection: Start detecting and tracking the object using your webcam. (default color is green)
    - Invisibility: Activate the invisibility mode to make the object disappear against a background.
    - Fond Vert: Apply the green screen effect to replace the background with a custom image.
    - Stop: Stop the object detection or background effect.
    - Clean: Clear the canvas.
    - Game: Run the dodge game window.
    - Moyen: Apply the mean filter to the displayed image.
    - Median: Apply the median filter to the displayed image.
    - Gradient: Apply the gradient filter to the displayed image.
    - Gaussien: Apply the Gaussian filter to the displayed image.
    - Laplacien: Apply the Laplacian filter to the displayed image.
    - Erode/Dilate: Apply morphological operations (erosion and dilation) to the displayed image.
    - Closing/Opening: Apply morphological operations (closing and opening) to the displayed image.
    - Prewitt(H/V): Apply the Prewitt filter (horizontal or vertical) to the displayed image.
    - Sobel: Apply the Sobel filter to the displayed image.
    - Threshold and Type Adjustments: Adjust the threshold and type for thresholding using the scales provided.
