import cv2
import numpy as np

def opening(image, kernel):
    # Erosion followed by dilation
    eroded = cv2.erode(image, kernel, iterations=1)
    opened = cv2.dilate(eroded, kernel, iterations=1)
    return opened

def closing(image, kernel):
    # Dilation followed by erosion
    dilated = cv2.dilate(image, kernel, iterations=1)
    closed = cv2.erode(dilated, kernel, iterations=1)
    return closed

# Example usage
image_path = "univerNB.jpg"  # Replace with the path to your image
input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Define the kernel (structuring element)
kernel_size = 5
kernel = np.ones((kernel_size, kernel_size), np.uint8)

# Perform opening and closing operations
opened_image = opening(input_image, kernel)
closed_image = closing(input_image, kernel)

# Display the original, opened, and closed images
cv2.imshow('Original Image', input_image)
cv2.imshow('Opened Image', opened_image)
cv2.imshow('Closed Image', closed_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
