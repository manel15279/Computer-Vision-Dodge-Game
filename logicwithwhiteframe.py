import cv2
import numpy as np

class GameObject:
    def __init__(self, speed, x, y):
        self.pixel_character = np.ones((48, 48, 3), dtype=np.uint8) * 255
        self.speed = speed
        self.x = x
        self.y = y
        
    def draw(self, frame):
        self.pixel_character[12:18, 12:18, :] = [0, 0, 0]
        self.pixel_character[12:18, 30:36, :] = [0, 0, 0]
        self.pixel_character[18:24, 12:36, :] = [12, 171, 233]
        self.pixel_character[24:30, 12:36, :] = [20, 128, 203]
        self.pixel_character[30:36, 18:30, :] = [34, 35, 188]
        self.pixel_character[36:42, 18:30, :] = [48, 35, 169]
        frame[self.y:self.y + self.pixel_character.shape[0],
        self.x:self.x + self.pixel_character.shape[1], :] = self.pixel_character

    def update_position(self, direction, frame):
        frame[:] = 255
        if direction == 2424832:
            self.x -= self.speed
        elif direction == 2555904:
            self.x += self.speed
    

        # Wrap around if the object goes beyond the edges
        if self.x < 0:
            self.x = 200  # Assuming the width of the frame is 640, adjust accordingly
        elif self.x > 600:
            self.x = 400

        self.draw(frame)

def Object_Color_Detection(image, surfacemin, surfacemax, lo, hi):
    points = []
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(image, lo, hi)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    elements = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    elements = sorted(elements, key=lambda x: cv2.contourArea(x), reverse=True)
    for element in elements:
        if surfacemin < cv2.contourArea(element) < surfacemax:
            ((x, y), rayon) = cv2.minEnclosingCircle(element)
            points.append(np.array([int(x), int(y), int(rayon)]))
        else:
            break
    return image, mask, points


lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])

VideoCap = cv2.VideoCapture(0)

# Main loop for capturing and processing video frames
while True:
    # Read a frame from the video capture
    ret, frame = VideoCap.read()

    # Create a white image with the same size as the captured frame
    game_frame = np.ones_like(frame) * 0
    white_frame = np.ones_like(frame) * 255

    # Split the white frame into three parts
    video_capture_width = 250
    game_frame_width = 250
    score_speed_width = frame.shape[1] - video_capture_width - game_frame_width  # Smaller width for scoring

    # Split the white frame into three parts
    video_capture_part = white_frame[:, :video_capture_width, :]
    game_frame_part = white_frame[:, video_capture_width: video_capture_width + game_frame_width, :]
    score_speed_part = white_frame[:, -score_speed_width:, :]

    # Replace each part with the corresponding content
    video_capture_part[:, :, :] = frame[:, :video_capture_width, :]
    game_frame_part[:, :, :] = game_frame[:, :game_frame_width, :]

    game_object = GameObject(speed=48, x=125, y=200)
    game_object.draw(game_frame_part)
    

    # Concatenate the three parts to get the final white frame
    white_frame = np.concatenate((video_capture_part, game_frame_part, score_speed_part), axis=1)


    # Call the detect_inrange function to process the frame and detect objects
    image, mask, points = Object_Color_Detection(game_frame_part, 3000, 7000, lower_red, upper_red)

    # Add text to the white frame
    cv2.putText(white_frame, 'Score: 17.6', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(white_frame, 'Vitesse: 60', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # If points were detected, draw a circle at the detected location on the frame
    if len(points) > 0:
        cv2.circle(frame, (points[0][0], points[0][1]), 10, (0, 0, 255), 2)

    # Display the mask and the original frame with overlays
    if mask is not None:
        cv2.imshow('frame', white_frame)

    key = cv2.waitKeyEx(10)
    if key == ord('q'):
        break

    game_object.update_position(key, game_frame_part)
    # Concatenate the three parts to get the final white frame
    white_frame = np.concatenate((video_capture_part, game_frame_part, score_speed_part), axis=1)

# Release the video capture object and close all windows
VideoCap.release()
cv2.destroyAllWindows()
