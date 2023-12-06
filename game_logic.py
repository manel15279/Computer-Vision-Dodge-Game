import cv2
import numpy as np
import random

class Player:
    def __init__(self, width, height):
        self.w = 50
        self.h = 50
        self.x = width // 2
        self.y = height - self.h

    def move_left(self):
        self.x -= 15

    def move_right(self):
        self.x += 15

    def display(self, img):
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), (0, 0, 255), -1)

class Enemy:
    def __init__(self, width):
        self.w = 50
        self.h = 50
        self.x = random.randint(0, width - self.w)
        self.y = 0 - self.h
        self.speed = 10

    def collision(self, obj):
        if obj.x < self.x < obj.x + obj.w and obj.y < self.y < obj.y + obj.h:
            return True
        return False

    def out_of_bounds(self, height):
        if self.y > height:
            return True
        return False

    def display(self, img):
        self.y += self.speed
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), (0, 0, 0), -1)

# Initialize game parameters
width, height = 640, 480
player = Player(width, height)
enemies = []
game_mode = False
score = 0

VideoCap = cv2.VideoCapture(0)

while True:

    ret, frame = VideoCap.read()
    print(frame.shape)

    video_capture_width = 320
    game_frame_width = 320

    img = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background


    if game_mode:
        cv2.putText(img, "Score: {}".format(score), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        if random.randint(0, 30) == 0:
            enemies.append(Enemy(game_frame_width))

        player.display(img)
        for enemy in enemies:
            enemy.display(img)
            if enemy.collision(player):
                enemies = []
                game_mode = False
            elif enemy.out_of_bounds(height):
                enemies.remove(enemy)
                score += 1
    else:
        img[:, :] = [0, 0, 255]  # Red background
        
    img_capture = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background for video capture

    # Replace each part with the corresponding content
    img_capture[:, :video_capture_width, :] = frame[:, :video_capture_width, :]
    img_capture[:, video_capture_width: video_capture_width + game_frame_width, :] = img[:, :game_frame_width, :]

    cv2.imshow('Split Interface', img_capture)

    key = cv2.waitKey(10)
    if key == ord('q'):  
        break
    elif key == ord(' '):  # Space key to start/restart the game
        score = 0
        game_mode = True
    elif key == ord('a'):
        player.move_left()
    elif key == ord('d'):
        player.move_right()

cv2.destroyAllWindows()
