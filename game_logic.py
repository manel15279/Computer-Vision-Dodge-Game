import cv2
import numpy as np
import random
import time

class Player:
    def __init__(self, width, height):
        self.w = 50
        self.h = 50
        self.x = width // 2
        self.y = height - self.h

    def move_left(self):
        self.x = max(0, self.x - 20)  # Ensure the player stays within the left border

    def move_right(self, width):
        self.x = min(width - self.w, self.x + 20)  # Ensure the player stays within the right border


    def display(self, img):
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), (0, 0, 255), -1)

class Enemy:
    def __init__(self, width, speed):
        self.w = 50
        self.h = 50
        self.x = random.randint(25, width - 25)
        self.y = 0 - self.h
        self.speed = speed

    def collision(self, obj):
        x_overlap = max(0, min(obj.x + obj.w, self.x + self.w) - max(obj.x, self.x))
        y_overlap = max(0, min(obj.y + obj.h, self.y + self.h) - max(obj.y, self.y))
        overlap_area = x_overlap * y_overlap
        return overlap_area > 0

    def out_of_bounds(self, height):
        if self.y > height:
            return True
        return False

    def display(self, img):
        self.y += self.speed
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), (0, 0, 0), -1)

class BorderEnemy:
    def __init__(self, x, speed):
        self.w = 25
        self.h = 25
        self.x = x
        self.y = random.randint(-height, 0)
        self.speed = speed

    def collision(self, obj):
        x_overlap = max(0, min(obj.x + obj.w, self.x + self.w) - max(obj.x, self.x))
        y_overlap = max(0, min(obj.y + obj.h, self.y + self.h) - max(obj.y, self.y))
        overlap_area = x_overlap * y_overlap
        return overlap_area > 0

    def out_of_bounds(self, height):
        if self.y > height:
            return True
        return False

    def display(self, img):
        self.y += self.speed
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), (0, 0, 0), -1)

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

def resize(img):
    target_width = 320
    target_height = 480
    img_resized = np.zeros((target_height, target_width, 3), img.dtype)

    for y in range(target_height):
        for x in range(target_width):
            img_resized[y, x, :] = img[int(y * (img.shape[0] / target_height)),
                                       int(x * (img.shape[1] / target_width)), :]

    return img_resized


# Initialize game parameters
width, height = 640, 480
player = Player(290, height)
enemies = []
game_mode = False
score = 0

VideoCap = cv2.VideoCapture(0)

lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])
speed = 5

last_enemy_time = 0  # Variable to track the time when the last enemy was displayed
enemy_delay = 0.6

nbr_enemies = 30

while True:

    ret, frame = VideoCap.read()
    frame = resize(frame)

    video_capture_width = 320
    game_frame_width = 320

    img = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background

    if game_mode:
        cv2.putText(img, "Score: {}".format(score), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(img, "Speed: {}".format(speed), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        if random.randint(0, nbr_enemies) == 0 and time.time() - last_enemy_time > enemy_delay:
            border_enemy_left = BorderEnemy(0, speed)
            border_enemy_right = BorderEnemy(game_frame_width - border_enemy_left.w, speed)
            enemies.append(Enemy(game_frame_width - border_enemy_left.w - border_enemy_right.w, speed))
            enemies.append(border_enemy_left)
            enemies.append(border_enemy_right)
            last_enemy_time = time.time()  # Update the last enemy display time

        # Update enemy positions first
        for enemy in enemies:
            enemy.display(img)

        # Check for collisions or out-of-bounds after updating positions
        for enemy in enemies:
            if enemy.collision(player):
                enemies = []
                game_mode = False
            elif enemy.out_of_bounds(height):
                enemies.remove(enemy)
                score += 1
                if score % 10 == 0:
                    speed += 1
                    if nbr_enemies > 5:
                        nbr_enemies -= 2
                    for enemy in enemies:
                        enemy.speed = speed

        player.display(img)

    else:
        img[:, :] = [0, 0, 255]  # Red background

    img_capture = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background for video capture

    # Replace each part with the corresponding content
    img_capture[:, :video_capture_width, :] = frame[:, :video_capture_width, :]
    img_capture[:, video_capture_width: video_capture_width + game_frame_width, :] = img[:, :game_frame_width, :]
    

    image, mask, points = Object_Color_Detection(img_capture[:, :video_capture_width, :], 3000, 7000, lower_red, upper_red)

    cv2.imshow('Game Interface', img_capture)

    key = cv2.waitKey(10)
    if key == ord('q'):
        break
    elif key == ord(' '):  # Space key to start/restart the game
        score = 0
        speed = 5
        game_mode = True
        nbr_enemies = 30
    elif key == ord('a'):
        player.move_left()
    elif key == ord('d'):
        player.move_right(game_frame_width)

cv2.destroyAllWindows()