import cv2
import numpy as np
import random
import time

##optimisé
#vert
#boule
#taille de frame en entrée

import cv2
import numpy as np
import time

lo = np.array([60,50,50])
hi = np.array([180,255,255])
def erode(mask,kernel):
    ym,xm=kernel.shape
    yi,xi=mask.shape
    m=xm//2
    mask2=mask.copy()
    for y in range(yi):
        for x in range(xi):
            if mask[y,x]==255:
                if  ( y<m or y>(yi-1-m) or x<m or x>(xi-1-m)):
                    mask2[y,x]=0
                else:
                    v=mask[y-m:y+m+1,x-m:x+m+1] 
                    for h in range(0,ym):
                        for w in range(0,xm): 
                            if(v[h,w]<kernel[h,w]):
                                mask2[y,x]=0
                                break
                        if(mask2[y,x]==0): 
                            break
    return mask2

# def erode(img,kernel):
    ym,xm=kernel.shape
    xi,yi=img.shape
    m=xm//2
    mask2=np.zeros(img.shape,img.dtype)
    for y in range(xi):
        for x in range(yi):   
             if not(y<m or y>(xi-1-m) or x<m or x>(yi-1-m)):
                v=img[y-m:y+m+1,x-m:x+m+1] 
                b=False
                for h in range(ym):
                     for w in range(xm): 
                        if(v[h,w]<kernel[h,w]):
                             b=True
                             break
                     if(b): 
                         break
                if(not b): 
                    mask2[y,x]=255
    return mask2
def inRange(img,lo,hi):
    mask=np.zeros((img.shape[0],img.shape[1]))
    for y in range(img.shape[0]):
         for x in range(img.shape[1]):
              if(img[y,x,0]<=hi[0] and img[y,x,0]>=lo[0] and img[y,x,2]<=hi[2] and img[y,x,2]>=lo[2] and img[y,x,1]<=hi[1] and img[y,x,1]>=lo[1] ):
                    mask[y,x]=255
    return mask
def center(img):
    b=True
    c=True
    premiery=0
    derniery=img.shape[0]
    premierx=0
    dernierx=img.shape[1]
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if(b and img[y,x]==255):
                b=False
                premiery=y
            if( c and img[img.shape[0]-y-1,x]==255):
                c=False
                derniery=img.shape[0]-y-1
            if(not b and not c):
                break
        if(not b and not c):
            break    
    b=True
    c=True
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            if(img[y,x]==255 and b):
                b=False
                premierx=x
            if(img[y,img.shape[1]-x-1]==255 and c):
                c=False
                dernierx=img.shape[1]-x-1
            if(not b and not c):
                break
        if(not b and not c):
            break

    x=((dernierx-premierx)/2)+ premierx
    y=((derniery-premiery)/2)+ premiery
    return (int(x),int(y))
def detect_inrange(image):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    mask = inRange(image,lo,hi)
    mask = erode(mask,kernel=np.ones((5,5)))
    return mask
def resize(img):
     img2=np.zeros(((int(img.shape[0]/2.5))+1,(int(img.shape[1]//2.5))+1,3),img.dtype)
     for y in range(0,int(img.shape[0]/2.5)):
         for x in range(0,int(img.shape[1]/2.5)):
             img2[y,x,:]=img[int(y*2.5),int(x*2.5),:]
     print(img2.shape)
     return img2

        


class Player:
    def __init__(self, width, height):
        self.w = 40
        self.h = 40
        self.x = width // 2
        self.y = height - self.h
        self.player_character = cv2.imread("chicken.png", cv2.IMREAD_UNCHANGED)  


    def move_left(self):
        self.x = max(0, self.x - 20)  # Ensure the player stays within the left border

    def move_right(self, width):
        self.x = min(width - self.w, self.x + 20)  # Ensure the player stays within the right border


    def display(self, img):
        player_character_rgb = self.player_character[:, :, :3]
        y_start = max(0, self.y)
        y_end = min(img.shape[0], self.y + self.h)
        x_start = max(0, self.x)
        x_end = min(img.shape[1], self.x + self.w)

        img[y_start:y_end, x_start:x_end, :] = player_character_rgb[:y_end - y_start, :x_end - x_start, :]

class Enemy:
    def __init__(self, width, speed):
        self.w = 40
        self.h = 40
        self.x = random.randint(20, width - 20)
        self.y = 0 - self.h
        self.speed = speed
        self.enemy_character = cv2.imread("fox.png", cv2.IMREAD_UNCHANGED)

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

        # Check if the updated position is still within the valid range of the image
        if 0 <= self.y < img.shape[0]:
            enemy_character_rgb = self.enemy_character[:, :, :3]
            y_start = max(0, self.y)
            y_end = min(img.shape[0], self.y + self.h)
            x_start = max(0, self.x)
            x_end = min(img.shape[1], self.x + self.w)

            img[y_start:y_end, x_start:x_end, :] = enemy_character_rgb[:y_end - y_start, :x_end - x_start, :]

class BorderEnemy:
    def __init__(self, x, speed):
        self.w = 20
        self.h = 20
        self.x = x
        self.y = random.randint(-height, 0)
        self.speed = speed
        self.border_enemy_character = cv2.imread("bush.png", cv2.IMREAD_UNCHANGED)

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

        # Check if the updated position is still within the valid range of the image
        if 0 <= self.y < img.shape[0]:
            border_enemy_character_rgb = self.border_enemy_character[:, :, :3]
            y_start = max(0, self.y)
            y_end = min(img.shape[0], self.y + self.h)
            x_start = max(0, self.x)
            x_end = min(img.shape[1], self.x + self.w)

            img[y_start:y_end, x_start:x_end, :] = border_enemy_character_rgb[:y_end - y_start, :x_end - x_start, :]
        
def resize(img):
     img2=np.zeros(((int(img.shape[0]/2.5))+1,(int(img.shape[1]//2.5))+1,3),img.dtype)
     for y in range(0,int(img.shape[0]/2.5)):
         for x in range(0,int(img.shape[1]/2.5)):
             img2[y,x,:]=img[int(y*2.5),int(x*2.5),:]
     print(img2.shape)
     return img2
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

# Initialize game parameters
width, height = 257, 480
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
vision = False

while True:

    ret, frame = VideoCap.read()
    frame=resize(frame)
    cv2.flip(frame,1, frame)

    img = cv2.imread("bg.png") 
    img = img[:height, :width, :]

    if game_mode:
        cv2.putText(img, "Score: {}".format(score), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(img, "Speed: {}".format(speed), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        if random.randint(0, nbr_enemies) == 0 and time.time() - last_enemy_time > enemy_delay:
            border_enemy_left = BorderEnemy(0, speed)
            border_enemy_right = BorderEnemy(width - border_enemy_left.w, speed)
            enemies.append(Enemy(width - border_enemy_left.w - border_enemy_right.w, speed))
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

    
    # # Concatenate images verticall
    # mask = detect_inrange(frame)
    # centre=center(mask)
    # cv2.circle(frame, centre, 5, (0, 0, 255),-1)
    # player.x=centre[0]
    # #player.y=centre[1]
    concatenated_image = cv2.vconcat([img, frame])

    
    cv2.imshow('Game Interface', concatenated_image)

    key = cv2.waitKey(10)
    if key == ord('q'):
        break
    elif key == ord(' '):  # Space key to start/restart the game
        score = 0
        speed = 10
        game_mode = True
        nbr_enemies = 30
    elif key == ord('v'):
        vision=True
    if not vision:
        if key == ord('a'):
            player.move_left()
        elif key == ord('d'):
            player.move_right(width)
cv2.destroyAllWindows()