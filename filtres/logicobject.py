import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk



# Fonctions associées aux boutons
def button_function_1():
    import filtreMoyen

def button_function_2():
    import filtreMedian

def button_function_3():
    import filtreGradient

def button_function_4():
    import filtreGaussien

def button_function_5():
    import laplacien

def button_function_6():
    import morpholog1

def button_function_7():
    import morpholog2

def button_function_8():
    print("Bouton 8 cliqué")

def button_function_9():
    print("Bouton 9 cliqué")    

# Fonction pour créer la fenêtre Tkinter avec les boutons
def create_gui():
    root = tk.Tk()
    root.title("Contrôles du jeu")

    # Créer les boutons
    button_1 = ttk.Button(root, text="Moyen", command=button_function_1)
    button_2 = ttk.Button(root, text="Median", command=button_function_2)
    button_3 = ttk.Button(root, text="Gradient", command=button_function_3)
    button_4 = ttk.Button(root, text="Gaussien", command=button_function_4)
    button_5 = ttk.Button(root, text="Laplacien", command=button_function_5)
    button_6 = ttk.Button(root, text="Erode/Dilate", command=button_function_6)
    button_7 = ttk.Button(root, text="Closing/Opening", command=button_function_7)
    button_8 = ttk.Button(root, text="Bouton 2", command=button_function_8)
    button_9 = ttk.Button(root, text="Bouton 3", command=button_function_9)


    # Positionner les boutons dans la fenêtre
    button_1.grid(row=0, column=0, padx=10, pady=10)
    button_2.grid(row=0, column=1, padx=10, pady=10)
    button_3.grid(row=0, column=2, padx=10, pady=10)
    button_4.grid(row=1, column=0, padx=10, pady=10)
    button_5.grid(row=1, column=1, padx=10, pady=10)
    button_6.grid(row=1, column=2, padx=10, pady=10)
    button_7.grid(row=2, column=0, padx=10, pady=10)
    button_8.grid(row=2, column=1, padx=10, pady=10)
    button_9.grid(row=2, column=2, padx=10, pady=10)


    root.mainloop()

# Appeler la fonction pour créer l'interface graphique
create_gui()



class GameObject:
    def __init__(self, size, speed, x, y):
        self.size = size
        self.speed = speed
        self.x = x
        self.y = y

    def update_position(self):
        self.x += self.speed

        # Wrap around if the object goes beyond the right edge
        if self.x > 640:  # Assuming the width of the frame is 640, adjust accordingly
            self.x = 0

    def draw(self, frame):
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.size, self.y + self.size), (0, 0, 0), -1)

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

# Create a game object
game_object = GameObject(size=50, speed=0, x=0, y=(480 // 2) - 25)  # Adjust size and initial y-coordinate accordingly

def move(dest):
    if dest == "Left":
        game_object.speed = -5
    elif dest == "Right":
        game_object.speed = 5
    elif dest == "Stop":
        game_object.speed = 0

while True:
    ret, frame = VideoCap.read()

    # Create a white image with the same size as the captured frame
    white_frame = np.ones_like(frame) * 255

    # Replace the right 3/4 of the white frame with the actual captured frame
    white_frame[:, frame.shape[1] // 4:, :] = frame[:, frame.shape[1] // 4:, :]

    # Call the Object_Color_Detection function to process the frame and detect objects
    _, mask, points = Object_Color_Detection(white_frame, 3000, 7000, lower_red, upper_red)

    # Add text to the white frame
    cv2.putText(white_frame, 'Score: 17.6', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(white_frame, 'Vitesse: 60', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # If points were detected, draw a circle at the detected location on the frame
    if len(points) > 0:
        cv2.circle(white_frame, (points[0][0], points[0][1]), 10, (0, 0, 255), 2)

    # Update and draw the game object
    game_object.update_position()
    game_object.draw(white_frame)

    if mask is not None:
        cv2.imshow('Brick Racer', white_frame)

    key = cv2.waitKey(10)
    if key == ord('q'):
        break
    elif key == ord('a'):
        move("Left")
    elif key == ord('d'):
        move("Right")
    elif key == ord('s'):
        move("Stop")

# Release the video capture object and close all windows
VideoCap.release()
cv2.destroyAllWindows()





