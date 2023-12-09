import subprocess
import time
import customtkinter as ctk
import tkinter
import tkinter as tk
from PIL import Image, ImageTk
import function_Interface
import cv2
import numpy as np
import threading

#====================objectDetection==============================
stop_detection_flag = False
def update_canvas(frame, mask):
    global frame_photo, mask_photo

    # Convert frame and mask to ImageTk.PhotoImage objects
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    mask_pil = Image.fromarray(mask)
    
    frame_photo = ImageTk.PhotoImage(frame_pil)
    mask_photo = ImageTk.PhotoImage(mask_pil)

    # Update the canvas with the new images
    update_canvas_with_images(frame_photo, mask_photo)

def object_Detection_Color():
    global stop_detection_flag
    stop_detection_flag = False

    VideoCap = cv2.VideoCapture(0)

    while True:
        ret, frame = VideoCap.read()
        frame = function_Interface.resize(frame)
        cv2.flip(frame, 1, frame)
        mask = function_Interface.detect_inrange(frame)
        centre = function_Interface.center(mask)

        if mask is not None:
            cv2.circle(frame, centre, 5, (0, 0, 255), -1)
            update_canvas(frame, mask)
        
        # Break out of the loop if the stop_detection_flag is True
        if stop_detection_flag:
            break

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    VideoCap.release()
    cv2.destroyAllWindows()


def display_image_on_canvas_from_image(image, frame):
    # Mettre à jour le canevas avec l'image
    image2 = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    update_canvas_with_images(image, image2)


def update_canvas_with_images(frame_image, mask_image):
    # Clear previous images on the canvas
    canvas.delete("all")

    # Display the images on the canvas
    canvas.create_image(canvas_width / 2, screen_height / 2, anchor=tk.CENTER, image=frame_image)
    canvas.create_image(0, 0, anchor=tk.NW, image=mask_image)

    # Keep references to the PhotoImage objects to prevent garbage collection
    canvas.frame_photo = frame_image
    canvas.mask_photo = mask_image
canvas_width = 740
screen_height = 400


def button_function_11():
    # Exécuter object_Detection_Color dans un thread
    threading.Thread(target=object_Detection_Color).start()

#=================================================Invisibility


def invisibility():
    global stop_detection_flag
    stop_detection_flag = False

    VideoCap = cv2.VideoCapture(0)
    time.sleep(2)
    ret, background = VideoCap.read()
    cv2.flip(background,1, background)
    background=function_Interface.resize(background)
    VideoCap.release()
    VideoCap = cv2.VideoCapture(0)
    while True:
        if not VideoCap.isOpened():
            print("Video capture is not open.")
            break
        ret, frame = VideoCap.read()
        cv2.flip(frame,1, frame)
        frame=function_Interface.resize(frame)
        mask = function_Interface.detect_inrange(frame)
        function_Interface.dilateV(frame,background,mask,np.ones((5,5)))
    
        if mask is not None:
            #cv2.imshow('mask', mask)
            #cv2.imshow('frame', frame)
            update_canvas(frame, mask)
        # Break out of the loop if the stop_detection_flag is True
        if stop_detection_flag:
            break

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    VideoCap.release()
    cv2.destroyAllWindows()

def button_function_12():
    # Exécuter object_Detection_Color dans un thread
    threading.Thread(target=invisibility).start()
#============================
def display_image_on_canvas(image_path):
    # Open the image file
    image = Image.open(image_path)

    # Create a PhotoImage object from the Image
    photo = ImageTk.PhotoImage(image)

    # Clear previous images on the canvas
    canvas.delete("all")

    # Display the image on the canvas
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)

    # Keep a reference to the PhotoImage to prevent it from being garbage collected
    canvas.image = photo


def display_second_image(image_path):
    image = cv2.imread(image_path)
    image_filtree = function_Interface.mean_filter(image, 5)
    new_image_path = image_path.replace('.jpg', '_filtree.jpg') 
    cv2.imwrite(new_image_path, image_filtree)
    display_image_on_canvas(new_image_path)

def display_second_image2(image_path):
    image = cv2.imread(image_path)
    image_filtree = function_Interface.median_filter(image, 5)
    new_image_path = image_path.replace('.jpg', '_filtree.jpg') 
    cv2.imwrite(new_image_path, image_filtree)
    display_image_on_canvas(new_image_path)

def display_second_image3(image_path):
    image = cv2.imread(image_path)
    image_filtree = function_Interface.gradient_filter(image, 3)
    new_image_path = image_path.replace('.jpg', '_filtree.jpg') 
    cv2.imwrite(new_image_path, image_filtree)
    display_image_on_canvas(new_image_path)


def display_second_image4(image_path):
    image = cv2.imread(image_path)
    gaussian_kernel_2d = function_Interface.generate_gaussian_kernel(5, 1.0)
    image_filtree = function_Interface.convolve(image, gaussian_kernel_2d)
    new_image_path = image_path.replace('.jpg', '_filtree.jpg') 
    cv2.imwrite(new_image_path, image_filtree)
    display_image_on_canvas(new_image_path)

def display_second_image5(image_path):
    #image = cv2.imread(image_path)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).tolist()
    image_filtree = function_Interface.filtre_laplacien(image)
    new_image_path = image_path.replace('.jpg', '_filtree.jpg') 
    cv2.imwrite(new_image_path, image_filtree)
    display_image_on_canvas(new_image_path)

def display_second_image6(image_path):
    seuil = 128
    BN = function_Interface.seuillage_binaire(image_path, seuil)
    kernel = np.array([ [1, 0, 1],
                        [0, 1, 0],
                        [1, 0, 1]], dtype=np.uint8)
    #image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_filtree = function_Interface.erode(BN, kernel)
    new_image_path = image_path.replace('.jpg', '_filtree.jpg') 
    cv2.imwrite(new_image_path, image_filtree)
    display_image_on_canvas(new_image_path)
    root.after(10, lambda: display_second_image6_1(image_path))


def display_second_image6_1(image_path):
    kernel = [
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1]
    ]
    seuil = 128
    BN = function_Interface.seuillage_binaire(image_path, seuil)
    image_filtree = function_Interface.dilation(BN, kernel)
    new_image_path = image_path.replace('.jpg', '_filtree.jpg') 
    cv2.imwrite(new_image_path, image_filtree)
    display_image_on_canvas(new_image_path)

def display_second_image7(image_path):
    closing_kernel = np.array([[1, 0, 1],
                           [0, 1, 0],
                           [1, 0, 1]], dtype=np.uint8)
    seuil = 128
    BN = function_Interface.seuillage_binaire(image_path, seuil)
    image_filtree = function_Interface.closing(BN, closing_kernel)
    new_image_path = image_path.replace('.jpg', '_filtree.jpg') 
    cv2.imwrite(new_image_path, image_filtree)
    display_image_on_canvas(new_image_path)
    root.after(100, lambda: display_second_image7_1(image_path))

def display_second_image7_1(image_path):
    kernel = np.array([[1, 0, 1],
                           [0, 1, 0],
                           [1, 0, 1]], dtype=np.uint8)
    seuil = 128
    BN = function_Interface.seuillage_binaire(image_path, seuil)
    image_filtree = function_Interface.opening(BN, kernel)
    new_image_path = image_path.replace('.jpg', '_filtree.jpg') 
    cv2.imwrite(new_image_path, image_filtree)
    display_image_on_canvas(new_image_path)

def display_second_image8(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_filtree = function_Interface.prewitt_filter(image, direction='horizontal')
    new_image_path = image_path.replace('.jpg', '_filtree.jpg') 
    cv2.imwrite(new_image_path, image_filtree)
    display_image_on_canvas(new_image_path)
    root.after(20, lambda: display_second_image8_1(image_path))


def display_second_image8_1(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_filtree = function_Interface.prewitt_filter(image, direction='vertical')
    new_image_path = image_path.replace('.jpg', '_filtree.jpg') 
    cv2.imwrite(new_image_path, image_filtree)
    display_image_on_canvas(new_image_path)

def display_second_image9(image_path):
    sobel_x_kernel = [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]
    sobel_y_kernel = [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ]
    seuil = 128
    BN = function_Interface.seuillage_binaire(image_path, seuil)
    sobel_x = function_Interface.sobel(BN, sobel_x_kernel)
    sobel_y = function_Interface.sobel(BN, sobel_y_kernel)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)  # magnitude du gradient
    new_image_path = image_path.replace('.jpg', '_filtree.jpg') 
    cv2.imwrite(new_image_path, gradient_magnitude)
    display_image_on_canvas(new_image_path)


# Fonctions associées aux boutons
def button_function_1(img):
    #import filtreMoyen
    display_image_on_canvas(img)
    # Wait for 3 seconds
    root.after(1000, lambda: display_second_image(img))
    

def button_function_2(img):
    #import filtreMedian
    display_image_on_canvas(img)
    # Wait for 3 seconds
    root.after(1000, lambda: display_second_image2(img))

def button_function_3(img):
    #import filtreGradient
    display_image_on_canvas(img)
    # Wait for 3 seconds
    root.after(1000, lambda: display_second_image3(img))

def button_function_4(img):
    #import filtreGaussien
    display_image_on_canvas(img)
    # Wait for 3 seconds
    root.after(1000, lambda: display_second_image4(img))

def button_function_5(img):
    #import laplacien
    display_image_on_canvas(img)
    # Wait for 3 seconds
    root.after(1000, lambda: display_second_image5(img))

def button_function_6(img):
    #import morpholog1
    display_image_on_canvas(img)
    # Wait for 3 seconds
    root.after(100, lambda: display_second_image6(img))

def button_function_7(img):
    #import morpholog2
    display_image_on_canvas(img)
    # Wait for 3 seconds
    root.after(100, lambda: display_second_image7(img))

def button_function_8(img):
    #import filtrePrewitt
    display_image_on_canvas(img)
    # Wait for 3 seconds
    root.after(100, lambda: display_second_image8(img))

def button_function_9(img):
    display_image_on_canvas(img)
    # Wait for 3 seconds
    root.after(100, lambda: display_second_image9(img)) 


def close():
    root.destroy()
    subprocess.run(["python", "Game.py"])



def reset():
    # Add reset functionality here
    pass

root = ctk.CTk()
root.title("Filters Test")

screen_width = 900
screen_height = 600
# Set the window dimensions to match the screen size
root.geometry(f"{screen_width}x{screen_height}")

ctk.set_appearance_mode("dark")

input_frame2 = ctk.CTkFrame(root)
input_frame2.pack(side="left", expand=True, padx=20, pady=20)
generate_button1 = ctk.CTkButton(input_frame2, text="Moyen", command=lambda: button_function_1('noisy.jpg'))
generate_button1.grid(row=3, column=0, columnspan=2, sticky="news", padx=10, pady=10)
generate_button2 = ctk.CTkButton(input_frame2, text="Median", command=lambda: button_function_2('noisy.jpg'))
generate_button2.grid(row=4, column=0, columnspan=2, sticky="news", padx=10, pady=10)
generate_button3 = ctk.CTkButton(input_frame2, text="Gradient", command=lambda: button_function_3('univer.jpg'))
generate_button3.grid(row=6, column=0, columnspan=2, sticky="news", padx=10, pady=10)
generate_button4 = ctk.CTkButton(input_frame2, text="Gaussien", command=lambda: button_function_4('noisy.jpg'))
generate_button4.grid(row=5, column=0, columnspan=2, sticky="news", padx=10, pady=10)
generate_button5 = ctk.CTkButton(input_frame2, text="Laplacien", command=lambda: button_function_5('univer.jpg'))
generate_button5.grid(row=7, column=0, columnspan=2, sticky="news", padx=10, pady=10)
generate_button6 = ctk.CTkButton(input_frame2, text="Erode/Dilate", command=lambda: button_function_6('univer.jpg'))
generate_button6.grid(row=8, column=0, columnspan=2, sticky="news", padx=10, pady=10)
generate_button7 = ctk.CTkButton(input_frame2, text="Closing/Opening", command=lambda: button_function_7('univer.jpg'))
generate_button7.grid(row=9, column=0, columnspan=2, sticky="news", padx=10, pady=10)
generate_button8 = ctk.CTkButton(input_frame2, text="Prewitt(H/V)", command=lambda: button_function_8('univer.jpg'))
generate_button8.grid(row=10, column=0, columnspan=2, sticky="news", padx=10, pady=10)
generate_button9 = ctk.CTkButton(input_frame2, text="Sobel", command=lambda: button_function_9('xford.jpg'))
generate_button9.grid(row=11, column=0, columnspan=2, sticky="news", padx=10, pady=10)




# Function to set the stop_detection_flag
def stop():
    global stop_detection_flag
    stop_detection_flag = True

def clean():
    canvas.delete("all")

input_frame3 = ctk.CTkFrame(root)
input_frame3.pack(side="left", expand=True, padx=20, pady=20)
generate_button10 = ctk.CTkButton(input_frame3, text="Jeu", command=close)
generate_button10.grid(row=14, column=0, columnspan=2, sticky="news", padx=10, pady=10)
generate_button14 = ctk.CTkButton(input_frame3, text="Invisibility", command=button_function_12)
generate_button14.grid(row=13, column=0, columnspan=2, sticky="news", padx=10, pady=10)
generate_button11 = ctk.CTkButton(input_frame3, text="ObjectDetection", command=button_function_11)
generate_button11.grid(row=12, column=0, columnspan=2, sticky="news", padx=10, pady=10)
generate_button12 = ctk.CTkButton(input_frame3, text="Stop", command=stop)
generate_button12.grid(row=15, column=0, columnspan=2, sticky="news", padx=10, pady=10)
generate_button13 = ctk.CTkButton(input_frame3, text="Clean", command=clean)
generate_button13.grid(row=16, column=0, columnspan=2, sticky="news", padx=10, pady=10)


canvas = tkinter.Canvas(root, width=740, height=screen_height)
canvas.pack(side="right")





root.mainloop()
