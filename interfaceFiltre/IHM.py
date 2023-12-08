import customtkinter as ctk
import tkinter
import subprocess
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import customtkinter as ctk
import requests, io
import function_Interface
import cv2
import time
import numpy as np




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
    kernel = np.array([ [55, 55, 55],
                        [55, 55, 55],
                        [55, 55, 55]], dtype=np.uint8)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_filtree = function_Interface.erode(image, kernel)
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
    closing_kernel = np.array([[55, 55, 55],
                           [55, 55, 55],
                           [55, 55, 55]], dtype=np.uint8)
    seuil = 128
    BN = function_Interface.seuillage_binaire(image_path, seuil)
    image_filtree = function_Interface.closing(BN, closing_kernel)
    new_image_path = image_path.replace('.jpg', '_filtree.jpg') 
    cv2.imwrite(new_image_path, image_filtree)
    display_image_on_canvas(new_image_path)
    root.after(10, lambda: display_second_image7_1(image_path))

def display_second_image7_1(image_path):
    kernel = np.array([[15, 15, 15],
                           [15, 15, 15],
                           [15, 15, 15]], dtype=np.uint8)
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

# Fonctions associées aux boutons
def button_function_1(img):
    #import filtreMoyen
    display_image_on_canvas(img)
    # Wait for 3 seconds
    root.after(3000, lambda: display_second_image(img))
    

def button_function_2(img):
    #import filtreMedian
    display_image_on_canvas(img)
    # Wait for 3 seconds
    root.after(3000, lambda: display_second_image2(img))

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

def button_function_9():
    #import filtre_seuillage  
    pass 


def close():
    root.destroy()


def open_proces():
    root.destroy()
    subprocess.run(["python", "proces.py"])

def generate():
    #user_prompt += "in style: " + dataset.get()
    pass

def reset():
    # Add reset functionality here
    pass

root = ctk.CTk()
root.title("Filters Test")

# Get the screen width and height
#screen_width = root.winfo_screenwidth()
#screen_height = root.winfo_screenheight()

screen_width = 800
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
generate_button3.grid(row=5, column=0, columnspan=2, sticky="news", padx=10, pady=10)
generate_button4 = ctk.CTkButton(input_frame2, text="Gaussien", command=lambda: button_function_4('noisy.jpg'))
generate_button4.grid(row=6, column=0, columnspan=2, sticky="news", padx=10, pady=10)
generate_button5 = ctk.CTkButton(input_frame2, text="Laplacien", command=lambda: button_function_5('univer.jpg'))
generate_button5.grid(row=7, column=0, columnspan=2, sticky="news", padx=10, pady=10)
generate_button6 = ctk.CTkButton(input_frame2, text="Erode/Dilate", command=lambda: button_function_6('univer.jpg'))
generate_button6.grid(row=8, column=0, columnspan=2, sticky="news", padx=10, pady=10)
generate_button7 = ctk.CTkButton(input_frame2, text="Closing/Opening", command=lambda: button_function_7('univer.jpg'))
generate_button7.grid(row=9, column=0, columnspan=2, sticky="news", padx=10, pady=10)
generate_button8 = ctk.CTkButton(input_frame2, text="Prewitt", command=lambda: button_function_8('univer.jpg'))
generate_button8.grid(row=10, column=0, columnspan=2, sticky="news", padx=10, pady=10)
generate_button9 = ctk.CTkButton(input_frame2, text="Seuillage", command=button_function_9)
generate_button9.grid(row=11, column=0, columnspan=2, sticky="news", padx=10, pady=10)

#generate_button10 = ctk.CTkButton(input_frame2, text="Clean", command=button_function_9)
#generate_button10.grid(row=12, column=0, columnspan=2, sticky="news", padx=10, pady=10)




canvas = tkinter.Canvas(root, width=740, height=screen_height)
canvas.pack(side="right")


root.mainloop()
