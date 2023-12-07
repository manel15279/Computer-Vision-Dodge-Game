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
    import filtrePrewitt

def button_function_9():
    import filtre_seuillage   

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
    button_8 = ttk.Button(root, text="Prewitt", command=button_function_8)
    button_9 = ttk.Button(root, text="Seuillage", command=button_function_9)


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

