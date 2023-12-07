import customtkinter as ctk
import tkinter
import subprocess

def open_visual():
    root.destroy()
    subprocess.run(["python", "visual.py"])

def open_proces():
    root.destroy()
    subprocess.run(["python", "proces.py"])

def generate():
    user_prompt += "in style: " + dataset.get()

def reset():
    # Add reset functionality here
    pass

root = ctk.CTk()
root.title("Analyse_nettoyage")

# Get the screen width and height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Set the window dimensions to match the screen size
root.geometry(f"{screen_width}x{screen_height}")

ctk.set_appearance_mode("dark")

# reset_frame = ctk.CTkFrame(root)
# reset_frame.pack(side="top", pady=10)
# reset_button = ctk.CTkButton(reset_frame, text="Reset", command=reset)
# reset_button.pack()

input_frame = ctk.CTkFrame(root)
input_frame.pack(side="left", expand=True, padx=20, pady=20)
style_label = ctk.CTkLabel(input_frame, text="Data")
style_label.grid(row=1, column=0, padx=10, pady=10)
dataset = ctk.CTkComboBox(input_frame, values=["Sol", "Covid-19", "Agriculteure"])
dataset.grid(row=1, column=1, padx=10, pady=10)
generate_button = ctk.CTkButton(input_frame, text="Generate", command=generate)
generate_button.grid(row=3, column=0, columnspan=2, sticky="news", padx=10, pady=10)

input_frame2 = ctk.CTkFrame(root)
input_frame2.pack(side="left", expand=True, padx=20, pady=20)
generate_button4 = ctk.CTkButton(input_frame2, text="EDA", command=open_visual)
generate_button4.grid(row=3, column=0, columnspan=2, sticky="news", padx=10, pady=10)
generate_button3 = ctk.CTkButton(input_frame2, text="Preprocess", command=open_proces)
generate_button3.grid(row=4, column=0, columnspan=2, sticky="news", padx=10, pady=10)
generate_button2 = ctk.CTkButton(input_frame2, text="Extraction", command=generate)
generate_button2.grid(row=5, column=0, columnspan=2, sticky="news", padx=10, pady=10)

canvas = tkinter.Canvas(root, width=800, height=screen_height)
canvas.pack(side="right")

root.mainloop()
