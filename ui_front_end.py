# Dhruv Kamalesh Kumar
# 17-04-2023

# Importing the libraries
import torch
from tkinter import *
from tkinter import ttk, filedialog

from PIL import ImageTk

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inception_model = torch.load("./inception.pth")
efficientnet_model = torch.load("./efficient_net.pth")
inception_model.to(device)
efficientnet_model.to(device)
inception_model.eval()
efficientnet_model.eval()


def upload_image():
    global image
    image = filedialog.askopenfilename(initialdir="/", title="Select an image",
                                       filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
    image = Image.open(image)
    image = image.resize((299, 299), Image.ANTIALIAS)
    image = ImageTk.PhotoImage(image)
    panel = ttk.Label(frm, image=image)
    panel.grid(column=0, row=2, pady=10)


def generate_caption():
    # generate the caption for the image using both the models
    # show the caption on the UI
    caption_inception = inception_model.generate_caption(image)
    caption_efficient_net = efficientnet_model.generate_caption(image)
    ttk.Label(frm, text=caption_inception, font=("Arial", 15)).grid(column=1, row=3, pady=10)
    ttk.Label(frm, text=caption_efficient_net, font=("Arial", 15)).grid(column=1, row=4, pady=10)


# We will create a UI for the user to interact, the user will be able to upload an image and the model will generate a caption for the image
# The user will also be able to see the generated caption and the image
# The user can see the results of both the models and compare them


root = Tk()
frm = ttk.Frame(root, padding=10)

# Creating the UI
root.title("Image Captioning")
root.geometry("800x600")
root.configure(background="white")

# Add a grid
frm.grid()
ttk.Label(frm, text="Image Captioning", font=("Arial", 20)).grid(column=0, row=0, columnspan=2, pady=10)
# Add a button to upload an image and show it on the UI
ttk.Button(frm, text="Upload Image", command=lambda: upload_image()).grid(column=1, row=1, pady=10)
# Add a button to generate the caption
ttk.Button(frm, text="Generate Caption", command=lambda: generate_caption()).grid(column=1, row=1, pady=10)
# Add 2 labels to show the generated caption
ttk.Label(frm, text="Caption 1", font=("Arial", 15)).grid(column=0, row=3, pady=10)
ttk.Label(frm, text="Caption 2", font=("Arial", 15)).grid(column=0, row=4, pady=10)

root.mainloop()