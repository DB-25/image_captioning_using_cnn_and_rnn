# Dhruv Kamalesh Kumar
# Kirti Deepak Kshirsagar
# Mrudula Vivek Acharya

# Importing the libraries
import torch
from tkinter import *
from tkinter import ttk, filedialog

from dataloader import get_loader
import torchvision.transforms as transforms
from PIL import ImageTk
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inception_model = torch.load("./inception.pth")
efficientnet_model = torch.load("./efficient_net.pth")
inception_model.to(device)
efficientnet_model.to(device)
inception_model.eval()
efficientnet_model.eval()

# Normalizing and transforming the images
transform = transforms.Compose(
    [
        transforms.Resize((356, 356)),
        transforms.RandomCrop((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# Loading the dataset
train_loader, test_loader, dataset = get_loader(
    root_folder="./dataset/images/",
    annotation_file="./dataset/captions.txt",
    transform=transform,
    num_workers=1,
)


def upload_image():
    global image_file_name
    image_file_name = filedialog.askopenfilename(initialdir="/", title="Select an image",
                                                 filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
    # show the image on the UI
    image = Image.open(image_file_name)
    image = image.resize((299, 299))
    image = ImageTk.PhotoImage(image)
    panel = ttk.Label(frm, image=image)
    panel.grid(column=1, row=2, pady=10)
    panel.image = image

    # refresh the UI
    root.update()


def generate_caption():
    # generate the caption for the image using both the models
    # show the caption on the UI
    image_tensor = transform(Image.open(image_file_name)).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    caption_inception = inception_model.caption_image(image_tensor, dataset.vocab)
    caption_efficient_net = efficientnet_model.caption_image(image_tensor, dataset.vocab)
    # Add 2 labels to show the generated caption
    ttk.Label(frm, text="Inception Caption:").grid(column=0, row=3, pady=10)
    ttk.Label(frm, text="EfficientNet Caption:").grid(column=0, row=4, pady=10)
    ttk.Label(frm, text=format_caption(caption_inception)).grid(column=1, row=3, pady=10)
    ttk.Label(frm, text=format_caption(caption_efficient_net)).grid(column=1, row=4, pady=10)
    # refresh the UI
    root.update()


# method which takes a string and formats it to be displayed on the UI
# remove the <SOS> and <EOS> tokens from start and end of the caption
# make it sentence case
def format_caption(sentence):
    sentence = sentence[1:-1]
    # join the elements of the list with spaces
    sentence = " ".join(sentence)
    # capitalize first letter
    sentence = sentence.capitalize()
    # add period at the end if there isn't one already
    if not sentence.endswith('.'):
        sentence += '.'
    return sentence


# We will create a UI for the user to interact, the user will be able to upload an image and the model will generate a caption for the image
# The user will also be able to see the generated caption and the image
# The user can see the results of both the models and compare them

root = Tk()
frm = ttk.Frame(root, padding=10)

# Creating the UI
root.title("Image Captioning")
root.geometry("800x600")

# Add a grid
frm.grid()
ttk.Label(frm, text="Image Captioning", font=("Arial", 20)).grid(column=0, row=0, columnspan=2, pady=10)
# Add a button to upload an image and show it on the UI
ttk.Button(frm, text="Select Image", command=lambda: upload_image()).grid(column=0, row=1, pady=10)
# Add a button to generate the caption
ttk.Button(frm, text="Generate Caption", command=lambda: generate_caption()).grid(column=1, row=1, pady=10)

root.mainloop()
