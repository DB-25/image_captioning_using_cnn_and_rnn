# Importing the necessary libraries
import torch
import torchvision.transforms as transforms
import PIL.Image as Image


# Method to print the examples
def print_examples(model, device, dataset):
    # Applying image transformations to get uniform images
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    model.eval()
    # Loading test images and predicting captions
    test_img1 = transform(Image.open("./test_examples/dog.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 1 CORRECT: Dog on a beach by the ocean")
    print(
        "Example 1 OUTPUT: "
        + " ".join(model.caption_image(test_img1.to(device), dataset.vocab))
    )
    test_img2 = transform(
        Image.open("./test_examples/child.jpg").convert("RGB")
    ).unsqueeze(0)
    print("Example 2 CORRECT: Child holding red frisbee outdoors")
    print(
        "Example 2 OUTPUT: "
        + " ".join(model.caption_image(test_img2.to(device), dataset.vocab))
    )
    test_img3 = transform(Image.open("./test_examples/bus.png").convert("RGB")).unsqueeze(
        0
    )
    print("Example 3 CORRECT: Bus driving by parked cars")
    print(
        "Example 3 OUTPUT: "
        + " ".join(model.caption_image(test_img3.to(device), dataset.vocab))
    )
    test_img4 = transform(
        Image.open("./test_examples/boat.png").convert("RGB")
    ).unsqueeze(0)
    print("Example 4 CORRECT: A small boat in the ocean")
    print(
        "Example 4 OUTPUT: "
        + " ".join(model.caption_image(test_img4.to(device), dataset.vocab))
    )
    test_img5 = transform(
        Image.open("./test_examples/horse.png").convert("RGB")
    ).unsqueeze(0)
    print("Example 5 CORRECT: A cowboy riding a horse in the desert")
    print(
        "Example 5 OUTPUT: "
        + " ".join(model.caption_image(test_img5.to(device), dataset.vocab))
    )
    test_img6 = transform(Image.open("./dataset/images/667626_18933d713e.jpg").convert("RGB")).unsqueeze(0)
    print("Example 6 CORRECT: Girl wearing a bikini lying on her back in a shallow pool of clear blue water")
    print("Example 6 OUTPUT: " + " ".join(model.caption_image(test_img6.to(device), dataset.vocab)))
    # Setting the model back to training mode
    model.train()


# Method to save the checkpoint of efficient net
def save_checkpoint_efficient_net(state, filename="my_checkpoint_efficient_net.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


# Method to save the checkpoint of inception
def save_checkpoint_inception(state, filename="my_checkpoint_inception.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


# Method to load the checkpoint of the model
def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step


# Method to save the efficient_net model
def save_model_efficient_net(model, filename="efficient_net.pth"):
    print("=> Saving model")
    torch.save(model, filename)


# Method to save the inception model
def save_model_inception(model, filename="inception.pth"):
    print("=> Saving model")
    torch.save(model, filename)
