# Dhruv Kamalesh Kumar
# 09-04-2023

# import the libraries
import torch
import helper
import dataloader
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.Resize((356, 356)),
        transforms.RandomCrop((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


# method to load the model and test its performance
def test():
    # load the model
    model = torch.load("./inception.pth")
    model.to(device)
    loader, dataset = dataloader.get_loader(
        root_folder="./dataset/images/",
        annotation_file="./dataset/captions.txt",
        transform=transform,
        batch_size=1,
        num_workers=2,
    )
    helper.print_examples(model, device, dataset)


if __name__ == "__main__":
    test()
