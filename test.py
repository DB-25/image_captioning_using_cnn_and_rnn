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

    model_inception = torch.load("./inception.pth")
    model_inception.to(device)
    model_efficient_net = torch.load("./efficient_net.pth")
    model_efficient_net.to(device)
    loader, dataset = dataloader.get_loader(
        root_folder="./dataset/images/",
        annotation_file="./dataset/captions.txt",
        transform=transform,
        batch_size=1,
        num_workers=2,
    )
    print("Model 1 - Inception")
    helper.print_examples(model_inception, device, dataset)
    print("Model 2 - Efficient Net")
    helper.print_examples(model_efficient_net, device, dataset)


if __name__ == "__main__":
    test()
