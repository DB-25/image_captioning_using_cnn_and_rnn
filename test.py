# Dhruv Kamalesh Kumar
# 09-04-2023

# import the libraries
import torch
import helper
import dataloader
import torchvision.transforms as transforms
from tqdm import tqdm

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
    train_loader, test_loader, dataset = dataloader.get_loader(
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
    accuracy(model_inception,train_loader,device,dataset)
    accuracy(model_inception, test_loader, device, dataset)
    accuracy(model_efficient_net,train_loader,device,dataset)
    accuracy(model_efficient_net,test_loader,device,dataset)





def accuracy(m,loader,device,dataset):
    #for each image in loader get model output get orignal predicted 
    
        for idx, (imgs, captions) in tqdm(
                enumerate(loader), total=len(loader), leave=False
        ):
            print(m.caption_image(imgs.to(device),dataset.vocab))
            print(captions)




    



if __name__ == "__main__":
    test()
