# Importing the libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataloader import get_loader
from helper import load_checkpoint, save_model_inception, save_model_efficient_net, save_checkpoint_inception, \
    save_checkpoint_efficient_net
from model import CNNtoRNNEfficientNet, CNNtoRNNInception

# Normalizing and transforming the images
transform = transforms.Compose(
    [
        transforms.Resize((356, 356)),
        transforms.RandomCrop((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_model = False
saveModel = True

# Loading the dataset
train_loader, test_loader, dataset = get_loader(
    root_folder="./dataset/images/",
    annotation_file="./dataset/captions.txt",
    transform=transform,
    num_workers=1,
)


# train the efficient_net model
def train_efficient_net():
    # Hyperparameters
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 100

    # for tensorboard
    writer = SummaryWriter("./runs/flickr")
    step = 0

    # initialize model, loss etc
    model_efficient_net = CNNtoRNNEfficientNet(embed_size, hidden_size, vocab_size, num_layers)
    model_efficient_net.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model_efficient_net.parameters(), lr=learning_rate)

    # only for loading the model
    if load_model:
        step = load_checkpoint(torch.load("my_checkpoint_efficient_net.pth.tar"), model_efficient_net, optimizer)

    model_efficient_net.train()

    for epoch in range(num_epochs):

        for idx, (imgs, captions) in tqdm(
                enumerate(train_loader), total=len(train_loader), leave=False
        ):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model_efficient_net(imgs, captions[:-1])
            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
            )

            writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()

            if idx % 100 == 0:
                print(f"\nEpoch [{epoch}/{num_epochs}] Loss: {loss.item():.4f}")
    # save the model
    if saveModel:
        checkpoint = {
            "state_dict": model_efficient_net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
        }
        save_checkpoint_efficient_net(checkpoint)
        save_model_efficient_net(model_efficient_net)


# train the inception model
def train_inception():
    # Hyperparameters
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 100

    # for tensorboard
    writer = SummaryWriter("./runs/flickr")
    step = 0

    # initialize model, loss etc
    model_inception = CNNtoRNNInception(embed_size, hidden_size, vocab_size, num_layers)
    model_inception.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model_inception.parameters(), lr=learning_rate)

    # only for loading the model
    if load_model:
        step = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model_inception, optimizer)

    model_inception.train()

    for epoch in range(num_epochs):

        for idx, (imgs, captions) in tqdm(
                enumerate(train_loader), total=len(train_loader), leave=False
        ):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model_inception(imgs, captions[:-1])
            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
            )

            writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()

            if idx % 100 == 0:
                print(f"\nEpoch [{epoch}/{num_epochs}] Loss: {loss.item():.4f}")
    # save the model
    if saveModel:
        checkpoint = {
            "state_dict": model_inception.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
        }
        save_checkpoint_inception(checkpoint)
        save_model_inception(model_inception)


# main function
if __name__ == "__main__":
    print("What model do you want to train? \nInception - 1\nEfficient_net - 2\nBoth - 3")
    # input for which model to training
    input_for_training = int(input())
    # training the Inception model
    if input_for_training == 1:
        print("Training - Inception")
        train_inception()
    # training the Efficient_net model
    elif input_for_training == 2:
        print("Training - Efficient_net")
        train_efficient_net()
    # training both the models
    elif input_for_training == 3:
        print("Training - Inception and Efficient_net")
        print("Training - Inception")
        train_inception()
        print("Training - Efficient_net")
        train_efficient_net()
    # wrong input from user
    else:
        print("Wrong input")
