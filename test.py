# Dhruv Kamalesh Kumar
# 09-04-2023

# import the libraries
import torch
import helper
import dataloader
import torchvision.transforms as transforms
from tqdm import tqdm
import nltk

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Normalizing and transforming the images
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
    accuracy(model_inception, device, dataset, train=False)
    accuracy(model_efficient_net, device, dataset, train=False)

# method to calculate the accuracy of the model
# load each image one by one and generate the caption
# calculate the bleu score between the generated caption and the actual caption
# at the end return the average bleu score
def accuracy(m, device, dataset, train=True):
    prohibited = ["<SOS>", "<EOS>", "<PAD>", "<UNK>", ".", "-"]
    m.eval()
    with torch.no_grad():
        # loop through the dataset
        # start_index is the index from where the test data starts
        # if train is true, then start_index is 0
        # end_index is the index till where the test data ends
        # if train is true, then end_index is 80% of the dataset
        # if train is false, then end_index is 100% of the dataset
        score = []
        len_dataset = len(dataset)
        len_train = int(0.8 * len_dataset)
        if train:
            start_index = 0
            end_index = len_train
        else:
            start_index = len_train
            end_index = len_dataset

        # loop through the dataset
        for k in tqdm(range(start_index, end_index)):
            # get the image and the caption
            imgs = dataset[k][0].unsqueeze(0)
            caption = dataset[k][1]
            # covert the caption to a string, caption is a tensor
            caption = [dataset.vocab.itos[i.item()] for i in caption]
            caption = [caption]
            # generate the caption
            generated_caption = m.caption_image(imgs.to(device), dataset.vocab)
            # convert the generated caption to a string
            generated_caption = [i for i in generated_caption if i not in prohibited]
            # remove the prohibited words from the generated caption and caption
            caption = [i for i in caption if i not in prohibited]
            # calculate the bleu score between the generated caption and the actual caption
            score.append(nltk.translate.bleu_score.sentence_bleu(caption, generated_caption, (1, 0, 0, 0)))
        print("score: ", sum(score) / len(score))




# calls the test method which loads the model and tests its performance
if __name__ == "__main__":
    test()
