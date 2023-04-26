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

    #print("Model 1 - Inception")
    helper.print_examples(model_inception, device, dataset)
    #print("Model 2 - Efficient Net")
    helper.print_examples(model_efficient_net, device, dataset)
    accuracy(model_inception,train_loader,device,dataset,train=True)
    accuracy(model_inception, test_loader, device, dataset, train=False)
    accuracy(model_efficient_net,train_loader,device,dataset,train=True)
    accuracy(model_efficient_net,test_loader,device,dataset,train=False)

def accuracy(m,loader,device,dataset,train=True):
    #for each image in loader get model output get orignal predicted

        counter=0
        score=[]

        for idx, (imgs, captions) in tqdm(
                enumerate(loader), total=len(loader), leave=False
        ):

            counter_index = counter
            candidate=[]
            prohibited = ["<SOS>", "<EOS>", "<PAD>", "<UNK>", "."]
            c=m.caption_image(imgs.to(device),dataset.vocab)
            for e in c:
                if e not in prohibited:
                    candidate.append(e)

            len_dataset = len(dataset)
            len_train = int(0.8 * len_dataset)
            if train:
                start_index=0
                end_index=len_train
            else:
                start_index=len_train
                end_index = len_dataset

            counter=start_index
            reference = []
            prohibited = ["<SOS>", "<EOS>", "<PAD>", "<UNK>", "."]
            for j in range(counter_index, counter_index+5):
                l = []
                for i in range(dataset[j][1].shape[0]):

                    word = dataset.vocab.itos[dataset[j][1][i].item()]
                    if word == "<EOS>":
                        reference.append(l)
                        l = []
                    elif word not in prohibited:
                        l.append(word)

                    print(dataset.vocab.itos[dataset[j][1][i].item()])
            print("reference: ", reference)
            print("candidate: ", candidate)
            score.append(nltk.translate.bleu_score.sentence_bleu(reference, candidate))
            counter+=5
        print("score: ", sum(score)/len(score))




if __name__ == "__main__":
    test()
