# Importing the necessary libraries
import os
import pandas as pd
import torch
import spacy
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms

# Loading the spacy english model
spacy_eng = spacy.load("en_core_web_sm")

# set random seed
torch.manual_seed(123)

# This class is used to create and maintain the vocabulary for the captions
class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    # Tokenize the sentence using spacy
    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    # Building the vocabulary
    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    # Numericalize the text
    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
                for token in tokenized_text]

# This class loads and preprocesses the images and captions from the Flickr8k dataset
class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform

        # Get image,caption columns
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        # Initialize vocabulary and build vocab

        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    # Get the image and caption at the specified index
    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])
        return img, torch.tensor(numericalized_caption)


# This class is used to pad the captions to the same length
class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)
        return imgs, targets

# This function is used to create the train and test dataloaders
def get_loader(
        root_folder,
        annotation_file,
        transform,
        batch_size=32,
        num_workers=8,
        shuffle=True,
        pin_memory=True,

):
    dataset = FlickrDataset(root_folder, annotation_file, transform=transform)
    pad_idx = dataset.vocab.stoi["<PAD>"]
    len_dataset = len(dataset)
    len_train = int(0.8 * len_dataset)


    # Created using indices from 0 to train_size.
    train_dataset = torch.utils.data.Subset(dataset, range(len_train))

    # Created using indices from train_size to train_size + test_size.
    test_dataset = torch.utils.data.Subset(dataset, range(len_train, len_dataset))

    # Create train and test data loaders.
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=shuffle,
                              pin_memory=pin_memory,
                              collate_fn=MyCollate(pad_idx=pad_idx),
                              )
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=shuffle,
                             pin_memory=pin_memory,
                             collate_fn=MyCollate(pad_idx=pad_idx),
                             )

    return train_loader, test_loader, dataset


if __name__ == "__main__":

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ]
    )
    train_loader, test_loader, dataset = get_loader("./dataset/images/",
                            annotation_file="./dataset/captions.txt",
                            transform=transform)

    print("Train_loader: ", len(train_loader))
    print("Test_loader: ", len(test_loader))

    print(dataset[0][1])



