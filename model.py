# Dhruv Kamalesh Kumar
# 09-04-2023

# Importing the libraries
import torch
import torch.nn as nn
import torchvision
from torchsummary import summary
import numpy as np


# Defining the CNN model, it will be used to extract the features from the images
class CNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(CNN, self).__init__()
        self.output_size = embed_size
        self.train_CNN = train_CNN
        self.model = torchvision.models.efficientnet_b7(weights=torchvision.models.EfficientNet_B7_Weights.IMAGENET1K_V1)
        # freeze the layers
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.classifier = nn.Linear(in_features=2560, out_features=self.output_size, bias=True)
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(p=0.5)

    def forward(self, images):
        features = self.model(images)
        return features


class RNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(RNN, self).__init__()
        self.embed_size = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, features, captions):

        # embeddings = self.dropout(self.embed_size(captions))
        # embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        # lstm_out, _ = self.lstm(embeddings)
        # outputs = self.linear(lstm_out)
        # return outputs
        embeddings = self.embed_size(captions)
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        lstm_out, _ = self.lstm(embeddings)
        outputs = self.linear(lstm_out)
        return outputs


class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, train_CNN=False):
        super(CNNtoRNN, self).__init__()
        self.encoder = CNN(embed_size, train_CNN)
        self.decoder = RNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def caption_image(self, image, vocab, max_length=50):
        result_caption = []

        with torch.no_grad():
            x = self.encoder(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoder.lstm(x, states)
                output = self.decoder.linear(hiddens.unsqueeze(0))
                predicted = torch.argmax(output)
                result_caption.append(predicted.item())
                x = self.decoder.embed_size(predicted).unsqueeze(0)

                if vocab.itos[predicted.item()] == '<EOS>':
                    break

        return [vocab.itos[idx] for idx in result_caption]

