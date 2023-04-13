# Dhruv Kamalesh Kumar
# 09-04-2023

# Importing the libraries
import torch
import torch.nn as nn
import torchvision
import numpy as np


# Defining the CNN model, it will be used to extract the features from the images
class CNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(CNN, self).__init__()
        self.output_size = embed_size
        self.train_CNN = train_CNN
        self.inception_model = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1)
        self.inception_model.fc = nn.Linear(self.inception_model.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, images):
        features = self.inception_model(images)

        for name, param in self.inception_model.named_parameters():
            if 'fc.weight' in name or 'fc.bias' in name:
                param.requires_grad = True
            else:
                param.requires_grad = self.train_CNN

        features = self.relu(features[0])
        features = self.dropout(features)
        return features


class RNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(RNN, self).__init__()
        self.embed_size = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed_size(captions))
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
                predicted = output.argmax(1)
                result_caption.append(predicted.item())
                x = self.decoder.embed_size(predicted).unsqueeze(0)

                if vocab.itos[predicted.item()] == '<EOS>':
                    break

        return [vocab.itos[idx] for idx in result_caption]

