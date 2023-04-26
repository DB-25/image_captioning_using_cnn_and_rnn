# Dhruv Kamalesh Kumar
# Mrudula Vivek Acharya
# Kirti Deepak Kshirsagar


# Importing the libraries
import torch
import torch.nn as nn
import torchvision


############################################## Efficient Net ########################################################

# Defining the CNN model, it will be used to extract the features from the images
class CNNEfficientNet(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(CNNEfficientNet, self).__init__()
        self.output_size = embed_size
        self.train_CNN = train_CNN
        self.model = torchvision.models.efficientnet_b7(
            weights=torchvision.models.EfficientNet_B7_Weights.IMAGENET1K_V1)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.classifier = nn.Linear(in_features=2560, out_features=self.output_size, bias=True)

    def forward(self, images):
        features = self.model(images)
        return features


# Defining the RNN model, it will be used to generate the captions given an image feature vector and a caption input
class RNNEfficientNet(nn.Module):
    # Constructor 
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(RNNEfficientNet, self).__init__()
        self.embed_size = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(p=0.5)

    # Forward propagation
    def forward(self, features, captions):
        embeddings = self.embed_size(captions)
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        lstm_out, _ = self.lstm(embeddings)
        outputs = self.linear(lstm_out)
        return outputs


# Combining the CNN and RNN models
class CNNtoRNNEfficientNet(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, train_CNN=False):
        super(CNNtoRNNEfficientNet, self).__init__()
        # Initializing the encoder and decoder
        self.encoder = CNNEfficientNet(embed_size, train_CNN)
        self.decoder = RNNEfficientNet(embed_size, hidden_size, vocab_size, num_layers)

    # Forward propagation
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    # Generates the caption for the image using decoder
    def caption_image(self, image, vocab, max_length=50):
        result_caption = []

        with torch.no_grad():
            x = self.encoder(image)
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


###################################################### Inception Model ################################################
# This block of code performs image classification using the Inception-v3 architecture
# Defining the CNN model, it will be used to extract the features from the images
class CNNInception(nn.Module):
    # Initialize the model
    def __init__(self, embed_size, train_CNN=False):
        super(CNNInception, self).__init__()
        self.output_size = embed_size
        self.train_CNN = train_CNN
        self.inception_model = torchvision.models.inception_v3(
            weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1)
        for param in self.inception_model.parameters():
            param.requires_grad = False
        self.inception_model.fc = nn.Linear(self.inception_model.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    # Forward propagation for image classification
    def forward(self, images):
        features = self.inception_model(images)
        features = self.relu(features[0])
        features = self.dropout(features)
        return features


# Defining the RNN model, it will be used to generate the captions
class RNNInception(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(RNNInception, self).__init__()
        self.embed_size = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(p=0.5)

    # Forward propagation
    def forward(self, features, captions):
        embeddings = self.dropout(self.embed_size(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        lstm_out, _ = self.lstm(embeddings)
        outputs = self.linear(lstm_out)
        return outputs


# Combining the CNN and RNN models
class CNNtoRNNInception(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, train_CNN=False):
        super(CNNtoRNNInception, self).__init__()
        self.encoder = CNNInception(embed_size, train_CNN)
        self.decoder = RNNInception(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    # Generates the caption for the image using a pre-trained encoder-decoder model
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
