import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()

        # layers definition
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, features, captions):

        # make embeddings for captions
        # drop last word because we don't want to predict anything after <stop>
        # embeddings size = (batch_size, sequence_size-1, embed_size)
        embeddings = self.embed(captions[:,:-1])

        # reshape features from encoder
        # features size = (batch_size, 1, embed_size)
        features = features.view(features.shape[0], 1, -1)

        # concatenate features form encoder with embedded captions
        # concatenate on 2nd dimention (1 in Python) because 1st is batch size,
        # 3rd is features (informations) and 2nd is ordered sequence
        # features in the first position because lstm will predict first word from features only
        # next model will take first word and some "history informations" from previous LSTM cell
        inputs = torch.cat([features, embeddings], dim=1)

        # get LSTM output (this is all we need)
        # we can do the entire sequence all at once (vectorization)
        # the first value returned by LSTM is all of the hidden states throughout the sequence
        # the second is just the most recent hidden state
        x, _ = self.lstm(inputs)

        # decode LSTM output to vector of values of vocab_size
        # we need this for loss function
        outputs = self.fc(x)

        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "

        # prepare empty list for caption
        caption = []

        # reshape features form encoder
        # imputs shape = (batch_size, 1, embed_size)
        # for prediction batch_size = 1
        inputs = inputs.view(inputs.shape[0], 1, -1)

        # loop until <stop> is predicted or caption = max_len
        while True:
            # get LSTM output
            x, states = self.lstm(inputs, states)
            # decode LSTM output to vector of values of vocab_size
            output = self.fc(x)
            # index of largest value in that vector is most probable token
            _, max_id = torch.max(output, dim=2)
            # append that token do caption list
            caption.append(max_id.item())
            # make embeddings from predicted word
            # we use it as an input for next prediction
            inputs = self.embed(max_id)
            # stoping condition
            if max_id == 1 or len(caption) == max_len:
                break

        return caption
