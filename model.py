import torch
import torch.nn as nn
import torch.nn.functional as F
from util.seed import set_seed

set_seed()





class ConvLayer(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim):
        super(ConvLayer, self).__init__()
        self.embedding_layer = nn.Conv1d(in_channels=input_dim, out_channels=embedding_dim, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm1d(embedding_dim)
        self.hidden_layer = nn.Conv1d(in_channels=embedding_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.batch_norm1(self.embedding_layer(x)))
        x = self.relu2(self.batch_norm2(self.hidden_layer(x)))
        return x


class BiLSTMLayer(nn.Module):
    def __init__(self, hidden_dim, lstm_hidden_dim):
        super(BiLSTMLayer, self).__init__()
        self.bilstm = nn.LSTM(input_size=hidden_dim, hidden_size=lstm_hidden_dim, num_layers=1,
                              bidirectional=True, batch_first=True)

    def forward(self, x):
        lstm_output, _ = self.bilstm(x)
        return lstm_output


class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, embedding_dim=505, hidden_dim=960, lstm_hidden_dim=512):
        super(FeatureExtractor, self).__init__()
        self.conv_layer = ConvLayer(input_dim, embedding_dim, hidden_dim)
        self.bilstm_layer = BiLSTMLayer(hidden_dim, lstm_hidden_dim)
        self.pooling = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = x.unsqueeze(2)  # Add a channel dimension for Conv1d
        conv_output = self.conv_layer(x)
        conv_output = conv_output.transpose(1, 2)  # Prepare for LSTM input
        lstm_output = self.bilstm_layer(conv_output)

        pooled_output = self.pooling(lstm_output.transpose(1, 2)).squeeze(2)  # Pooling for fixed-size representation
        return pooled_output
class Classifier(nn.Module):
    def __init__(self, lstm_hidden_dim, num_classes=2, dropout_rate=0.42):
        super(Classifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier_layer = nn.Linear(2 * lstm_hidden_dim, num_classes)

    def forward(self, features):
        features = self.dropout(features)
        logits = self.classifier_layer(features)
        return logits