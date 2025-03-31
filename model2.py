import torch
import torch.nn as nn
import torch.nn.functional as F

class Second_Classifier(nn.Module):
    def __init__(self, lstm_hidden_dim, num_classes=2, dropout_rate=0.3, ff_hidden_dim=512):
        super(Second_Classifier, self).__init__()

        self.fc1 = nn.Linear(lstm_hidden_dim, ff_hidden_dim)
        self.fc2 = nn.Linear(ff_hidden_dim, ff_hidden_dim)

        self.classifier_layer = nn.Linear(ff_hidden_dim, num_classes)

        self.dropout = nn.Dropout(dropout_rate)

        self.batch_norm1 = nn.BatchNorm1d(lstm_hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(ff_hidden_dim)

        self._init_weights()

    def _init_weights(self):
        for layer in [self.fc1, self.fc2, self.classifier_layer]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, features):
        features = self.batch_norm1(features)

        x = F.leaky_relu(self.fc1(features))
        x = self.batch_norm2(x)
        x = self.dropout(x)

        x = F.leaky_relu(self.fc2(x))
        x = self.dropout(x)

        logits = self.classifier_layer(x)

        return logits

class ModelCombiner(nn.Module):
    def __init__(self, feature_extractor, classifier):
        super(ModelCombiner, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier
    def forward(self, x):
        features = self.feature_extractor(x)
        logits= self.classifier(features)
        return logits,features
