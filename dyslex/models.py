from torch import nn
from torchvision import models

# Multi-layer perceptron
class MLP(nn.Module):
    def __init__(self, input_feature_size, hidden_layer_size, output_class_count, hidden_layer_count=1, dropout=0.5):
        super().__init__()
        self.flatten = nn.Flatten()

        classifier_layers = []
        classifier_layers.append(
            nn.Linear(input_feature_size, hidden_layer_size))
        classifier_layers.append(nn.ReLU())
        for i in range(hidden_layer_count - 1):
            classifier_layers.append(
                nn.Linear(hidden_layer_size, hidden_layer_size))
            classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Dropout(dropout))
        classifier_layers.append(
            nn.Linear(hidden_layer_size, output_class_count))
        self.layers = nn.Sequential(*classifier_layers)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.layers(x)
        return logits

# ResNet18
def binary_resnet18():
    network = models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = network.fc.in_features
    network.fc = nn.Linear(num_ftrs, 2)
    return network

# ResNet50
def binary_resnet50():
    network = models.resnet50(weights='IMAGENET1K_V1')
    num_ftrs = network.fc.in_features
    network.fc = nn.Linear(num_ftrs, 2)
    return network
