import torch.nn as nn


class VGGTransferLearningNet(nn.Module):
    def __init__(self, model, num_classes=133, freeze=True):
        super(VGGTransferLearningNet, self).__init__()
        self.net = model
        self.net.classifier[6] = nn.Linear(
            model.classifier[6].in_features, num_classes, bias=True
        )

        if freeze:
            for param in self.net.parameters():
                param.requires_grad = False

            for param in self.net.classifier.parameters():
                param.requires_grad = True

    def forward(self, x):
        x = self.net(x)
        return x


class ResNetTransferLearningNet(nn.Module):
    def __init__(self, model, num_classes=133, freeze=True):
        super(ResNetTransferLearningNet, self).__init__()
        self.net = model
        self.net.fc = nn.Linear(model.fc.in_features, num_classes, bias=True)

        if freeze:
            for param in self.net.parameters():
                param.requires_grad = False

            for param in self.net.fc.parameters():
                param.requires_grad = True

    def forward(self, x):
        x = self.net(x)
        return x


class SimpleNet(nn.Module):
    def __init__(self, num_classes=133):
        super(SimpleNet, self).__init__()

        self.cnn_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(7 * 7 * 128, 500),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(500, num_classes),
        )

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.cnn_layer(x)
        x = x.reshape(batch_size, 7 * 7 * 128)
        x = self.fc_layer(x)
        return x
