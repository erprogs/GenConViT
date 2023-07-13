import torch
import torch.nn as nn
from torchvision import transforms
from timm import create_model
import timm
from .model_embedder import HybridEmbed

class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(    
            nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),

            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),

            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        )

    def forward(self, x):
        return self.features(x)

class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(16, 3, kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.features(x)

class GenConViTED(nn.Module):
    def __init__(self, config, pretrained=True):
        super(GenConViTED, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.backbone = timm.create_model(config['model']['backbone'], pretrained=pretrained)
        self.embedder = timm.create_model(config['model']['embedder'], pretrained=pretrained)
        self.backbone.patch_embed = HybridEmbed(self.embedder, img_size=config['img_size'], embed_dim=768)

        self.num_features = self.backbone.head.fc.out_features * 2
        self.fc = nn.Linear(self.num_features, self.num_features//4)
        self.fc2 = nn.Linear(self.num_features//4, 2)
        self.relu = nn.GELU()

    def forward(self, images):

        encimg = self.encoder(images)
        decimg = self.decoder(encimg)

        x1 = self.backbone(decimg)
        x2 = self.backbone(images)

        x = torch.cat((x1,x2), dim=1)

        x = self.fc2(self.relu(self.fc(self.relu(x))))

        return x