from torch import nn

__all__ = ["Model"]

class Normalizer(nn.Module):
    def __init__(self):
        super().__init__()
        return

    def forward(self, X):
        return X

class Block(nn.Module):
    def __init__(self, channels, normalizer, downsample=False):
        super().__init__()
        self.normalizer1 = normalizer()
        self.normalizer2 = normalizer()
        self.downsample = downsample
        self.layer1 = nn.Conv2d(in_channels=int(channels/(downsample+1)), out_channels=channels, kernel_size=(3, 3), stride=downsample+1, padding=1)
        self.layer2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), stride=1, padding=1)
        if downsample:
            self.project = nn.Conv2d(in_channels=int(channels/2), out_channels=channels, kernel_size=(1, 1), stride=2)

    def forward(self, X):
        temp = self.normalizer2(self.layer2(nn.functional.relu(self.normalizer1(self.layer1(X)))))
        if self.downsample:
            X = self.project(X)
        return nn.functional.relu(temp+X)

class Model(nn.Module):
    def __init__(self, normalizer=Normalizer):
        super().__init__()
        self.layer = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=2, padding=3),
                nn.modules.pooling.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
                Block(channels=64, normalizer=normalizer),
                Block(channels=64, normalizer=normalizer),
                Block(channels=128, normalizer=normalizer, downsample=True),
                Block(channels=128, normalizer=normalizer),
                Block(channels=256, normalizer=normalizer, downsample=True),
                Block(channels=256, normalizer=normalizer),
                Block(channels=512, normalizer=normalizer, downsample=True),
                Block(channels=512, normalizer=normalizer),
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
                )
        self.Linear = nn.Linear(in_features=512, out_features=100)

    def forward(self, X):
        X = self.layer(X)
        X = X.squeeze(-1).squeeze(-1)
        return self.Linear(X)

if __name__ == "__main__":
    from data import train_dataset
    from torchvision import transforms
    transform = transforms.Compose([
            transforms.RandomCrop((224, 224), pad_if_needed=True),
            transforms.RandomHorizontalFlip(p=0.5)
            ])
    testing_dataset = train_dataset("../val", transform)
    from torch.utils.data import DataLoader
    loader = DataLoader(testing_dataset, batch_size=8, shuffle=True)
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model()
    for batch in loader:
        X = batch["img"].to(device)
        model(X)
        break
    print("End of Script")
