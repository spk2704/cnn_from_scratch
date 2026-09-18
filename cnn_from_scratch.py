#Basic CNN with ASL dataset
#Download the dataset from https://www.kaggle.com/datasets/ayuraj/asl-dataset


import torch
from torchvision import datasets
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch import nn


transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

full_dataset = datasets.ImageFolder(root="asl_dataset", transform=transforms)
train_dataset, test_dataset = train_test_split(full_dataset, test_size=0.2, random_state=42, shuffle=True)

batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_pool_linear = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=9, out_channels=27, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(27*97*97, 36)
        )

    def forward(self, x):
        x = self.conv_pool_linear(x)
        return(x)

model = CNN().to(device)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
epochs = 20

def train(dataloader):
    model.train()

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()


def test(dataloader):
    model.eval()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y)
            test_loss += loss.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


for i in range(epochs):
    print(f"Epoch {i+1}\n-------------------------------")
    train(dataloader=train_dataloader)
    test(dataloader=test_dataloader)

    if (i == 19):
        torch.save(model.state_dict(), f"model.pth")
print("Done")


classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a",
    "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l",
    "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", 
    "x", "y", "z"]

model.eval()
x, y = test_dataset[20][0], test_dataset[20][1]
x = x.unsqueeze(0).to(device)

with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')    
