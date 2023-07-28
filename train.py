import torch
import torch.nn as nn
import torch.optim as optim
from data.data import *
from model.backbone import VGG_New
import torchvision.datasets as datasets


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



train_loader , val_loader , _ = build_dataloader()

vgg_model = VGG_New().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg_model.parameters(), lr=0.00001)


def train_vgg(vgg_model, dataloader, criterion, optimizer, num_epochs=50):
    vgg_model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = vgg_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader)}')

        val_loss = calculate_validation_loss(vgg_model, val_loader, criterion)
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss}')
        print("*" * 100)


def calculate_validation_loss(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()

    avg_val_loss = running_loss / len(val_loader)
    return avg_val_loss


train_vgg(vgg_model, train_loader, criterion, optimizer, num_epochs=50)


torch.save(vgg_model.state_dict(), 'vgg_model.pth')