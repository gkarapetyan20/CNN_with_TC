from torch.utils.data import DataLoader , random_split
from torchvision import transforms
import torchvision.datasets as datasets
# Set your desired transformations (e.g., resize, normalize, etc.)
####################----------IMAGENET---------------###############
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),  # Resize images and masks to a consistent size
#     transforms.ToTensor(),           # Convert images and masks to tensors
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225],
#     ),
# ])

transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))  # Normalize images to [-1, 1]
])

dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)

def build_dataloader():
    # dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)

    # Step 3: Split the dataset into train, validation, and test sets
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    total_samples = len(dataset)
    train_size = int(train_ratio * total_samples)
    val_size = int(val_ratio * total_samples)
    test_size = total_samples - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    # Step 4: Create DataLoader for train, validation, and test sets
    batch_size = 64

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader , test_loader

