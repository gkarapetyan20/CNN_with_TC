import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
# Step 1: Define the transformation
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Step 2: Load the CIFAR-100 dataset
train_dataset = datasets.CIFAR100(root='./data', train=True, transform=transform, download=True)

# Step 3: Calculate the mean and std
def calculate_mean_std(dataset):
    num_samples = len(dataset)
    sum_channel = torch.zeros(3)
    sum_channel_squared = torch.zeros(3)

    for inputs, _ in dataset:
        sum_channel += torch.mean(inputs, dim=(1, 2))
        sum_channel_squared += torch.mean(inputs**2, dim=(1, 2))

    mean = sum_channel / num_samples
    std = torch.sqrt(sum_channel_squared / num_samples - mean**2)
    return mean, std

mean, std = calculate_mean_std(train_dataset)

print("Mean of the CIFAR-100 dataset:")
print(mean)
print("\nStandard Deviation of the CIFAR-100 dataset:")
print(std)
