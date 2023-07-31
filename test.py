#import torchvision.datasets as datasets
#import torch.utils.data as data
#from model.backbone import VGG_New
#import torch
#from data.data import *

#vgg_model = VGG_New()

#vgg_model.load_state_dict(torch.load('vgg_model.pth',map_location=torch.device('cpu') ) )
#vgg_model.eval()  # Set the model to evaluation mode (important for inference)

#_,_,test_loader = build_dataloader()

# Step 4: Perform inference on the test dataset
#def test_model(model, dataloader):
#    model.eval()
#    correct = 0
#    total = 0

#    with torch.no_grad():
#        for inputs, targets in dataloader:
#            inputs, targets = inputs, targets
#            outputs = model(inputs)
#            _, predicted = torch.max(outputs.data, 1)
#            total += targets.size(0)
#            correct += (predicted == targets).sum().item()

#    accuracy = 100 * correct / total
#    return accuracy


#test_accuracy = test_model(vgg_model, test_loader)
#print(f'Test Accuracy: {test_accuracy:.2f}%')


import torchvision.datasets as datasets
import torch.utils.data as data
from model.backbone import VGG_New
import torch
from data.data import *
from sklearn.metrics import precision_score, recall_score, f1_score

vgg_model = VGG_New()

vgg_model.load_state_dict(torch.load('vgg_model.pth',map_location=torch.device('cpu') ) )
vgg_model.eval()  # Set the model to evaluation mode (important for inference)

_,_,test_loader = build_dataloader()

# Step 4: Perform inference on the test dataset
def test_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    all_predicted = []
    all_labels = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs, targets
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_predicted, average='macro')
    recall = recall_score(all_labels, all_predicted, average='macro')
    f1 = f1_score(all_labels, all_predicted, average='macro')
    print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    return accuracy


test_accuracy = test_model(vgg_model, test_loader)
print("*" * 150)
print(f'Test Accuracy: {test_accuracy:.2f}%')


