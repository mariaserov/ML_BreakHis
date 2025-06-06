# Import

import os
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from urllib.request import urlopen
from PIL import Image
from torchvision import transforms
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset, random_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder

n_epochs = 20

folder_to_save = "array_job_output_multiclass"

# Array job 

idx = int(os.environ['PBS_ARRAY_INDEX'])
idx = idx-1 # 0-4

# Set up dataset 

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Converts to [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                         std=[0.229, 0.224, 0.225]),
])

# Define custom dataset


class BreakHisDataset(Dataset): # Subclass Dataset, which is required for using DataLoader
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.le = LabelEncoder()
        self.df['label_diagnosis'] = self.le.fit_transform(self.df['tumor_subtype'])
        self.class_names = list(self.le.classes_)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'filepath']
        label = self.df.loc[idx, 'label']
        subtype = self.df.loc[idx, 'label_diagnosis']   
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long), torch.tensor(subtype, dtype=torch.long)


## TEMP - UNCOMMENT LATER 

dataset_train = BreakHisDataset(csv_path="../data/augmented_train_dataset.csv", transform=transform) # Load the data
dataset_test = BreakHisDataset(csv_path="../data/new_test.csv", transform=transform)

# Create iterable data loaders

train_loader = DataLoader(dataset_train, batch_size=16, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=16, shuffle=False)


# TRAIN THE MODEL: Set up the training loop for different architectures

models = ['HeadOnly', 'Stage_4', 'Stages_3-4', 'Stages_2-4', 'Stages_1-4']


# Initialise loss criterion & optimiser
criterion = nn.CrossEntropyLoss()

# Sequentially unfreeze stages from top down

n_stages = 4

# NON-PARALLELISED LOOP 

# perf_train = pd.DataFrame(index=range(n_epochs), columns=models)
# perf_test = pd.DataFrame(index=range(n_epochs), columns=models)

# for i in range(n_stages+1):
    
#     # Initialise model
    
#     model = timm.create_model('convnextv2_atto.fcmae', pretrained=True, num_classes=2)
#     optimiser = optim.Adam(
#         [p for p in model.parameters() if p.requires_grad], 
#         lr=1e-3)
    
#     for param in model.parameters():
#         param.requires_grad = False
        
#     for param in model.head.parameters():
#         param.requires_grad = True
    
#     n_stages_to_unfreeze = i
#     print(f' {i}. Stages to unfreeze: {i}')
    
#     # Unfreeze stages
#     for a in range(n_stages - n_stages_to_unfreeze, n_stages):
#         for param in model.stages[a].parameters():
#             param.requires_grad = True
    
#     # Initialise tracking 
    
#     train_accuracies = []
#     test_accuracies = []
#     train_losses = []
#     test_losses = []
    
#     for epoch in range(n_epochs):
        
#         model.train()
        
#         running_loss = 0
#         correct = 0
#         total = 0
        
#         for images, labels in train_loader:
#             optimiser.zero_grad()
#             pred = model(images) # forward pass
#             loss = criterion(pred, labels) 
#             loss.backward() # updates model.grad with partial derivatives calc via chain rule
#             optimiser.step() # take a step in negative direction of grad using calculated derivatives
            
#             running_loss += loss.item() * images.size(0) # Accumulate loss per batch
            
#             _, preds = torch.max(pred, 1) # Take the maximum one as the class with the highest predicted probability => predicted class
            
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)
        
#         epoch_train_loss = running_loss/total
#         train_losses.append(epoch_train_loss)
        
#         epoch_train_accuracy = correct/total 
#         train_accuracies.append(epoch_train_accuracy)
        
#         model.eval()
        
#         running_loss_test = 0
#         correct_test = 0
#         total_test = 0
        
#         with torch.no_grad():
#             for images, labels in test_loader:
#                 pred = model(images)
#                 loss = criterion(pred, labels)
#                 running_loss_test += loss.item() * images.size(0)
#                 _, preds = torch.max(pred,1)
                
#                 correct_test += (preds == labels).sum().item()
#                 total_test += labels.size(0)
            
            
#         test_losses.append(running_loss_test/total_test)
            
#         epoch_test_accuracy = correct_test/total_test
#         test_accuracies.append(epoch_test_accuracy)
        
#         print(f"Epoch {epoch+1}/{n_epochs}: Train Accuracy {epoch_train_accuracy}, Test Accuracy {epoch_test_accuracy}")
    
#     perf_train.iloc[:,i] = train_accuracies
#     perf_test.iloc[:,i] = test_accuracies

# PARALLELISED
experiment = models[idx]

perf_train = pd.DataFrame(index=range(n_epochs), columns=[experiment])
perf_test = pd.DataFrame(index=range(n_epochs), columns=[experiment])

# Initialise model
    
model = timm.create_model('convnextv2_atto.fcmae', pretrained=True, num_classes=8)

for param in model.parameters():
    param.requires_grad = False
    
for param in model.head.parameters():
    param.requires_grad = True

n_stages_to_unfreeze = idx
print(f' {idx}. Stages to unfreeze: {idx}')

# Unfreeze stages
for a in range(n_stages - n_stages_to_unfreeze, n_stages):
    for param in model.stages[a].parameters():
        param.requires_grad = True

optimiser = optim.Adam(
    [p for p in model.parameters() if p.requires_grad], 
    lr=1e-4)

# Initialise tracking 

train_accuracies = []
test_accuracies = []
train_losses = []
test_losses = []

for epoch in range(n_epochs):
    
    model.train()
    
    running_loss = 0
    correct = 0
    total = 0
    
    for images, _, labels in train_loader:
        optimiser.zero_grad()
        pred = model(images) # forward pass
        loss = criterion(pred, labels) 
        loss.backward() # updates model.grad with partial derivatives calc via chain rule
        optimiser.step() # take a step in negative direction of grad using calculated derivatives
        
        running_loss += loss.item() * images.size(0) # Accumulate loss per batch
        
        _, preds = torch.max(pred, 1) # Take the maximum one as the class with the highest predicted probability => predicted class
        
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    epoch_train_loss = running_loss/total
    train_losses.append(epoch_train_loss)
    
    epoch_train_accuracy = correct/total 
    train_accuracies.append(epoch_train_accuracy)
    
    model.eval()
    
    running_loss_test = 0
    correct_test = 0
    total_test = 0
    
    with torch.no_grad():
        for images, _, labels in test_loader:
            pred = model(images)
            loss = criterion(pred, labels)
            running_loss_test += loss.item() * images.size(0)
            _, preds = torch.max(pred,1)
            
            correct_test += (preds == labels).sum().item()
            total_test += labels.size(0)
        
        
    test_losses.append(running_loss_test/total_test)
        
    epoch_test_accuracy = correct_test/total_test
    test_accuracies.append(epoch_test_accuracy)
    
    print(f"Epoch {epoch+1}/{n_epochs}: Train Accuracy {epoch_train_accuracy}, Test Accuracy {epoch_test_accuracy}")

perf_train.loc[:,experiment] = train_accuracies
perf_test.loc[:,experiment] = test_accuracies

perf_train.to_csv(f'convnext_v2_outputs/{folder_to_save}/{experiment}_train.csv')
perf_test.to_csv(f'convnext_v2_outputs/{folder_to_save}/{experiment}_test.csv')

print(perf_train)
print(perf_test)
