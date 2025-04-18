# Import

import os
import timm
import torch
import itertools
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder

n_epochs = 10
folder_to_save = "array_job_output_HPO"


# Array job 

os.mkdir(f"convnext_v2_outputs/{folder_to_save}")
idx = int(os.environ['PBS_ARRAY_INDEX'])-1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up dataset 

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Converts to [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                         std=[0.229, 0.224, 0.225]),
])

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

dataset_train = BreakHisDataset(csv_path="../data/augmented_train_dataset.csv", transform=transform) # Load the data
dataset_test = BreakHisDataset(csv_path="../data/new_test.csv", transform=transform)
# iterable data loaders defined later in this case as batch size is in HPO

# HYPERPARAMS
hyperparams = {
    'lr': [1e-3, 1e-4, 1e-5],
    'optimizer': ['adam', 'sgd'],
    'batch_size': [16, 32],
    'weight_decay': [0, 1e-4]}

keys = hyperparams.keys()
values = hyperparams.values()
all_configs = []
for prod in itertools.product(*values): # Make all combinations of parameters
    config = dict(zip(keys, prod))
    all_configs.append(config)

config_cols = []
for i in range(len(all_configs)):
    config_cols.append(f"Config_{i}")

n_stages = 4

# Initialise model

model = timm.create_model('convnextv2_atto.fcmae', pretrained=True, num_classes=8)
model.to(device) 

criterion = nn.CrossEntropyLoss()
config = all_configs[idx]

perf_train = pd.DataFrame(index=range(n_epochs), columns=[config_cols[idx]])
perf_test = pd.DataFrame(index=range(n_epochs), columns=[config_cols[idx]])

for param in model.parameters():
    param.requires_grad = False
    
for param in model.head.parameters():
    param.requires_grad = True

for a in range(1,4):
    for param in model.stages[a].parameters():
            param.requires_grad = True

# Initialise optimiser 

if config['optimizer'] == 'adam':
    optimiser = optim.Adam([p for p in model.parameters() if p.requires_grad],lr=config['lr'], weight_decay = config['weight_decay'])
elif config['optimizer'] == 'sgd':
    optimiser = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=config['lr'], weight_decay = config['weight_decay'])
else:
    raise ValueError(f"Optimiser {config['optimizer']} not recognised")

# Initialise data loaders
train_loader_inloop = DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True)
test_loader_inloop = DataLoader(dataset_test, batch_size=config['batch_size'], shuffle=False)

# Initialise tracking of accuracies

train_accuracies = []
test_accuracies = []
train_losses = []
test_losses = []

print(f"Config{idx+1}: {config}")
for epoch in range(n_epochs):
    
    model.train()
    
    running_loss = 0
    correct = 0
    total = 0
    
    for images, labels, _ in train_loader_inloop:
        images = images.to(device)
        labels = labels.to(device)
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
        for images, labels, _ in test_loader_inloop:
            images = images.to(device)
            labels = labels.to(device)
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

perf_train.loc[:,config_cols[idx]] = train_accuracies
perf_test.loc[:,config_cols[idx]] = test_accuracies

perf_train.to_csv(f'convnext_v2_outputs/{folder_to_save}/Config{idx+1}_train.csv')
perf_test.to_csv(f'convnext_v2_outputs/{folder_to_save}/Config{idx+1}_test.csv')

print(perf_train)
print(perf_test)
