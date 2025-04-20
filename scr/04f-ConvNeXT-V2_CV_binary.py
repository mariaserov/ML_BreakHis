# Import
import timm
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset, random_split
from urllib.request import urlopen
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from torch.utils.data import Dataset
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset
from sklearn.metrics import roc_auc_score as auc

n_epochs = 30

# Array job 

idx = int(os.environ['PBS_ARRAY_INDEX'])-1

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

data = BreakHisDataset(csv_path="../data/aug_train_test_remerge.csv", transform=transform) # Load the data

# Load the split & iteration

with open("convnext_v2_outputs/convnextv2_cv_folds.json", "r") as f:
    folds = json.load(f)

train_idx = folds[idx]['train_idx']
test_idx = folds[idx]['val_idx']

# Set up params & performance tracking

perfcols = ['Fold', 'Epoch', 'Train_Loss', 'Train_Acc', 'Train_Recall', 'Train_Spec', 'Train_F1', 'Train_AUC',
          'Test_Loss', 'Test_Acc', 'Test_Recall', 'Test_Spec', 'Test_F1', 'Test_AUC' ]

df = pd.read_csv('convnext_v2_outputs/cv_perf.csv', index_col = 'Unnamed: 0') # tracking performance

params = {'lr': 0.0001, 'optimizer': 'adam', 'batch_size': 32, 'weight_decay': 0}

# Start model trainiing

print(f"CV iteration {idx+1}")

model = timm.create_model('convnextv2_atto.fcmae', pretrained=True, num_classes=2)
for param in model.parameters(): # Freeze all
    param.requires_grad = False  
for param in model.head.parameters(): # Unfreeze head
    param.requires_grad = True
for a in range(1, 4):
    for param in model.stages[a].parameters():
        param.requires_grad = True

optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],lr=params['lr'], weight_decay = params['weight_decay'])
criterion = nn.CrossEntropyLoss()

train = Subset(data, train_idx)
train_loader = DataLoader(train, batch_size=params['batch_size'], shuffle=True)
test = Subset(data, test_idx)
test_loader = DataLoader(test, batch_size=params['batch_size'], shuffle=False)

train_loss = []
test_loss = []
train_acc = []
test_acc = []
train_recall = []
test_recall = []
train_spec = []
test_spec = []
train_f1= []
test_f1= []
train_auc= []
test_auc= []

for epoch in range(n_epochs):

    train_labels_list = []
    train_probs_list = []
    test_labels_list = []
    test_probs_list = []

    model.train()
    running_loss = 0
    total = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    for image, label, _ in train_loader:
        
        optimizer.zero_grad()
        pred = model(image)
        probs = torch.softmax(pred, dim=1)[:,1]
        train_probs_list.append(probs)
        train_labels_list.append(label)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * image.size(0)
        _, preds = torch.max(pred, 1)
        label = label.long()
        preds = preds.long()
        tp += ((preds == 1) & (label == 1)).sum().item()
        tn += ((preds == 0) & (label == 0)).sum().item()
        fp += ((preds == 1) & (label == 0)).sum().item()
        fn += ((preds == 0) & (label == 1)).sum().item()
        total += label.size(0)
    
    all_train_labels = torch.cat(train_labels_list).detach().numpy()
    all_train_probs  = torch.cat(train_probs_list).detach().numpy()
    auc_train = auc(all_train_labels, all_train_probs)
    
    metric_lists_train = [train_loss, train_acc, train_recall, train_spec, train_f1, train_auc]
    metrics_train = [running_loss/total, (tp+tn)/total, tp/(tp+fn), tn/(tn+fp), tp/(tp+0.5*(fp+tn)), auc_train]
    
    for i in range(len(metric_lists_train)):
        metric_lists_train[i].append(metrics_train[i])
    
    model.eval()
    running_loss_test = 0
    total_test = 0
    tp_test = 0
    tn_test = 0
    fp_test = 0
    fn_test = 0
    
    for image, label, _ in test_loader:
        pred = model(image)
        loss = criterion(pred, label)
        _, preds = torch.max(pred, 1)
        probs = torch.softmax(pred, dim=1)[:,1]
        test_probs_list.append(probs)
        test_labels_list.append(label)
        running_loss_test += loss.item() * image.size(0)
        label = label.long()
        preds = preds.long()
        tp_test += ((preds == 1) & (label == 1)).sum().item()
        tn_test += ((preds == 0) & (label == 0)).sum().item()
        fp_test += ((preds == 1) & (label == 0)).sum().item()
        fn_test += ((preds == 0) & (label == 1)).sum().item()
        total_test += label.size(0)
    
    
    all_test_labels = torch.cat(test_labels_list).detach().numpy()
    all_test_probs  = torch.cat(test_probs_list).detach().numpy()
    auc_test = auc(all_test_labels, all_test_probs)
    
    metric_lists_test = [test_loss, test_acc, test_recall, test_spec, test_f1, test_auc]
    metrics_test = [running_loss_test/total_test, (tp_test+tn_test)/total_test, tp_test/(tp_test+fn_test), 
                tn_test/(tn_test+fp_test), tp_test/(tp_test+0.5*(fp_test+tn_test)), auc_test]
    
    for i in range(len(metric_lists_test)):
        metric_lists_test[i].append(metrics_test[i])
    
    row = pd.DataFrame(data=[[f"Fold{idx+1}", f"Epoch{epoch+1}"] + metrics_train + metrics_test], columns = perfcols)
    df = pd.concat([df, row]).reset_index(drop=True)
    print(f"Epoch {epoch+1}/{n_epochs}: Train Accuracy {(tp+tn)/total}, Test Accuracy {(tp_test+tn_test)/total_test}")

df.to_csv('convnext_v2_outputs/cv_perf.csv')