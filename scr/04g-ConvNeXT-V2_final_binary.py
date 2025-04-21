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
from sklearn.metrics import confusion_matrix
import numpy as np

n_epochs = 30

# Array job 

params = {'lr': 0.0001, 'optimizer': 'adam', 'batch_size': 32, 'weight_decay': 0}

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

# # For testing on toy
# dataset = BreakHisDataset(csv_path="../data/toy_data/toy_metadata.csv", transform=transform) # Load the data
# train_size = int(0.8 * len(dataset))
# test_size = len(dataset) - train_size
# train, test = random_split(dataset, [train_size, test_size])

# For full dataset
train = BreakHisDataset(csv_path="../data/augmented_train_dataset.csv", transform=transform) # Load the data
test = BreakHisDataset(csv_path="../data/new_test.csv", transform=transform)

train_loader = DataLoader(train, batch_size=params['batch_size'], shuffle=True)
test_loader = DataLoader(test, batch_size=params['batch_size'], shuffle=False)

folder = 'convnext_v2_outputs/Single_Model'
perfcols = ['Epoch', 'Train_Loss', 'Train_Acc', 'Train_Recall', 'Train_Spec', 'Train_F1', 'Train_AUC',
          'Test_Loss', 'Test_Acc', 'Test_Recall', 'Test_Spec', 'Test_F1', 'Test_AUC' ]
df = pd.read_csv(f"{folder}/Single_model_binary.csv", index_col = 'Unnamed: 0') # tracking performance


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

# Define values for early stopping

patience      = 3                   # how many epochs to wait
best_val_loss = float('inf')        # “best so far”
counter       = 0                   # epochs since last improvement
best_weights  = None  

for epoch in range(n_epochs):
   
    train_labels_list = []
    train_probs_list = []
    train_preds_list = []

    model.train()
    running_loss = 0

    for image, label, _ in train_loader:

        optimizer.zero_grad()
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)[:,1]
        train_probs_list.append(probs)
        train_labels_list.append(label)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * image.size(0)
        _, preds = torch.max(outputs, 1)
        train_preds_list.append(preds)
        label = label.long()
        preds = preds.long()
    
    all_train_labels = torch.cat(train_labels_list).detach().numpy()
    all_train_probs  = torch.cat(train_probs_list).detach().numpy()
    all_train_preds = torch.cat(train_preds_list).detach().numpy()

    cm = confusion_matrix(all_train_labels, all_train_preds)
    total = cm.sum()
    auc_train = auc(all_train_labels, all_train_probs)
    
    tp = cm[1,1]
    fn = cm[1, 0]
    fp = cm[0,1]
    tn = cm[0,0]
    rec = tp/(tp+fn) if (tp+fn) >0 else 0
    spec = tn/(tn+fp) if (tn+fp) >0 else 0
    f1 = tp / (tp+0.5*(fp+fn)) if (tp+0.5*(fp+fn))>0 else 0
    


    metric_lists_train = [train_loss, train_acc, train_recall, train_spec, train_f1, train_auc]
    metrics_train = [running_loss/total, (tp+tn)/total, rec, spec, f1, auc_train]

    for i in range(len(metric_lists_train)):
        metric_lists_train[i].append(metrics_train[i])
    
    model.eval()
    
    running_loss_test = 0
    total_test = 0
    test_labels_list = []
    test_probs_list = []
    test_preds_list = []
    
    with torch.no_grad():
        for image, label, _ in test_loader:
            
            outputs = model(image)
            probs = torch.softmax(outputs, dim=1)[:,1]
            test_probs_list.append(probs)
            test_labels_list.append(label)
            loss = criterion(outputs, label)
            running_loss_test += loss.item() * image.size(0)
            _, preds = torch.max(outputs, 1)
            test_preds_list.append(preds)
            label = label.long()
            preds = preds.long()

        all_test_labels = torch.cat(test_labels_list).detach().numpy()
        all_test_probs  = torch.cat(test_probs_list).detach().numpy()
        all_test_preds = torch.cat(test_preds_list).detach().numpy()
        
        cm = confusion_matrix(all_test_labels, all_test_preds)
        total_test = cm.sum()
        auc_test = auc(all_test_labels, all_test_probs)
        
        tp_test = cm[1,1]
        fn_test = cm[1, 0]
        fp_test = cm[0,1]
        tn_test = cm[0,0]
        rec_test = tp_test/(tp_test+fn_test) if (tp_test+fn_test) >0 else 0
        spec_test = tn_test/(tn_test+fp_test) if (tn_test+fp_test) >0 else 0
        f1_test = tp_test / (tp_test+0.5*(fp_test+fn_test)) if (tp_test+0.5*(fp_test+fn_test))>0 else 0

        
        metric_lists_test = [test_loss, test_acc, test_recall, test_spec, test_f1, test_auc]
        metrics_test = [running_loss_test/total_test, (tp_test+tn_test)/total_test, rec_test, spec_test, f1_test, auc_test]

        for i in range(len(metric_lists_test)):
            metric_lists_test[i].append(metrics_test[i])
    
    row = pd.DataFrame(data=[[f"Epoch{epoch+1}"] + metrics_train + metrics_test], columns = perfcols)
    df = pd.concat([df, row]).reset_index(drop=True)
    print(f"Epoch {epoch+1}/{n_epochs}: Train Accuracy {(tp+tn)/total}, Test Accuracy {(tp_test+tn_test)/total_test}")
    
    val_loss = running_loss_test/total_test
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        best_weights = model.state_dict()
    else:
        counter += 1
        print(f"No improvement in validation loss for {counter} epochs")
        if counter >= patience:
            print(f"Stopping early at epoch {epoch+1}")
            break
    if best_weights is not None:
        model.load_state_dict(best_weights)
        print("Loaded best model from the epoch with lowest val_loss")

df.to_csv(f"{folder}/Single_model_binary.csv")

# Evaluate on test / holdout set 

holdout = BreakHisDataset(csv_path="../data/new_holdout.csv", transform=transform)
holdout_loader = DataLoader(holdout, batch_size=params['batch_size'], shuffle=False)

model.eval()
all_labels = []
all_probs = []
all_preds = []
running_loss = 0

with torch.no_grad():
    for image, label, _ in holdout_loader:
        outputs = model(image)
        all_labels.append(label)
        probs = torch.softmax(outputs, dim=1)[:,1]
        all_probs.append(probs)
        _, preds = torch.max(outputs, 1)
        all_preds.append(preds)
        running_loss += criterion(outputs, label).item() * label.size(0)

all_holdout_labels = torch.cat(all_labels).detach().numpy()
all_holdout_probs  = torch.cat(all_probs).detach().numpy()
all_holdout_preds = torch.cat(all_preds).detach().numpy()

cm_holdout = confusion_matrix(all_holdout_labels, all_holdout_preds)
total_holdout = cm_holdout.sum()
auc_holdout = auc(all_holdout_labels, all_holdout_probs)

tp_holdout = cm_holdout[1,1]
fn_holdout = cm_holdout[1, 0]
fp_holdout = cm_holdout[0,1]
tn_holdout = cm_holdout[0,0]
rec_holdout = tp_holdout/(tp_holdout+fn_holdout) if (tp_holdout+fn_holdout) >0 else 0
spec_holdout = tn_holdout/(tn_holdout+fp_holdout) if (tn_holdout+fp_holdout) >0 else 0
f1_holdout = tp_holdout / (tp_holdout+0.5*(fp_holdout+fn_holdout)) if (tp_holdout+0.5*(fp_holdout+fn_holdout))>0 else 0 

perfcols_h = ['Loss', 'Accuracy', 'Recall', 'Specificity', 'F1', 'AUC']
perf_row = [running_loss/total_holdout, (tp_holdout+tn_holdout)/total_holdout, 
           rec_holdout, spec_holdout, f1_holdout, auc_holdout]

perf_h = pd.DataFrame(data = [perf_row], columns = perfcols_h )
perf_h.to_csv("convnext_v2_outputs/Single_Model/Single_model_binary_holdout.csv")  