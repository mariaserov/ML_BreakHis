# Import
import timm
import json
import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset
from urllib.request import urlopen
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset
from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import confusion_matrix

n_epochs = 30
n_classes = 8
folder_to_save = 'convnext_v2_outputs/array_job_output_CV_milticlass'

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

perfcols = ['Fold', 'Epoch', 'Train_Recall', 'Train_Spec', 'Train_F1', 'Train_AUC',
          'Test_Recall', 'Test_Spec', 'Test_F1', 'Test_AUC' ]

perf_dfs = []

for c in range(n_classes):
    tracking_df = pd.read_csv(f"{folder_to_save}/cv_perf_class{c+1}.csv", index_col = 'Unnamed: 0') # tracking performance
    perf_dfs.append(tracking_df)

params = {'lr': 0.0001, 'optimizer': 'adam', 'batch_size': 16, 'weight_decay': 0.0001}

# Start model trainiing

print(f"CV iteration {idx+1}")

model = timm.create_model('convnextv2_atto.fcmae', pretrained=True, num_classes=n_classes)
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
train_acc = []
train_recall = [ [] for _ in range(n_classes) ]
train_spec   = [ [] for _ in range(n_classes) ]
train_f1     = [ [] for _ in range(n_classes) ]
train_auc    = [ [] for _ in range(n_classes) ]
test_loss = []
test_acc = []
test_recall = [ [] for _ in range(n_classes) ]
test_spec   = [ [] for _ in range(n_classes) ]
test_f1     = [ [] for _ in range(n_classes) ]
test_auc    = [ [] for _ in range(n_classes) ]

for epoch in range(n_epochs):

    train_labels_list = []
    train_probs_list = []
    test_labels_list = []
    test_probs_list = []

    model.train()
    running_loss = 0
    total = 0
    
    for image, _, label in train_loader:
        
        optimizer.zero_grad()
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)  # accumulate all probabilities => (batch_size, n_classes)
        train_probs_list.append(probs.detach()) # N, n_classes
        train_labels_list.append(label)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * image.size(0)
        _, preds = torch.max(outputs, 1)
        label = label.long()
        preds = preds.long()

    all_train_labels = torch.cat(train_labels_list).detach().numpy()
    all_train_probs  = torch.cat(train_probs_list).detach().numpy()
    train_pred_labels = all_train_probs.argmax(axis=1)
    cm = confusion_matrix(all_train_labels, train_pred_labels, labels=range(n_classes))
    total = cm.sum()
    train_loss.append(running_loss / total)
    train_acc.append(np.trace(cm) / total)

    # one-vs-rest metrics:

    for k in range(n_classes):
        tp = cm[k,k]
        fn = cm[k, :].sum() - tp
        fp = cm[:, k].sum() - tp
        tn = total - (tp+fn+fp)
        rec = tp/(tp+fn) if (tp+fn) >0 else 0
        spec = tn/(tn+fp) if (tn+fp) >0 else 0
        f1 = tp / (tp+0.5*(fp+fn)) if (tp+0.5*(fp+fn))>0 else 0
        auck = auc((all_train_labels == k).astype(int),all_train_probs[:, k])

        train_recall[k].append(rec)
        train_spec[k].append(spec)
        train_f1[k].append(f1)
        train_auc[k].append(auck)

    model.eval()
    running_loss_test = 0
    total_test = 0
    
    with torch.no_grad(): # to speed things up
        for image, _, label in test_loader:
            outputs = model(image)
            loss = criterion(outputs, label)
            _, preds = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)  # accumulate all probabilities => (batch_size, n_classes)
            test_probs_list.append(probs.detach())
            test_labels_list.append(label)
            running_loss_test += loss.item() * image.size(0)
            label = label.long()
            preds = preds.long()
    
    
    all_test_labels = torch.cat(test_labels_list).detach().numpy()
    all_test_probs  = torch.cat(test_probs_list).detach().numpy()
    test_pred_labels = all_test_probs.argmax(axis=1)
    cm_test = confusion_matrix(all_test_labels, test_pred_labels, labels=range(n_classes))
    total_test = cm_test.sum()
    test_loss.append(running_loss_test / total_test)
    test_acc.append(np.trace(cm_test) / total_test)

    for k in range(n_classes):
        tp = cm_test[k,k]
        fn = cm_test[k, :].sum() - tp
        fp = cm_test[:, k].sum() - tp
        tn = total_test - (tp+fn+fp)
        rec = tp/(tp+fn) if (tp+fn) >0 else 0
        spec = tn/(tn+fp) if (tn+fp) >0 else 0
        f1 = tp / (tp+0.5*(fp+fn)) if (tp+0.5*(fp+fn))>0 else 0
        auck = auc((all_test_labels == k).astype(int),all_test_probs[:, k])

        test_recall[k].append(rec)
        test_spec[k].append(spec)
        test_f1[k].append(f1)
        test_auc[k].append(auck)
    
    for k in range(n_classes):
        row = pd.DataFrame(data = [[
            f"Fold{idx+1}", f"Epoch{epoch+1}", 
            train_recall[k][epoch], train_spec[k][epoch], train_f1[k][epoch], train_auc[k][epoch],
            test_recall[k][epoch], test_spec[k][epoch], test_f1[k][epoch], test_auc[k][epoch]
        ]], columns = perfcols)
        perf_dfs[k] = pd.concat([perf_dfs[k], row]).reset_index(drop=True)

    print(f"Epoch {epoch+1}/{n_epochs}: Train Accuracy {np.trace(cm)/total}, Test Accuracy {np.trace(cm_test)/total_test}")

for k in range(n_classes):
    df = perf_dfs[k]
    df.to_csv(f"{folder_to_save}/cv_perf_class{k+1}.csv")

