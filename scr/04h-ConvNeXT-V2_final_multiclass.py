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
n_classes = 8
patience = 5                  # how many epochs to wait
folder = 'convnext_v2_outputs/Single_Model/Multiclass'

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


# For full dataset
train = BreakHisDataset(csv_path="../data/augmented_train_dataset.csv", transform=transform) # Load the data
test = BreakHisDataset(csv_path="../data/new_test.csv", transform=transform)

train_loader = DataLoader(train, batch_size=params['batch_size'], shuffle=True)
test_loader = DataLoader(test, batch_size=params['batch_size'], shuffle=False)

# Tracking class-specific metrics 
perfcols_perclass = ['Epoch', 'Class', 'Train_Recall', 'Train_Spec', 'Train_F1', 'Train_AUC',
          'Test_Recall', 'Test_Spec', 'Test_F1', 'Test_AUC' ]

# Tracking general metrics
perfcols_gen = ['Epoch', 'Train_Loss', 'Train_Acc', 'Test_Loss', 'Test_Acc']
df_gen = pd.DataFrame(columns = perfcols_gen)

model = timm.create_model('convnextv2_atto.fcmae', pretrained=True, num_classes=8)
for param in model.parameters(): # Freeze all
    param.requires_grad = False  
for param in model.head.parameters(): # Unfreeze head
    param.requires_grad = True
for a in range(1, 4):
    for param in model.stages[a].parameters():
        param.requires_grad = True

optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],lr=params['lr'], weight_decay = params['weight_decay'])
criterion = nn.CrossEntropyLoss()

train_recall = [ [] for _ in range(n_classes) ]
train_spec   = [ [] for _ in range(n_classes) ]
train_f1     = [ [] for _ in range(n_classes) ]
train_auc    = [ [] for _ in range(n_classes) ]
test_recall = [ [] for _ in range(n_classes) ]
test_spec   = [ [] for _ in range(n_classes) ]
test_f1     = [ [] for _ in range(n_classes) ]
test_auc    = [ [] for _ in range(n_classes) ]

# Define values for early stopping

best_val_loss = float('inf')        # “best so far”
counter       = 0                   # epochs since last improvement
best_weights  = None  

for epoch in range(n_epochs):
   
    train_labels_list = []
    train_probs_list = []
    train_preds_list = []
    running_loss = 0

    model.train()

    for image, _, label in train_loader:

        optimizer.zero_grad()
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        train_probs_list.append(probs.detach()) # N, n_classes
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

    test_labels_list = []
    test_probs_list = []
    test_preds_list = []
    
    with torch.no_grad():
        for image, _, label in test_loader:
            
            outputs = model(image)
            probs = torch.softmax(outputs, dim=1)
            test_probs_list.append(probs.detach())
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
        
        cm_test = confusion_matrix(all_test_labels, all_test_preds)
        total_test = cm_test.sum()
        
    for k in range(n_classes):
        tp_test = cm_test[k,k]
        fn_test = cm_test[k, :].sum() - tp_test
        fp_test = cm_test[:, k].sum() - tp_test
        tn_test = total_test - (tp_test+fn_test+fp_test)
        rec_test = tp_test/(tp_test+fn_test) if (tp_test+fn_test) >0 else 0
        spec_test = tn_test/(tn_test+fp_test) if (tn_test+fp_test) >0 else 0
        f1_test = tp_test / (tp_test+0.5*(fp_test+fn_test)) if (tp_test+0.5*(fp_test+fn_test))>0 else 0
        auck_test = auc((all_test_labels == k).astype(int),all_test_probs[:, k])

        test_recall[k].append(rec_test)
        test_spec[k].append(spec_test)
        test_f1[k].append(f1_test)
        test_auc[k].append(auck_test)

    row = pd.DataFrame(data=[[epoch, running_loss, cm.trace() / cm.sum(), running_loss_test, cm_test.trace() / cm_test.sum()]])
    df_gen = pd.concat([df_gen, row], ignore_index=True)
    print(f"Epoch {epoch+1}/{n_epochs}: Train Accuracy {cm.trace() / cm.sum()}, Test Accuracy {cm_test.trace() / cm_test.sum()}")
    
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

# Get the best-performing model
if best_weights is not None:
    model.load_state_dict(best_weights)
    print("Loaded best model from the epoch with lowest val_loss")
    torch.save(model.state_dict(), f"{folder}/best_model_multiclass.pth")

# Record all per-class metrics to be saved in dataframe

all_perf = [train_recall, train_spec, train_f1, train_auc, 
          test_recall, test_spec, test_f1, test_auc]

for a in all_perf:
    for b in a:
        if len(b)< 8:
            n_append = 8-len(b)
            add = [float("nan") for _ in range(n_append)]
            b.extend(add)
        else:
            pass

rows = []
for epoch in range(len(train_recall[0])):            # number of recorded epochs
    for k in range(n_classes):
        rows.append({
            'Epoch':       epoch,
            'Class':       k,
            'Train_Recall': train_recall[k][epoch],
            'Train_Spec':   train_spec[k][epoch],
            'Train_F1':     train_f1[k][epoch],
            'Train_AUC':    train_auc[k][epoch],
            'Test_Recall':  test_recall[k][epoch],
            'Test_Spec':    test_spec[k][epoch],
            'Test_F1':      test_f1[k][epoch],
            'Test_AUC':     test_auc[k][epoch],
        })

df_perclass = pd.DataFrame(rows, columns=perfcols_perclass)

# Evaluate on test / holdout set 

holdout = BreakHisDataset(csv_path="../data/new_holdout.csv", transform=transform)
holdout_loader = DataLoader(holdout, batch_size=params['batch_size'], shuffle=False)

model.eval()
all_labels = []
all_probs = []
all_preds = []
running_loss_holdout = 0

with torch.no_grad():
    for image, _, label in holdout_loader:
        outputs = model(image)
        all_labels.append(label)
        probs = torch.softmax(outputs, dim=1)
        all_probs.append(probs.detach())
        _, preds = torch.max(outputs, 1)
        all_preds.append(preds)
        running_loss_holdout += criterion(outputs, label).item() * label.size(0)

all_holdout_labels = torch.cat(all_labels).detach().numpy()
all_holdout_probs  = torch.cat(all_probs).detach().numpy()
all_holdout_preds = torch.cat(all_preds).detach().numpy()

cm_holdout = confusion_matrix(all_holdout_labels, all_holdout_preds)
total_holdout = cm_holdout.sum()
overall_acc = np.trace(cm_holdout) / total_holdout
avg_loss = running_loss_holdout / total_holdout

perclass_rows = []
for k in range(n_classes):
    tp_holdout = cm_holdout[k,k]
    fn_holdout = cm_holdout[k, :].sum() - tp_holdout
    fp_holdout = cm_holdout[:, k].sum() - tp_holdout
    tn_holdout = total_holdout - (tp_holdout + fn_holdout + fp_holdout)
    rec_holdout = tp_holdout/(tp_holdout+fn_holdout) if (tp_holdout+fn_holdout) >0 else 0
    spec_holdout = tn_holdout/(tn_holdout+fp_holdout) if (tn_holdout+fp_holdout) >0 else 0
    f1_holdout = tp_holdout / (tp_holdout+0.5*(fp_holdout+fn_holdout)) if (tp_holdout+0.5*(fp_holdout+fn_holdout))>0 else 0 
    auc_holdout    = auc((all_holdout_labels == k).astype(int), all_holdout_probs[:, k])

    perclass_rows.append({
        'Class':         k,
        'Holdout_Recall':    rec_holdout,
        'Holdout_Specificity': spec_holdout,
        'Holdout_F1':         f1_holdout,
        'Holdout_AUC':        auc_holdout,
    })

df_holdout_perclass = pd.DataFrame(perclass_rows, columns=['Class','Holdout_Recall',
                                                           'Holdout_Specificity','Holdout_F1','Holdout_AUC'])
df_holdout_summary = pd.DataFrame([{'Holdout_Loss':  avg_loss, 'Holdout_Accuracy': overall_acc}])

# Save all performances

df_gen.to_csv(f"{folder}/Epochs_summary.csv")
df_perclass.to_csv(f"{folder}/Epochs_perclass.csv")
df_holdout_perclass.to_csv(f"{folder}/Holdout_perclass.csv")
df_holdout_summary.to_csv(f"{folder}/Holdout_summary.csv")

