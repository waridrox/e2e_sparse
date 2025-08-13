import torch
import torch.nn as nn

import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import roc_curve
from sklearn import metrics
import gc
import wandb
import h5py as h5
import argparse
import random
import sys
sys.path.append("./Supervised/1DCNN/")
import model as CnnModel

def metric(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc = metrics.auc(fpr, tpr)
    return auc


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description="Training script")
    argparser.add_argument("--datapath", type=str, default="config.yaml", help="Path to config file")
    argparser.add_argument("--Nepochs", type=int, default=100, help="Number of training epochs")
    argparser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    argparser.add_argument("--model_variant",type=str, default="ResNet_PC_768_S", help="Model variant to use. Options: ResNet_PC_768_S, ResNet_PC_768_M, ResNet_PC_1024_S, ResNet_PC_1024_M,, ResNet_PC_1024_L")
    
    args = argparser.parse_args()
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    assert args.model_variant in ["ResNet_PC_768_S", "ResNet_PC_768_M", "ResNet_PC_1024_S", "ResNet_PC_1024_M", "ResNet_PC_1024_L"], "Invalid model variant"
    data_file = h5.File(args.datapath, 'r')

    BATCH_SIZE = 512
    Nepochs = args.Nepochs
    device = "cuda:0"

    model = getattr(CnnModel, args.model_variant)().model
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           'max',
                                                           verbose = True,
                                                           threshold = 0.0001,
                                                           patience = 5,
                                                           factor = 0.5)
    model = model.to(device)


    for epoch in range(Nepochs):
        
        train_loss = 0
        
        label_list = []
        output_list = []
        
        idx = torch.randperm(data_file["train_dataset"]["X"].shape[0])

        for batch in tqdm(range(0,data_file["train_dataset"]["X"].shape[0],BATCH_SIZE)):
            x_batch = torch.tensor(data_file["train_dataset"]["X"][idx[batch:(batch+BATCH_SIZE)]],device=device)
            y_batch = torch.tensor(data_file["train_dataset"]["Y"][idx[batch:(batch+BATCH_SIZE)]],device=device)

            optimizer.zero_grad()
            
            logit_out = model(x_batch.float())
            
            loss = criterion(logit_out, y_batch.float())
            loss.backward()
            optimizer.step()
        
            train_loss+= loss.item()
            
            label_list.append(y_batch.detach().cpu().numpy())
            output_list.append(logit_out.detach().cpu().numpy())
            
        
        label_list = np.concatenate(label_list)
        output_list = np.concatenate(output_list)
        
        train_loss /= data_file["train_dataset"]["X"].shape[0] // BATCH_SIZE
        train_auc = metric(y_true=label_list,y_pred=output_list)
        
        
        val_loss = 0
        
        label_list = []
        output_list = []
        
        for batch in tqdm(range(0,data_file["test_dataset"]["X"].shape[0],BATCH_SIZE)):
            
            x_batch = torch.tensor(data_file["test_dataset"]["X"][batch:(batch+BATCH_SIZE)],device=device)
            y_batch = torch.tensor(data_file["test_dataset"]["Y"][batch:(batch+BATCH_SIZE)],device=device)
            
            logit_out = model(x_batch.float())
            
            loss = criterion(logit_out, y_batch.float())
        
            val_loss+= loss.item()
            
            label_list.append(y_batch.detach().cpu().numpy())
            output_list.append(logit_out.detach().cpu().numpy())
            
        
        label_list = np.concatenate(label_list)
        output_list = np.concatenate(output_list)
        
        val_loss /= data_file["test_dataset"]["X"].shape[0] // BATCH_SIZE
        val_auc = metric(y_true=label_list,y_pred=output_list)

        print(f"Epoch {epoch+1}/{Nepochs} - "
            f"Train loss: {train_loss:.4f} - Train AUC: {train_auc:.4f} - "
            f"Val loss: {val_loss:.4f} - Val AUC: {val_auc:.4f}")

        scheduler.step(val_auc)
        gc.collect()
