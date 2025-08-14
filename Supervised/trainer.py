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
from CNN import model as CnnModel

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
    
    argparser.add_argument("--UseWandb",type=bool, default=False, help="Use WandB for logging")


    argparser.add_argument("--wandb_project", type=str, default=None, help="WandB project name")
    argparser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity name")
    argparser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name")
    argparser.add_argument("--wandb_key", type=str, default=None, help="WandB API key")

    args = argparser.parse_args()

    if args.UseWandb:
        assert args.wandb_project is not None, "WandB project name must be specified"
        assert args.wandb_entity is not None, "WandB entity name must be specified"
        assert args.wandb_run_name is not None, "WandB run name must be specified"
        assert args.wandb_key is not None, "WandB API key must be specified"

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



    config = {
        "Epochs": Nepochs,
        "Learning Rate": args.lr,
        "Model Variant": args.model_variant,
        "Batch Size": BATCH_SIZE
    }

    if args.UseWandb:
        wandb.login(key=args.wandb_key)
        wandb.init(project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=args.wandb_run_name,
                    settings=wandb.Settings(_disable_stats=True),
                    config=config,
                    dir="/dev/shm"
                    )
        
    for epoch in range(Nepochs):
        
        train_loss = 0
        
        label_list = []
        output_list = []
        
        idx = torch.randperm(data_file["train_dataset"]["X"].shape[0])

        for batch in tqdm(range(0,data_file["train_dataset"]["X"].shape[0],BATCH_SIZE)):

            batch_idx,_ = torch.sort(idx[batch:(batch+BATCH_SIZE)])
            x_batch = torch.tensor(data_file["train_dataset"]["X"][batch_idx],device=device)
            y_batch = torch.tensor(data_file["train_dataset"]["Y"][batch_idx],device=device)

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
        train_accuracy = np.where((label_list == output_list))[0].shape[0] / label_list.shape[0]

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
        val_accuracy = np.where((label_list == output_list))[0].shape[0] / label_list.shape[0]

        print(f"Epoch {epoch+1}/{Nepochs} - "
            f"Train loss: {train_loss:.4f} - Train AUC: {train_auc:.4f} - Train Accuracy: {train_accuracy:.4f} - "
            f"Val loss: {val_loss:.4f} - Val AUC: {val_auc:.4f} - Val Accuracy: {val_accuracy:.4f}")

        if args.UseWandb:
            wandb.log({
                "Epoch": epoch + 1,
                "Train Loss": train_loss,
                "Train AUC": train_auc,
                "Val Loss": val_loss,
                "Val AUC": val_auc,
                "Train Accuracy": train_accuracy,
                "Val Accuracy": val_accuracy,
                "Lr": optimizer.param_groups[0]["lr"]
            })

        scheduler.step(val_auc)
        gc.collect()

    wandb.finish()
