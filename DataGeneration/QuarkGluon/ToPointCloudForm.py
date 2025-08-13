# This .py holds the functions to convert the raw output files post h5 creation to 
# point cloud form for the QuarkGluon dataset.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import h5py as h5
from tqdm.auto import tqdm
import argparse



def calculate_max_hits():
    max_hit = 0
    nevents = 1024
    for i in tqdm(range(0,Data_file["train_jet"].shape[0]//nevents + 1,1)):
        x = Data_file["train_jet"][i*nevents:(i+1)*nevents,:]
        mask = np.sum((x > 0),axis = -1) > 0
        nhits = np.sum(mask, (1,2))
        if nhits.max() > max_hit:
            max_hit = nhits.max()
    return max_hit


def trim(sample,CUT_POINTS):
    cost = sample[:,0]**2 + sample[:,1]**2 + sample[:,2]**2
    sort_idx = np.argsort(cost)[-CUT_POINTS:]
    sample_new = sample[sort_idx,:]
    return sample_new


def get_point_clouds(sample):
    mask = np.sum((sample > 0),axis = -1) > 0
    
    coord_x = []
    coord_y = []
    for i in range(mask.shape[1]):
        l_ = np.where(mask[i,:])[0]
        coord_x.append(l_)
        coord_y.append(np.array([i]*l_.shape[0]))
    
    coord_x = np.concatenate(coord_x,0)
    coord_y = np.concatenate(coord_y,0)

    point_cloud = [sample[:,:,0][mask[:,:]][:,None],
                   sample[:,:,1][mask[:,:]][:,None],
                   sample[:,:,2][mask[:,:]][:,None],
                   coord_x[:,None],
                   coord_y[:,None]]

    point_cloud = np.concatenate(point_cloud,-1)
    point_cloud = np.concatenate([point_cloud,np.zeros(((max_hit - point_cloud.shape[0]),5))], 0)
    return point_cloud


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Convert QuarkGluon dataset to point cloud form.")
    argparser.add_argument("--datadir",
                           default="/global/cfs/cdirs/m4392/ACAT_Backup/Data/QG",
                           type=str,
                           required=True,
                           help="Path to the data directory.")
    argparser.add_argument("--MaxPoints",
                           default=768,
                           type=int,
                           help="Maximum number of points in the point cloud.")
    argparser.add_argument("--OutFile",
                           default="output.h5",
                           type=str,
                           help="Path to the output file.")
    
    args = argparser.parse_args()
    Data_file = h5.File(glob.glob(args.datadir+"/*.h5")[0], "r")
    TrimPoints = args.MaxPoints
    max_hit = calculate_max_hits()

    ### Extracting Train set
    X = []
    Y = []
    for i in tqdm(range(Data_file["train_jet"].shape[0])):
        x = Data_file["train_jet"][i,:]
        x = get_point_clouds(x)
        x = trim(x,CUT_POINTS=TrimPoints)
        y = Data_file["train_meta"][i,2]
        X.append(x[None,:,:])
        Y.append(y)

    X_ = np.concatenate(X,0)
    Y_ = np.array(Y)
    
    ### Extracting Test set
    X_test = []
    Y_test = []
    for i in tqdm(range(Data_file["test_jet"].shape[0])):
        x = Data_file["test_jet"][i,:]
        x = get_point_clouds(x)
        x = trim(x,CUT_POINTS=TrimPoints)
        y = Data_file["test_meta"][i,2]
        X_test.append(x[None,:,:])
        Y_test.append(y)

    X_test_ = np.concatenate(X_test,0)
    Y_test_ = np.array(Y_test)

    ## Saving in H5 file
    file = h5.File(args.OutFile, "w")
    train_dataset = file.create_group("train_dataset")
    file["train_dataset"]["X"] = X_
    file["train_dataset"]["Y"] = Y_

    test_dataset = file.create_group("test_dataset")
    file["test_dataset"]["X"] = X_test_
    file["test_dataset"]["Y"] = Y_test_
    file.close()
