import nibabel as nib
import numpy as np
import os
import re

def probability_tissue_maps(tissue_maps):
    CSF = []
    WM = []
    GM = []
    for file in os.listdir(tissue_maps):
        if file.endswith(".nii.gz"):
            if file.find("CSF"):
                Data_list = []
                file_id = int(re.search(r'\d+', file)[0])
                imgpath = os.path.join(tissue_maps, file)
                img = nib.load(imgpath)
                data = img.get_fdata()
                Data_list.append(file_id)
                Data_list.append(data)
                CSF.append(Data_list)
            if file.find("GM"):
                Data_list = []
                file_id = int(re.search(r'\d+', file)[0])
                imgpath = os.path.join(tissue_maps, file)
                img = nib.load(imgpath)
                data = img.get_fdata()
                Data_list.append(file_id)
                Data_list.append(data)
                print(np.mean(data))
                GM.append(Data_list)
            if file.find("WM"):
                Data_list = []
                file_id = int(re.search(r'\d+', file)[0])
                imgpath = os.path.join(tissue_maps, file)
                img = nib.load(imgpath)
                data = img.get_fdata()
                Data_list.append(file_id)
                Data_list.append(data)
                print(np.mean(data))
                WM.append(Data_list)
    return CSF, WM, GM

def read_data(T1_scan, FLAIR_scan, T1_Lacunes_Correct, T1_Soft_Tissue):
    #Read all data into list
    T1_scan_data = []
    for file in os.listdir(T1_scan):
        if file.endswith(".nii.gz"):
            Data_list = []
            file_id = int(re.search(r'\d+', file)[0])
            imgpath = os.path.join(T1_scan, file)
            img = nib.load(imgpath)
            data = img.get_fdata()
            Data_list.append(file_id)
            Data_list.append(data)
            T1_scan_data.append(Data_list)

    #Read all FLAIR data into list
    FLAIR_scan_data = []
    for file in os.listdir(FLAIR_scan):
        if file.endswith(".nii.gz"):
            Data_list = []
            file_id = int(re.search(r'\d+', file)[0])
            imgpath = os.path.join(FLAIR_scan, file)
            img = nib.load(imgpath)
            data = img.get_fdata()
            Data_list.append(file_id)
            Data_list.append(data)
            FLAIR_scan_data.append(Data_list)

    #Lacune Exists
    Lacune_indicator_data = []
    for file in os.listdir(T1_Lacunes_Correct):
        if file.endswith(".nii.gz"):
            Data_list = []
            file_id = int(re.search(r'\d+', file)[0])
            imgpath = os.path.join(T1_Lacunes_Correct, file)
            img = nib.load(imgpath)
            data = img.get_fdata()
            Data_list.append(file_id)
            Data_list.append(data)
            Lacune_indicator_data.append(Data_list)

    Soft_tiss_data = []
    for file in os.listdir(T1_Soft_Tissue):
        if file.endswith(".nii.gz"):
            Data_list = []
            file_id = int(re.search(r'\d+', file)[0])
            imgpath = os.path.join(T1_Soft_Tissue, file)
            img = nib.load(imgpath)
            data = img.get_fdata()
            Data_list.append(file_id)
            Data_list.append(data)
            Soft_tiss_data.append(Data_list)

    return T1_scan_data, FLAIR_scan_data, Lacune_indicator_data, Soft_tiss_data
