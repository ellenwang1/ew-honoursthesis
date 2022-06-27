from turtle import shape
import nibabel as nib
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2
from PIL import Image
from matplotlib import cm
import pandas as pd
import imblearn
import re
import seaborn as sns
import math
import random
from numpy import linalg as LA


from 2_normalise import normalise
from 3_import_data import 

# Setting paths for different folders
FLAIR_scan = r'C:\Users\ellen\Documents\ew-honoursthesis\Data\forAudrey.tar\Normalised\FLAIRinT1space_withLacunes_35.tar'
T1_Lacunes_Incorrect = r'C:\Users\ellen\Documents\ew-honoursthesis\Data\forAudrey.tar\Original\lacune_T1space.tar'
T1_Lacunes_Correct = r'C:\Users\ellen\Documents\ew-honoursthesis\Data\forAudrey.tar\Original\lacune_T1space_JiyangCorrected20210920'
T1_scan = r'C:\Users\ellen\Documents\ew-honoursthesis\Data\forAudrey.tar\Normalised\T1_withLacunes_35.tar'
T1_Soft_Tissue = r'C:\Users\ellen\Documents\ew-honoursthesis\Data\forAudrey.tar\Normalised\T1softTiss_withLacunes_35.tar'
T1_Soft_Tissue_Mask = r'C:\Users\ellen\Documents\ew-honoursthesis\Data\forAudrey.tar\Original\T1softTissMask_withLacunes_35.tar'
T1_Soft_Tissue_Binary_Mask = r'C:\Users\ellen\Documents\ew-honoursthesis\Data\forAudrey.tar\Original\T1softTissMask_withLacunes_35_binary.tar'
tissue_maps = r'C:\Users\ellen\Downloads\tissue_prob_maps.tar\tissue_prob_maps'
mypath_og = r'C:\Users\ellen\Documents\ew-honoursthesis\Data\forAudrey.tar\Original'

# Prepare data
normalise(mypath_og)

# Import normalised data into relevant variables
CSF, WM, GM = probability_tissue_maps(tissue_maps)
T1_scan_data, FLAIR_scan_data, Lacune_indicator_data, Soft_tiss_data = read_data(T1_Scan, FLAIR_scan, T1_Lacunes_Correct, T1_Soft_Tissue)

# Sample Train - Lacunes
X_train_3D_lacune, Y_train_3D_lacune, Y_train_segment_3D_lacune, X_train_3D_nlacune, Y_train_3D_nlacune, Y_train_segment_3D_nlacune = def sample_lacunes(T1_Soft_Tissue_Binary_Mask, T1_scan_data, FLAIR_scan_data, Lacune_indicator_data, Soft_tiss_data)

# Sample Train - Non-Lacunes
X_train_3D_nlacune_func2, Y_train_3D_nlacune_func2, Y_train_segment_3D_nlacune_func2 = def non_lacune_sampling(T1_Soft_Tissue_Binary_Mask, T1_scan_data, FLAIR_scan_data, Lacune_indicator_data, Soft_tiss_data)

# Sample Test
X_test_3D_lacune, Y_test_3D_lacune, Y_test_segment_3D_lacune, X_test_3D_nlacune, Y_test_3D_nlacune, Y_test_segment_3D_nlacune = def test_sampling(T1_Soft_Tissue_Binary_Mask, T1_scan_data, FLAIR_scan_data, Lacune_indicator_data, Soft_tiss_data)

# Combine Train Test Results
X_train_3D_nlacune_all, Y_train_3D_nlacune_all, Y_train_segment_3D_nlacune_all, X_train, Y_train, Y_train_segment, Y_test_segment, Y_test, X_test = def train_test_combine(X_train_3D_lacune, Y_train_3D_lacune, Y_train_segment_3D_lacune, X_train_3D_nlacune, Y_train_3D_nlacune, Y_train_segment_3D_nlacune, X_train_3D_nlacune_func2, Y_train_3D_nlacune_func2, Y_train_segment_3D_nlacune_func2, X_test_3D_lacune, Y_test_3D_lacune, Y_test_segment_3D_lacune, X_test_3D_nlacune, Y_test_3D_nlacune, Y_test_segment_3D_nlacune)