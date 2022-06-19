# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ### Import Libararies

# %%
pip install opencv-python;


# %%
pip install imbalanced-learn;


# %%
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

np.set_printoptions(threshold=sys.maxsize)
mypath_og = r'C:\Users\ellen\Documents\ew-honoursthesis\Data\forAudrey.tar\Original'
mypath_norm = r'C:\Users\ellen\Documents\ew-honoursthesis\Data\forAudrey.tar\Normalised'

# %% [markdown]
# ### Pre-Normalisation Data Spread

# %%
# Histogram of all values
img = nib.load(r'C:\Users\ellen\Documents\ew-honoursthesis\Data\forAudrey.tar\Original\FLAIRinT1space_withLacunes_35.tar\r0046_tp2_flair.nii.gz')
data = img.get_fdata()
plt.hist(data.ravel(), bins=50)


# %%
np.where(data < 0)


# %%
plt.imshow(data[31].T, cmap="gray", origin="lower")
plt.show()


# %%
plt.imshow(data[29].T, cmap="gray", origin="lower")
plt.show()

# %% [markdown]
# ### Normalise across 3 dimensions (Outlier removal)

# %%
def show_slices(slices):
   fig, axes = plt.subplots(1, len(slices))
   for i, slice in enumerate(slices):
       axes[i].imshow(slice.T, cmap="gray", origin="lower")
    
for root, dirs, files in os.walk(mypath_og, topdown=False):
    for file in files:
        if file.endswith(".nii.gz"):
            imgpath = os.path.join(root, file)
            img = nib.load(imgpath)
            data = img.get_fdata()
            data = abs(data)
            cutoff = np.percentile(data, 99)
            data[data >= cutoff] = cutoff
            #workaround to (256, 256, 180)
            if data.shape == (256, 256, 190):
                data = data[:, :, 5:-5]

            # normalizing per channel data:
            #data = (data - np.min(data)) / (np.max(data) - np.min(data)) #(np.percentile(data, 98) - np.min(data))
            data = (data - np.min(data)) / (np.max(data) - np.min(data)) #(np.percentile(data, 98) - np.min(data))

            # replace all nans with 0
            data[np.isnan(data)] = 0
            
            #save to new nib file
            #print(data.shape)
            norm_img = nib.Nifti1Image(data, img.affine, img.header)
            norm_root = root.replace("Original", "Normalised")
            norm_imgpath_file = os.path.join(norm_root, file)
            nib.save(norm_img, norm_imgpath_file)

# %% [markdown]
# ### Post Normalisation Data Spread

# %%
# Histogram of all values
img = nib.load(r'C:\Users\ellen\Documents\ew-honoursthesis\Data\forAudrey.tar\Normalised\FLAIRinT1space_withLacunes_35.tar\r0046_tp2_flair.nii.gz')
data = img.get_fdata()
print(data.shape)
plt.hist(data.ravel(), bins = 50)


# %%
plt.imshow(data[29].T, cmap="gray", origin="lower")
plt.show()


# %%
# Histogram of all values
img = nib.load(r'C:\Users\ellen\Documents\ew-honoursthesis\Data\forAudrey.tar\Original\lacune_T1space_JiyangCorrected20210920\0046_lacuneT1space.nii.gz')
data = img.get_fdata()
x,y,z = data.nonzero()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, -z, zdir='z', c= 'red', s= 10)
ax.set_xlim(50, 150)
ax.set_ylim(150, 256)
ax.set_zlim(-50, -80)
plt.savefig("demo.png")


# %%
plt.imshow(data[140, 145:155, 90:100], cmap="gray", origin="lower")
plt.show()


# %%
data[data > 0]


# %%
len(data[140:150, 145:155, 90:100][data[140:150, 145:155, 90:100] > 0]) > 50

# %% [markdown]
# ### Setting paths for different folders

# %%
# Setting paths for different folders
FLAIR_scan = r'C:\Users\ellen\Documents\ew-honoursthesis\Data\forAudrey.tar\Normalised\FLAIRinT1space_withLacunes_35.tar'
T1_Lacunes_Incorrect = r'C:\Users\ellen\Documents\ew-honoursthesis\Data\forAudrey.tar\Original\lacune_T1space.tar'
T1_Lacunes_Correct = r'C:\Users\ellen\Documents\ew-honoursthesis\Data\forAudrey.tar\Original\lacune_T1space_JiyangCorrected20210920'
T1_scan = r'C:\Users\ellen\Documents\ew-honoursthesis\Data\forAudrey.tar\Normalised\T1_withLacunes_35.tar'
T1_Soft_Tissue = r'C:\Users\ellen\Documents\ew-honoursthesis\Data\forAudrey.tar\Normalised\T1softTiss_withLacunes_35.tar'
T1_Soft_Tissue_Mask = r'C:\Users\ellen\Documents\ew-honoursthesis\Data\forAudrey.tar\Original\T1softTissMask_withLacunes_35.tar'
T1_Soft_Tissue_Binary_Mask = r'C:\Users\ellen\Documents\ew-honoursthesis\Data\forAudrey.tar\Original\T1softTissMask_withLacunes_35_binary.tar'

# %% [markdown]
# ### Setting up data 

# %%
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


# %%
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


# %%
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


# %%
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


# %%
sum = 0
for brain in range(0,35):
    print(len(Lacune_indicator_data[brain][1][Lacune_indicator_data[brain][1] > 0]))
    sum += len(Lacune_indicator_data[brain][1][Lacune_indicator_data[brain][1] > 0])
    
print(sum)


# %%
T1_scan_data[0][1][46:66, 126, 100:120]


# %%
Lacune_indicator_data[1][1][100,40,150]

# %% [markdown]
# ### Where does soft tissue start?

# %%
def show_slices(slices):
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")

for file in os.listdir(T1_Soft_Tissue):
    if file.endswith(".nii.gz"):
        file_id = int(re.search(r'\d+', file)[0])
        imgpath = os.path.join(T1_Soft_Tissue, file)
        img = nib.load(imgpath)
        data = img.get_fdata()
        slice_0 = data[50, :, :]
        slice_1 = data[:, 70, :]
        slice_2 = data[:, :, 10]
        show_slices([slice_0, slice_1, slice_2])
        plt.suptitle("Center slices for EPI image")  

# %% [markdown]
# ### Sampling within the brain 3D

# %%
# Lacune not as centred, random sampling all around brain
X_train_3D_lacune = []
X_test_3D_lacune = []
Y_train_3D_lacune = []
Y_test_3D_lacune = []
Y_train_segment_3D_lacune = []
Y_test_segment_3D_lacune = []

brain_image = 0
for file in os.listdir(T1_Soft_Tissue_Binary_Mask):
    if file.endswith(".nii.gz"):
        file_id = int(re.search(r'\d+', file)[0])
        imgpath = os.path.join(T1_Soft_Tissue_Binary_Mask, file)
        img = nib.load(imgpath)
        data = img.get_fdata()
        data = data.astype(np.uint8)  # converting array of ints to floats
        T1_data_scans = T1_scan_data[brain_image][1]
        FLAIR_data_scans = FLAIR_scan_data[brain_image][1]
        SoftTiss = Soft_tiss_data[brain_image][1]
        
        #Sample lacunes
        for x in range(0, data.shape[0]):
            for y in range(0, data.shape[1]):
                for z in range(0, data.shape[2]):
                    #filter for soft tissue
                    if (x < 50) | (y < 70) | (z < 10) | (x > 200) | (y > 210) | (z > 165):
                        next
                    else:
                        brain_values = []
                        brain_values.append(file_id)
                        brain_values.append(x)
                        brain_values.append(y)
                        brain_values.append(z)

                        patch_3D_T1 = T1_data_scans[x-15:x+15, y-15:y+15, z-15:z+15]
                        brain_values.append(patch_3D_T1)

                        patch_3D_FLAIR = FLAIR_data_scans[x-15:x+15, y-15:y+15, z-15:z+15]
                        brain_values.append(patch_3D_FLAIR)

                        patch_3D_softtiss_binary = data[x-15:x+15, y-15:y+15, z-15:z+15]
                        brain_values.append(patch_3D_softtiss_binary)

                        patch_3D_softtiss = SoftTiss[x-15:x+15, y-15:y+15, z-15:z+15]
                        brain_values.append(patch_3D_softtiss)
                        if brain_image <= 24:
                            if (Lacune_indicator_data[brain_image][1][x,y,z] == 1):
                                X_train_3D_lacune.append(brain_values)
                                lacune_binary = Lacune_indicator_data[brain_image][1][x-15:x+15, y-15:y+15, z-15:z+15]
                                Y_train_3D_lacune.append(1)
                                Y_train_segment_3D_lacune.append(lacune_binary)
                        else:
                            if (Lacune_indicator_data[brain_image][1][x,y,z] == 1):
                                X_test_3D_lacune.append(brain_values)
                                lacune_binary = Lacune_indicator_data[brain_image][1][x-15:x+15, y-15:y+15, z-15:z+15]
                                Y_test_3D_lacune.append(1)
                                Y_test_segment_3D_lacune.append(lacune_binary)
                        
        brain_image += 1
        print(brain_image)


# %%
Y_test_3D.count(1)


# %%
Y_train_3D.index(1)


# %%
len(X_train_3D)


# %%
len(Y_test_3D)


# %%
Y_train_segment_3D[1]


# %%
x,y,z = Y_train_segment_3D[3].nonzero()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, -z, zdir='z', c= 'red', s= 10)
ax.set_xlim(0, 30)
ax.set_ylim(0, 30)
ax.set_zlim(-0, -30)
plt.savefig("demo.png")

# %% [markdown]
# ### Sampling Non-Lacunes 3D

# %%
# Lacune not as centred, random sampling all around brain
X_train_3D_nlacune = []
X_test_3D_nlacune = []
Y_train_3D_nlacune = []
Y_test_3D_nlacune = []
Y_train_segment_3D_nlacune = []
Y_test_segment_3D_nlacune = []

brain_image = 0
for file in os.listdir(T1_Soft_Tissue_Binary_Mask):
    if file.endswith(".nii.gz"):
        file_id = int(re.search(r'\d+', file)[0])
        imgpath = os.path.join(T1_Soft_Tissue_Binary_Mask, file)
        img = nib.load(imgpath)
        data = img.get_fdata()
        data = data.astype(np.uint8)  # converting array of ints to floats
        T1_data_scans = T1_scan_data[brain_image][1]
        FLAIR_data_scans = FLAIR_scan_data[brain_image][1]
        SoftTiss = Soft_tiss_data[brain_image][1]
        Lacune_indicator = Lacune_indicator_data[brain_image][1]
        
        #Sample lacunes
        for x in range(0, data.shape[0], 5):
            for y in range(0, data.shape[1], 5):
                for z in range(0, data.shape[2], 5):
                    #filter for soft tissue
                    if (x < 50) | (y < 70) | (z < 10) | (x > 200) | (y > 210) | (z > 165):
                        next
                    else:
                        brain_values = []
                        brain_values.append(file_id)
                        brain_values.append(x)
                        brain_values.append(y)
                        brain_values.append(z)

                        patch_3D_T1 = T1_data_scans[x-15:x+15, y-15:y+15, z-15:z+15]
                        brain_values.append(patch_3D_T1)

                        patch_3D_FLAIR = FLAIR_data_scans[x-15:x+15, y-15:y+15, z-15:z+15]
                        brain_values.append(patch_3D_FLAIR)

                        patch_3D_softtiss_binary = data[x-15:x+15, y-15:y+15, z-15:z+15]
                        brain_values.append(patch_3D_softtiss_binary)

                        patch_3D_softtiss = SoftTiss[x-15:x+15, y-15:y+15, z-15:z+15]
                        brain_values.append(patch_3D_softtiss)
                        if brain_image <= 24:
                            # No lacune exists in the 3D patch, add to train set
                            print(Lacune_indicator[x-15:x+15, y-15:y+15, z-15:z+15])
                            print(len(Lacune_indicator[x-15:x+15, y-15:y+15, z-15:z+15] > 0))
                            if len(Lacune_indicator_data[brain_image][1][x-15:x+15, y-15:y+15, z-15:z+15] > 0) == 0:
                                next
                            else:
                                X_train_3D_nlacune.append(brain_values)
                                lacune_binary = Lacune_indicator_data[brain_image][1][x-15:x+15, y-15:y+15, z-15:z+15]
                                Y_train_3D_nlacune.append(0)
                                Y_train_segment_3D_nlacune.append(lacune_binary)
                        else:
                            if any(1 in sublist for sublist in Lacune_indicator_data[brain_image][1][x-15:x+15, y-15:y+15, z-15:z+15]):
                                next
                            else:
                                X_test_3D_nlacune.append(brain_values)
                                lacune_binary = Lacune_indicator_data[brain_image][1][x-15:x+15, y-15:y+15, z-15:z+15]
                                Y_test_3D_nlacune.append(0)
                                Y_test_segment_3D_nlacune.append(lacune_binary)     
        brain_image += 1
        print(brain_image)

# %% [markdown]
# ### Sampling within the brain (2D candidate)

# %%
# Save T1 soft tissue images to X array
X_train_T1 = []
X_train_FLAIR = []
X_test_T1 = []
X_test_FLAIR = []
Y_train = []
Y_test = []
Y_train_segment = []
Y_test_segment = []

brain_image = 0
for file in os.listdir(T1_Soft_Tissue_Binary_Mask):
    if file.endswith(".nii.gz"):
        file_id = int(re.search(r'\d+', file)[0])
        imgpath = os.path.join(T1_Soft_Tissue_Binary_Mask, file)
        img = nib.load(imgpath)
        data = img.get_fdata()
        data = data.astype(np.uint8)  # converting array of ints to floats
        T1_data_scans = T1_scan_data[brain_image][1]
        FLAIR_data_scans = FLAIR_scan_data[brain_image][1]
        for x in range(0, data.shape[0], 5):
            for y in range(0, data.shape[1], 5):
                for z in range(0, data.shape[2], 5):
                    #filter for soft tissue
                    if (x < 25) | (y < 25) | (z < 25) | (data[x][y][z] == 0) | (x > 231) | (y > 231) | (z > 155):
                        next
                    else:
                        brain_values = []
                        brain_values.append(file_id)
                        brain_values.append(x)
                        brain_values.append(y)
                        brain_values.append(z)
                        
                        
                        patch_T1 = T1_data_scans[x-25:x+25, y, z-25:z+25]
                        brain_values_T1 = brain_values
                        brain_values_T1.append(patch_T1)
                        
                        patch_FLAIR = FLAIR_data_scans[x-25:x+25, y, z-25:z+25]
                        brain_values_FLAIR = brain_values
                        brain_values_FLAIR.append(patch_FLAIR)
                        
                        #Append train data values
                        if brain_image <= 24:
                            X_train_T1.append(brain_values_T1)
                            X_train_FLAIR.append(brain_values_FLAIR)

                            #Lacune binary 
                            lacune_binary = Lacune_indicator_data[brain_image][1][x-25:x+25, y, z-25:z+25]

                            #Append if or if there isn't a lacune in this location
                            if any(1 in sublist for sublist in Lacune_indicator_data[brain_image][1][x-25:x+25, y, z-25:z+25]):
                                Y_train.append(1)
                                Y_train_segment.append(lacune_binary)
                            else:
                                Y_train.append(0)
                                Y_train_segment.append(lacune_binary)
                        
                        #Append test data values
                        else:
                            X_test_T1.append(brain_values_T1)
                            X_test_FLAIR.append(brain_values_FLAIR)

                            #Lacune binary 
                            lacune_binary = Lacune_indicator_data[brain_image][1][x-25:x+25, y, z-25:z+25]

                            #Append if or if there isn't a lacune in this location
                            if any(1 in sublist for sublist in Lacune_indicator_data[brain_image][1][x-25:x+25, y, z-25:z+25]):
                                Y_test.append(1)
                                Y_test_segment.append(lacune_binary)
                            else:
                                Y_test.append(0)
                                Y_test_segment.append(lacune_binary)
                        
        brain_image += 1
        print(brain_image)


# %%
Y_train.count(1)


# %%
len(X_train_T1)


# %%
Y_train.index(1)

# %% [markdown]
# ### Keep images only

# %%
X_train_T1_im = []
X_train_FLAIR_im = []
X_test_T1_im = []
X_test_FLAIR_im = []

for sample in range(len(X_train_T1)):
    X_train_T1_im.append(X_train_T1[sample][4])
    X_train_FLAIR_im.append(X_train_FLAIR[sample][4])

for sample in range(len(X_test_T1)):
    X_test_T1_im.append(X_test_T1[sample][4])
    X_test_FLAIR_im.append(X_test_FLAIR[sample][4])


# %%
len(X_train_T1_im)


# %%
X_train_T1_im[0]

# %% [markdown]
# ### Re-Sampling (Lower Minority Count, Increase Majority Count)
# %% [markdown]
# ### Create Features

# %%
#Minimum T1 value
Min_T1 = []
for obj in len(X_train_3D):
    Min_T1.append(min(X_train_3D[obj][4])


# %%
#Maximum T1 value
Max_T1 = []
for obj in len(X_train_3D):
    Max_T1.append(max(X_train_3D[obj][4])


# %%
#Minimum FLAIR value
Min_FLAIR = []
for obj in len(X_train_3D):
    Min_FLAIR.append(min(X_train_3D[obj][5])


# %%
#Maximum FLAIR value
Max_FLAIR = []
for obj in len(X_train_3D):
    Max_FLAIR.append(max(X_train_3D[obj][5])


# %%
# %soft tissue in 3D matrix
Soft_Tiss_PCt = []
for obj in len(X_train_3D):
    Soft_Tiss_PCt.append(len(X_train_3D[obj][6] > 0))


# %%
# Location
X = X_train_3D[1]
Y = X_train_3D[2]
Z = X_train_3D[3]


# %%
# Clustering


# %%
#Taken from Uchiyama
filterSize =(16, 16)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                   filterSize)
  
# Reading the image named 'input.jpg'
input_image = X_train_T1_im[1188]
  
# Applying the Black-Hat operation
tophat_img = cv2.morphologyEx(input_image, 
                              cv2.MORPH_BLACKHAT,
                              kernel)

plt.imshow(tophat_img.T, cmap="gray", origin="lower")
plt.show()


# %%
np.amax(input_image)


# %%
np.amin(input_image)


# %%
np.amax(tophat_img)


# %%
np.amin(tophat_img)


# %%
plt.imshow(input_image.T, cmap="gray", origin="lower")
plt.show()


# %%
#Taken from Uchiyama
filterSize =(16, 16)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                   filterSize)
  
# Reading the image named 'input.jpg'
input_image = X_train_T1_im[900]
  
# Applying the Black-Hat operation
tophat_img = cv2.morphologyEx(input_image, 
                              cv2.MORPH_TOPHAT,
                              kernel)

plt.imshow(tophat_img.T, cmap="gray", origin="lower")
plt.show()


# %%
np.amax(input_image)


# %%
np.amin(input_image)


# %%
np.amax(tophat_img)


# %%
np.amin(tophat_img)


# %%
plt.imshow(input_image.T, cmap="gray", origin="lower")
plt.show()

# %% [markdown]
# ### Feature 1: Top-hat transform, get minimum
# %% [markdown]
# ### Feature 2: Top-hat transform, get average of surroundings

# %%



# %%


# %% [markdown]
# ### Sampling within the brain (pixels)

# %%
print(data.shape[1])


# %%



# %%



# %%



# %%



