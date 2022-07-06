import nibabel as nib
import re
import os
import numpy as np

def sample_lacunes(CSF, GM, WM, T1_Soft_Tissue_Binary_Mask, T1_scan_data, FLAIR_scan_data, Lacune_indicator_data, Soft_tiss_data):
    # Lacune not as centred, random sampling all around brain
    X_train_3D_lacune = []
    Y_train_3D_lacune = []
    Y_train_segment_3D_lacune = []
    X_train_3D_nlacune = []
    Y_train_3D_nlacune = []
    Y_train_segment_3D_nlacune = []
    
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
            Lacune_data = Lacune_indicator_data[brain_image][1]
            CSF_data = CSF[brain_image][1]
            GM_data = GM[brain_image][1]
            WM_data = WM[brain_image][1]

            #Sample lacunes
            for x in range(0, data.shape[0]):
                for y in range(0, data.shape[1]):
                    for z in range(0, data.shape[2]):
                        #filter for soft tissue
                        if (x < 50) | (y < 70) | (z < 15) | (x > 200) | (y > 210) | (z > 165) | (T1_data_scans[x,y,z] == 0) | (FLAIR_data_scans[x,y,z] == 0):
                            next
                        else:
                    
                            if brain_image <= 24:
                                if (Lacune_data[x,y,z] == 1) & (sum(sum(sum(Lacune_data[x-1:x+1, y-1:y+1, z-1:z+1]))) >= 4):
                                    
                                    brain_values = []
                                    brain_values_nlacune = []
                                    brain_values_rot90x = []
                                    brain_values_rot90y = []
                                    brain_values_rot90z = []
                                    brain_values_rot180x = []
                                    brain_values_rot180y = []
                                    brain_values_rot180z = []
                                    brain_values_rot270x = []
                                    brain_values_rot270y = []
                                    brain_values_rot270z = []

                                    brain_values.append(file_id)
                                    brain_values_nlacune.append(file_id)
                                    brain_values_rot90x.append(file_id)
                                    brain_values_rot90y.append(file_id)
                                    brain_values_rot90z.append(file_id)
                                    brain_values_rot180x.append(file_id)
                                    brain_values_rot180y.append(file_id)
                                    brain_values_rot180z.append(file_id)
                                    brain_values_rot270x.append(file_id)
                                    brain_values_rot270y.append(file_id)
                                    brain_values_rot270z.append(file_id)

                                    brain_values.append(x)
                                    brain_values_nlacune.append(256-x)
                                    brain_values_rot90x.append(x)
                                    brain_values_rot90y.append(x)
                                    brain_values_rot90z.append(x)
                                    brain_values_rot180x.append(x)
                                    brain_values_rot180y.append(x)
                                    brain_values_rot180z.append(x)
                                    brain_values_rot270x.append(x)
                                    brain_values_rot270y.append(x)
                                    brain_values_rot270z.append(x)

                                    brain_values.append(y)
                                    brain_values_nlacune.append(y)
                                    brain_values_rot90x.append(y)
                                    brain_values_rot90y.append(y)
                                    brain_values_rot90z.append(y)
                                    brain_values_rot180x.append(y)
                                    brain_values_rot180y.append(y)
                                    brain_values_rot180z.append(y)
                                    brain_values_rot270x.append(y)
                                    brain_values_rot270y.append(y)
                                    brain_values_rot270z.append(y)

                                    brain_values.append(z)
                                    brain_values_nlacune.append(z)
                                    brain_values_rot90x.append(z)
                                    brain_values_rot90y.append(z)
                                    brain_values_rot90z.append(z)
                                    brain_values_rot180x.append(z)
                                    brain_values_rot180y.append(z)
                                    brain_values_rot180z.append(z)
                                    brain_values_rot270x.append(z)
                                    brain_values_rot270y.append(z)
                                    brain_values_rot270z.append(z)

                                    patch_3D_T1 = T1_data_scans[x-10:x+10, y-10:y+10, z-10:z+10]
                                    patch_3D_T1_nlacune = T1_data_scans[(256-x)-10:(256-x)+10, y-10:y+10, z-10:z+10]
                                    brain_values.append(patch_3D_T1)
                                    brain_values_nlacune.append(patch_3D_T1_nlacune)
                                    brain_values_rot90x.append(np.rot90(patch_3D_T1, k=1, axes=(0,1)))
                                    brain_values_rot90y.append(np.rot90(patch_3D_T1, k=1, axes=(1,2)))
                                    brain_values_rot90z.append(np.rot90(patch_3D_T1, k=1, axes=(0,2)))
                                    brain_values_rot180x.append(np.rot90(patch_3D_T1, k=2, axes=(0,1)))
                                    brain_values_rot180y.append(np.rot90(patch_3D_T1, k=2, axes=(1,2)))
                                    brain_values_rot180z.append(np.rot90(patch_3D_T1, k=2, axes=(0,2)))
                                    brain_values_rot270x.append(np.rot90(patch_3D_T1, k=3, axes=(0,1)))
                                    brain_values_rot270y.append(np.rot90(patch_3D_T1, k=3, axes=(1,2)))
                                    brain_values_rot270z.append(np.rot90(patch_3D_T1, k=3, axes=(0,2)))


                                    patch_3D_FLAIR = FLAIR_data_scans[x-10:x+10, y-10:y+10, z-10:z+10]
                                    patch_3D_FLAIR_nlacune = FLAIR_data_scans[(256-x)-10:(256-x)+10, y-10:y+10, z-10:z+10]
                                    brain_values.append(patch_3D_FLAIR)
                                    brain_values_nlacune.append(patch_3D_FLAIR_nlacune)
                                    brain_values_rot90x.append(np.rot90(patch_3D_FLAIR, k=1, axes=(0,1)))
                                    brain_values_rot90y.append(np.rot90(patch_3D_FLAIR, k=1, axes=(1,2)))
                                    brain_values_rot90z.append(np.rot90(patch_3D_FLAIR, k=1, axes=(0,2)))
                                    brain_values_rot180x.append(np.rot90(patch_3D_FLAIR, k=2, axes=(0,1)))
                                    brain_values_rot180y.append(np.rot90(patch_3D_FLAIR, k=2, axes=(1,2)))
                                    brain_values_rot180z.append(np.rot90(patch_3D_FLAIR, k=2, axes=(0,2)))
                                    brain_values_rot270x.append(np.rot90(patch_3D_FLAIR, k=3, axes=(0,1)))
                                    brain_values_rot270y.append(np.rot90(patch_3D_FLAIR, k=3, axes=(1,2)))
                                    brain_values_rot270z.append(np.rot90(patch_3D_FLAIR, k=3, axes=(0,2)))

                                    patch_3D_softtiss_binary = data[x-10:x+10, y-10:y+10, z-10:z+10]
                                    patch_3D_softtiss_binary_nlacune = data[(256-x)-10:(256-x)+10, y-10:y+10, z-10:z+10]
                                    brain_values.append(patch_3D_softtiss_binary)
                                    brain_values_nlacune.append(patch_3D_softtiss_binary_nlacune)
                                    brain_values_rot90x.append(np.rot90(patch_3D_softtiss_binary, k=1, axes=(0,1)))
                                    brain_values_rot90y.append(np.rot90(patch_3D_softtiss_binary, k=1, axes=(1,2)))
                                    brain_values_rot90z.append(np.rot90(patch_3D_softtiss_binary, k=1, axes=(0,2)))
                                    brain_values_rot180x.append(np.rot90(patch_3D_softtiss_binary, k=2, axes=(0,1)))
                                    brain_values_rot180y.append(np.rot90(patch_3D_softtiss_binary, k=2, axes=(1,2)))
                                    brain_values_rot180z.append(np.rot90(patch_3D_softtiss_binary, k=2, axes=(0,2)))
                                    brain_values_rot270x.append(np.rot90(patch_3D_softtiss_binary, k=3, axes=(0,1)))
                                    brain_values_rot270y.append(np.rot90(patch_3D_softtiss_binary, k=3, axes=(1,2)))
                                    brain_values_rot270z.append(np.rot90(patch_3D_softtiss_binary, k=3, axes=(0,2)))

                                    patch_3D_softtiss = SoftTiss[x-10:x+10, y-10:y+10, z-10:z+10]
                                    patch_3D_softtiss_nlacune = SoftTiss[(256-x)-10:(256-x)+10, y-10:y+10, z-10:z+10]
                                    brain_values.append(patch_3D_softtiss)
                                    brain_values_nlacune.append(patch_3D_softtiss_nlacune)
                                    brain_values_rot90x.append(np.rot90(patch_3D_softtiss, k=1, axes=(0,1)))
                                    brain_values_rot90y.append(np.rot90(patch_3D_softtiss, k=1, axes=(1,2)))
                                    brain_values_rot90z.append(np.rot90(patch_3D_softtiss, k=1, axes=(0,2)))
                                    brain_values_rot180x.append(np.rot90(patch_3D_softtiss, k=2, axes=(0,1)))
                                    brain_values_rot180y.append(np.rot90(patch_3D_softtiss, k=2, axes=(1,2)))
                                    brain_values_rot180z.append(np.rot90(patch_3D_softtiss, k=2, axes=(0,2)))
                                    brain_values_rot270x.append(np.rot90(patch_3D_softtiss, k=3, axes=(0,1)))
                                    brain_values_rot270y.append(np.rot90(patch_3D_softtiss, k=3, axes=(1,2)))
                                    brain_values_rot270z.append(np.rot90(patch_3D_softtiss, k=3, axes=(0,2)))
                                    
                                    patch_3D_CSF = CSF_data[x-10:x+10, y-10:y+10, z-10:z+10]
                                    patch_3D_CSF_nlacune = CSF_data[(256-x)-10:(256-x)+10, y-10:y+10, z-10:z+10]
                                    brain_values.append(patch_3D_CSF)
                                    brain_values_nlacune.append(patch_3D_CSF_nlacune)
                                    brain_values_rot90x.append(np.rot90(patch_3D_CSF, k=1, axes=(0,1)))
                                    brain_values_rot90y.append(np.rot90(patch_3D_CSF, k=1, axes=(1,2)))
                                    brain_values_rot90z.append(np.rot90(patch_3D_CSF, k=1, axes=(0,2)))
                                    brain_values_rot180x.append(np.rot90(patch_3D_CSF, k=2, axes=(0,1)))
                                    brain_values_rot180y.append(np.rot90(patch_3D_CSF, k=2, axes=(1,2)))
                                    brain_values_rot180z.append(np.rot90(patch_3D_CSF, k=2, axes=(0,2)))
                                    brain_values_rot270x.append(np.rot90(patch_3D_CSF, k=3, axes=(0,1)))
                                    brain_values_rot270y.append(np.rot90(patch_3D_CSF, k=3, axes=(1,2)))
                                    brain_values_rot270z.append(np.rot90(patch_3D_CSF, k=3, axes=(0,2)))
                                    
                                    patch_3D_WM = WM_data[x-10:x+10, y-10:y+10, z-10:z+10]
                                    patch_3D_WM_nlacune = WM_data[(256-x)-10:(256-x)+10, y-10:y+10, z-10:z+10]
                                    brain_values.append(patch_3D_WM)
                                    brain_values_nlacune.append(patch_3D_WM_nlacune)
                                    brain_values_rot90x.append(np.rot90(patch_3D_WM, k=1, axes=(0,1)))
                                    brain_values_rot90y.append(np.rot90(patch_3D_WM, k=1, axes=(1,2)))
                                    brain_values_rot90z.append(np.rot90(patch_3D_WM, k=1, axes=(0,2)))
                                    brain_values_rot180x.append(np.rot90(patch_3D_WM, k=2, axes=(0,1)))
                                    brain_values_rot180y.append(np.rot90(patch_3D_WM, k=2, axes=(1,2)))
                                    brain_values_rot180z.append(np.rot90(patch_3D_WM, k=2, axes=(0,2)))
                                    brain_values_rot270x.append(np.rot90(patch_3D_WM, k=3, axes=(0,1)))
                                    brain_values_rot270y.append(np.rot90(patch_3D_WM, k=3, axes=(1,2)))
                                    brain_values_rot270z.append(np.rot90(patch_3D_WM, k=3, axes=(0,2)))
                                    
                                    patch_3D_GM = GM_data[x-10:x+10, y-10:y+10, z-10:z+10]
                                    patch_3D_GM_nlacune = GM_data[(256-x)-10:(256-x)+10, y-10:y+10, z-10:z+10]
                                    brain_values.append(patch_3D_GM)
                                    brain_values_nlacune.append(patch_3D_GM_nlacune)
                                    brain_values_rot90x.append(np.rot90(patch_3D_GM, k=1, axes=(0,1)))
                                    brain_values_rot90y.append(np.rot90(patch_3D_GM, k=1, axes=(1,2)))
                                    brain_values_rot90z.append(np.rot90(patch_3D_GM, k=1, axes=(0,2)))
                                    brain_values_rot180x.append(np.rot90(patch_3D_GM, k=2, axes=(0,1)))
                                    brain_values_rot180y.append(np.rot90(patch_3D_GM, k=2, axes=(1,2)))
                                    brain_values_rot180z.append(np.rot90(patch_3D_GM, k=2, axes=(0,2)))
                                    brain_values_rot270x.append(np.rot90(patch_3D_GM, k=3, axes=(0,1)))
                                    brain_values_rot270y.append(np.rot90(patch_3D_GM, k=3, axes=(1,2)))
                                    brain_values_rot270z.append(np.rot90(patch_3D_GM, k=3, axes=(0,2)))

                                    #brain_rotation
                                    lacune_binary = Lacune_data[x-10:x+10, y-10:y+10, z-10:z+10] 
                                    lacune_binary_nlacune = Lacune_data[(256-x)-10:(256-x)+10, y-10:y+10, z-10:z+10]
                                    lacune_binary_rot90x = np.rot90(lacune_binary, k=1, axes=(0,1))
                                    lacune_binary_rot90y = np.rot90(lacune_binary, k=1, axes=(1,2))
                                    lacune_binary_rot90z = np.rot90(lacune_binary, k=1, axes=(0,2))
                                    lacune_binary_rot180x = np.rot90(lacune_binary, k=2, axes=(0,1))
                                    lacune_binary_rot180y = np.rot90(lacune_binary, k=2, axes=(1,2))
                                    lacune_binary_rot180z = np.rot90(lacune_binary, k=2, axes=(0,2))
                                    lacune_binary_rot270x = np.rot90(lacune_binary, k=3, axes=(0,1))
                                    lacune_binary_rot270y = np.rot90(lacune_binary, k=3, axes=(1,2))
                                    lacune_binary_rot270z = np.rot90(lacune_binary, k=3, axes=(0,2))

                                    X_train_3D_lacune.append(brain_values)
                                    X_train_3D_nlacune.append(brain_values_nlacune)
                                    X_train_3D_lacune.append(brain_values_rot90x)
                                    X_train_3D_lacune.append(brain_values_rot90y)
                                    X_train_3D_lacune.append(brain_values_rot90z)
                                    X_train_3D_lacune.append(brain_values_rot180x)
                                    X_train_3D_lacune.append(brain_values_rot180y)
                                    X_train_3D_lacune.append(brain_values_rot180z)
                                    X_train_3D_lacune.append(brain_values_rot270x)
                                    X_train_3D_lacune.append(brain_values_rot270y)
                                    X_train_3D_lacune.append(brain_values_rot270z)
                                    
                                    Y_train_3D_lacune.append(1)
                                    Y_train_3D_nlacune.append(0)
                                    Y_train_3D_lacune.append(1)
                                    Y_train_3D_lacune.append(1)
                                    Y_train_3D_lacune.append(1)
                                    Y_train_3D_lacune.append(1)
                                    Y_train_3D_lacune.append(1)
                                    Y_train_3D_lacune.append(1)
                                    Y_train_3D_lacune.append(1)
                                    Y_train_3D_lacune.append(1)
                                    Y_train_3D_lacune.append(1)
                                    
                                    Y_train_segment_3D_lacune.append(lacune_binary)
                                    Y_train_segment_3D_nlacune.append(lacune_binary_nlacune)
                                    Y_train_segment_3D_lacune.append(lacune_binary_rot90x)
                                    Y_train_segment_3D_lacune.append(lacune_binary_rot90y)
                                    Y_train_segment_3D_lacune.append(lacune_binary_rot90z)
                                    Y_train_segment_3D_lacune.append(lacune_binary_rot180x)
                                    Y_train_segment_3D_lacune.append(lacune_binary_rot180y)
                                    Y_train_segment_3D_lacune.append(lacune_binary_rot180z)
                                    Y_train_segment_3D_lacune.append(lacune_binary_rot270x)
                                    Y_train_segment_3D_lacune.append(lacune_binary_rot270y)
                                    Y_train_segment_3D_lacune.append(lacune_binary_rot270z)
                                    
                                    print("appended a sample")
                            
            brain_image += 1
            print(brain_image)
    return X_train_3D_lacune, Y_train_3D_lacune, Y_train_segment_3D_lacune, X_train_3D_nlacune, Y_train_3D_nlacune, Y_train_segment_3D_nlacune

def non_lacune_sampling(CSF, GM, WM, T1_Soft_Tissue_Binary_Mask, T1_scan_data, FLAIR_scan_data, Lacune_indicator_data, Soft_tiss_data):
    # Lacune not as centred, random sampling all around brain
    X_train_3D_nlacune_func2 = []
    Y_train_3D_nlacune_func2 = []
    Y_train_segment_3D_nlacune_func2 = []

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
            Lacune_data = Lacune_indicator_data[brain_image][1]
            CSF_data = CSF[brain_image][1]
            GM_data = GM[brain_image][1]
            WM_data = WM[brain_image][1]
            
            #Sample lacunes
            sampled_list_x = np.random.choice(data.shape[0], 2500)
            sampled_list_y = np.random.choice(data.shape[1], 2500)
            sampled_list_z = np.random.choice(data.shape[2], 2500)
            for x,y,z in set(list(zip(sampled_list_x, sampled_list_y,sampled_list_z))):
                #filter for soft tissue
                if (x < 50) | (y < 70) | (z < 15) | (x > 200) | (y > 210) | (z > 165) |  (T1_data_scans[x,y,z] == 0) | (FLAIR_data_scans[x,y,z] == 0):
                    next
                else:
                    brain_values = []
                    brain_values.append(file_id)
                    brain_values.append(x)
                    brain_values.append(y)
                    brain_values.append(z)

                    patch_3D_T1 = T1_data_scans[x-10:x+10, y-10:y+10, z-10:z+10]
                    brain_values.append(patch_3D_T1)

                    patch_3D_FLAIR = FLAIR_data_scans[x-10:x+10, y-10:y+10, z-10:z+10]
                    brain_values.append(patch_3D_FLAIR)

                    patch_3D_softtiss_binary = data[x-10:x+10, y-10:y+10, z-10:z+10]
                    brain_values.append(patch_3D_softtiss_binary)

                    patch_3D_softtiss = SoftTiss[x-10:x+10, y-10:y+10, z-10:z+10]
                    brain_values.append(patch_3D_softtiss)

                    patch_3D_CSF = CSF_data[x-10:x+10, y-10:y+10, z-10:z+10]
                    brain_values.append(patch_3D_CSF)

                    patch_3D_WM = WM_data[x-10:x+10, y-10:y+10, z-10:z+10]
                    brain_values.append(patch_3D_WM)

                    patch_3D_GM = GM_data[x-10:x+10, y-10:y+10, z-10:z+10]
                    brain_values.append(patch_3D_GM)

                    lacune_binary = Lacune_data[x-10:x+10, y-10:y+10, z-10:z+10]

                    if brain_image <= 24:
                        # No lacune exists in the 3D patch, add to train set
                        if any(1 in sublist for sublist in lacune_binary):
                            next
                        else:
                            X_train_3D_nlacune_func2.append(brain_values)
                            Y_train_3D_nlacune_func2.append(0)
                            Y_train_segment_3D_nlacune_func2.append(lacune_binary)
                                    
            brain_image += 1
            print(brain_image)

    return X_train_3D_nlacune_func2, Y_train_3D_nlacune_func2, Y_train_segment_3D_nlacune_func2

def test_sampling(CSF, GM, WM, T1_Soft_Tissue_Binary_Mask, T1_scan_data, FLAIR_scan_data, Lacune_indicator_data, Soft_tiss_data):
    # Lacune not as centred, random sampling all around brain
    X_test_3D_nlacune = []
    Y_test_3D_nlacune = []
    Y_test_segment_3D_nlacune = []
    X_test_3D_lacune = []
    Y_test_3D_lacune = []
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
            Lacune_data = Lacune_indicator_data[brain_image][1]
            CSF_data = CSF[brain_image][1]
            GM_data = GM[brain_image][1]
            WM_data = WM[brain_image][1]
            
            #Sample lacunes
            #Sample lacunes
            sampled_list_x = np.random.choice(data.shape[0], 30000)
            sampled_list_y = np.random.choice(data.shape[1], 30000)
            sampled_list_z = np.random.choice(data.shape[2], 30000)
            for x,y,z in set(list(zip(sampled_list_x, sampled_list_y,sampled_list_z))):
                #filter for soft tissue
                if (x < 50) | (y < 70) | (z < 15) | (x > 200) | (y > 210) | (z > 165) | (T1_data_scans[x,y,z] == 0) | (FLAIR_data_scans[x,y,z] == 0):
                    next
                else:
                    if brain_image <= 24:
                        next

                    else:
                        brain_values = []
                        brain_values.append(file_id)
                        brain_values.append(x)
                        brain_values.append(y)
                        brain_values.append(z)

                        patch_3D_T1 = T1_data_scans[x-10:x+10, y-10:y+10, z-10:z+10]
                        brain_values.append(patch_3D_T1)

                        patch_3D_FLAIR = FLAIR_data_scans[x-10:x+10, y-10:y+10, z-10:z+10]
                        brain_values.append(patch_3D_FLAIR)

                        patch_3D_softtiss_binary = data[x-10:x+10, y-10:y+10, z-10:z+10]
                        brain_values.append(patch_3D_softtiss_binary)

                        patch_3D_softtiss = SoftTiss[x-10:x+10, y-10:y+10, z-10:z+10]
                        brain_values.append(patch_3D_softtiss)

                        patch_3D_CSF = CSF_data[x-10:x+10, y-10:y+10, z-10:z+10]
                        brain_values.append(patch_3D_CSF)

                        patch_3D_WM = WM_data[x-10:x+10, y-10:y+10, z-10:z+10]
                        brain_values.append(patch_3D_WM)

                        patch_3D_GM = GM_data[x-10:x+10, y-10:y+10, z-10:z+10]
                        brain_values.append(patch_3D_GM)

                        lacune_binary = Lacune_data[x-10:x+10, y-10:y+10, z-10:z+10]

                        if Lacune_data[x,y,z] == 1:
                            print(brain_values[0])
                            X_test_3D_lacune.append(brain_values)
                            Y_test_3D_lacune.append(1)
                            Y_test_segment_3D_lacune.append(lacune_binary)
                        else:
                            X_test_3D_nlacune.append(brain_values)
                            Y_test_3D_nlacune.append(0)
                            Y_test_segment_3D_nlacune.append(lacune_binary)
            brain_image += 1
            print(brain_image)

    return X_test_3D_lacune, Y_test_3D_lacune, Y_test_segment_3D_lacune, X_test_3D_nlacune, Y_test_3D_nlacune, Y_test_segment_3D_nlacune

def train_test_combine(X_train_3D_lacune, Y_train_3D_lacune, Y_train_segment_3D_lacune, X_train_3D_nlacune, Y_train_3D_nlacune, Y_train_segment_3D_nlacune, X_train_3D_nlacune_func2, Y_train_3D_nlacune_func2, Y_train_segment_3D_nlacune_func2, X_test_3D_lacune, Y_test_3D_lacune, Y_test_segment_3D_lacune, X_test_3D_nlacune, Y_test_3D_nlacune, Y_test_segment_3D_nlacune):
    X_train_3D_nlacune_all = np.concatenate((X_train_3D_nlacune, X_train_3D_nlacune_func2), axis=0)
    Y_train_3D_nlacune_all = np.concatenate((Y_train_3D_nlacune, Y_train_3D_nlacune_func2), axis=0)
    Y_train_segment_3D_nlacune_all = np.concatenate((Y_train_segment_3D_nlacune, Y_train_segment_3D_nlacune_func2), axis=0)
    X_train = np.concatenate((X_train_3D_lacune, X_train_3D_nlacune_all), axis=0)
    Y_train = np.concatenate((Y_train_3D_lacune, Y_train_3D_nlacune_all), axis=0)
    Y_train_segment = np.concatenate((Y_train_segment_3D_lacune, Y_train_segment_3D_nlacune_all), axis=0)
    Y_test_segment = np.concatenate((Y_test_segment_3D_lacune, Y_test_segment_3D_nlacune), axis=0)
    Y_test = np.concatenate((Y_test_3D_lacune, Y_test_3D_nlacune), axis=0)
    X_test = np.concatenate((X_test_3D_lacune, X_test_3D_nlacune), axis=0)
    return X_train_3D_nlacune_all, Y_train_3D_nlacune_all, Y_train_segment_3D_nlacune_all, X_train, Y_train, Y_train_segment, Y_test_segment, Y_test, X_test
