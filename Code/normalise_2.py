import os
import nibabel as nib
import numpy as np

def normalise(mypath_og):
    for root, dirs, files in os.walk(mypath_og, topdown=False):
        for file in files:
            if file.endswith(".nii.gz"):
                imgpath = os.path.join(root, file)
                img = nib.load(imgpath)
                data = img.get_fdata()
                #workaround to (256, 256, 180)
                #if data.shape == (256, 256, 180):
                #    data = np.dstack((data, matrix_append))

                #make all negative data positive
                data = abs(data)
                #normalising values between 0 and 1
                data = (data - np.min(data)) / (np.max(data) - np.min(data)) 
          
                print(data.shape)
                # save to new nib file
                norm_img = nib.Nifti1Image(data, img.affine, img.header)
                norm_root = root.replace("Original", "Normalised")
                norm_imgpath_file = os.path.join(norm_root, file)
                nib.save(norm_img, norm_imgpath_file)