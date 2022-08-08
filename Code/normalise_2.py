import os
import nibabel as nib
import numpy as np

# Normalise the brain scan data for all patients
def normalise(mypath_og):
    for root, dirs, files in os.walk(mypath_og, topdown=False):
        print(root)
        for file in files:
            if file.endswith(".nii.gz"):
                imgpath = os.path.join(root, file)
                img = nib.load(imgpath)
                data = img.get_fdata()

                # Make all negative data zero
                data[data < 0] = 0

                # Normalising values between 0 and 1
                data = (data - np.min(data)) / (np.max(data) - np.min(data)) 
          
                print(data.shape)
                # Save to new nib file
                norm_img = nib.Nifti1Image(data, img.affine, img.header)
                norm_root = root.replace("Original", "Normalised")
                norm_imgpath_file = os.path.join(norm_root, file)
                nib.save(norm_img, norm_imgpath_file)
