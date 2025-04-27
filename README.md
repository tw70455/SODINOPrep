# SODINOPrep

This toolbox is designed specifically for SORDINO resting-state fMRI preprocessing.

To use this pipeline, the following software packages are required:

	 •	AFNI (for basic preprocessing and motion correction)

	 •	ANTs (for anatomical-to-template registration)

	 •	MATLAB (for motion processing steps)

	 •	FSL (required for ICA-based denoising)

Before running the UNet segmentation pipeline, please ensure the following Python packages are installed:

	 •	tensorflow (tested with TensorFlow 2.x)

	 •	keras

	 •	h5py

	 •	SimpleITK

	 •	scipy

	 •	Pillow

	 •	scikit-image

Please ensure that all software is correctly installed and properly added to your system path before running the scripts.




##############################################################################################################################################################################################################

Before using the pipeline, you also need to download the following files (these file sizes exceed GitHub’s upload limit):

1. Template Download: https://www.dropbox.com/scl/fo/2z81khp7jhaocjzzuuspc/AEjrPOTUD91SlmYRdTx9hrs?rlkey=rver8dg04auxbqaduhb7rvi5q&dl=0

	•	After downloading, make sure all template files are placed inside a folder named Template.

	•	The Template folder must be located in the same directory as the pipeline scripts.

3. UNet SORDINO Download: https://www.dropbox.com/scl/fi/7uyyk2hmfh3enz784u0gz/SORDINO_UNet_model.hdf5?rlkey=38rnbk0nr0ngpr62cc64sn533&dl=0

	•	Place the SORDINO_UNet_model.hdf5 file inside the Ana02_BrainSeg folder.




##############################################################################################################################################################################################################

Below is the use instruction:

1. All anatomical images must be named as ***_Anat.nii.gz**, and all functional images must be named as ***_Func.nii.gz**.

2. For each script, modify the folder path and subject names accordingly.

Example:

**set FILE_PATH="/Volumes/ProcessDisk/fMRIData/"**

**set NUMS="Subj01 Subj02 Subj03"**

Here, **FILE_PATH** should be the folder containing your images, and **NUMS** should list the subject IDs (the prefix before **_Func.nii.gz**).

Please put your image file path and each subject name (file name before **_Func.nii.gz**).

3. Run Ana01 to process the anatomical images (command: **tcsh Ana01_shift_imgt.sh**). After running Ana01, copy all the _Anat.nii.gz files into the IMG folder under Ana02_BrainSeg.
**tcsh Ana03_ResampleUNetMask.sh**

4. Run Ana02 for brain segmentation using UNet (command: **python segEvaluation_Main.py**)

5. The segmented masks will be saved into the **IMGMask** folder. After segmentation, copy the output images back to the same folder as your original functional images (e.g., **/Volumes/ProcessDisk/fMRIData/**).

6. Run Ana03 (command: **tcsh Ana03_ResampleUNetMask.sh**)
IMPORTANT: After running Ana03, please carefully check the brain segmentation results for each subject.
Review both the anatomical image (***_Anat.nii.gz**) and its corresponding mask (***_Anat_mask.nii.gz**).
If any brain regions are missing or not fully covered by the mask, you must manually correct them.
To speed up manual correction, you can use the downsampled anatomical image (***_Anat_032.nii.gz**).
Please save any manually added mask as a new file named *_Anat_032_add.nii.gz for next step Ana04.
If a ***_Anat_032_add.nii.gz** file is present, Ana04 will automatically incorporate it to improve the brain mask based on ***_Anat_mask.nii.gz**.
If no ***_Anat_032_add.nii.gz** file is available, Ana04 will proceed using only *_Anat_mask.nii.gz for brain masking.

8. Run Ana04 and Ana05 sequentially:
**tcsh Ana04_shift2Temp_ANTs.sh**
**tcsh Ana05_fMRIProcess.sh**

9. Run Ana06 in MATLAB for further preprocessing.

10. Run Ana07 and Ana08 for motion correction and scrubbing:
**tcsh Ana07_RS_36motionrs.sh**
**tcsh Ana08_runScrubbing.sh**

11. Ana09 and Ana10 are optional steps for manual ICA denoising. If you choose not to perform ICA denoising, you can directly use the following output file for further analysis: ***_Func_ANTsWarped_masked_unify5_volreg_blur_despike_detrend_blur_rmmotion36_bandpass015_admean_scrub.nii.gz**

12. If you want to perform ICA denoising:
	•	Run Ana09 to perform ICA decomposition.






Summary: Initially, brain extraction is performed using a 3D UNet-based segmentation model to accurately isolate brain tissue. Anatomical images are subsequently normalized to a standardized template space using Advanced Normalization Tools (ANTs). Functional images undergo spatial smoothing with a Gaussian kernel of 0.6 mm full-width at half-maximum (FWHM) to improve signal-to-noise ratio. Motion artifacts are mitigated through a 36-parameter regression model, followed by temporal band-pass filtering (0.01–0.15 Hz) to retain physiologically relevant frequency components. Scrubbing is applied to censor volumes exhibiting framewise displacement (FD) greater than 0.04 mm, ensuring the exclusion of high-motion frames. Finally, independent component analysis (ICA)-based denoising is optionally performed to further remove residual noise sources. This comprehensive preprocessing framework aims to produce high-quality, standardized fMRI datasets suitable for subsequent functional connectivity and network analyses.




	•	Run Ana10 to manually select noise components.
You will need to manually specify the noise components in the command line option. (e.g., **-f "1,2,3,4,5,6,7,8,9,10"**)
