# SODINOPrep
Preprocessing steps for rodent SORDINO fMRI

Before using the pipeline, you need to download the following files:

1. Template Download: https://www.dropbox.com/scl/fo/2z81khp7jhaocjzzuuspc/AEjrPOTUD91SlmYRdTx9hrs?rlkey=rver8dg04auxbqaduhb7rvi5q&dl=0

2. UNet SORDINO Download: https://www.dropbox.com/scl/fi/7uyyk2hmfh3enz784u0gz/SORDINO_UNet_model.hdf5?rlkey=38rnbk0nr0ngpr62cc64sn533&dl=0

Place the Template folder in the same directory as the pipeline script, and place the UNet SORDINO model in the Ana02_BrainSeg folder.



Below is the use instruction:

1. All anatomical images must be named as *_Anat.nii.gz, and all functional images must be named as *_Func.nii.gz.
2. For each script, modify the folder path and subject names accordingly.
Example: 
set FILE_PATH="/Volumes/ProcessDisk/fMRIData/"
set NUMS="Subj01 Subj02 Subj03"

Here, FILE_PATH should be the folder containing your images, and NUMS should list the subject IDs (the prefix before _Func.nii.gz).

Please put your image file path and each subject name (file name before _Func.nii.gz).

3. Run Ana01 to process the anatomical images (code: tcsh Ana01_shift_imgt.sh). After running Ana01, copy all the _Anat.nii.gz files into the IMG folder under Ana02_BrainSeg.
tcsh Ana03_ResampleUNetMask.sh

4. Run Ana02 for brain segmentation using UNet (code: python segEvaluation_Main.py)

5. The segmented masks will be saved into the IMGMask folder. After segmentation, copy the output images back to the same folder as your original functional images (e.g., /Volumes/ProcessDisk/fMRIData/).

6. Run Ana03, Ana04, and Ana05 sequentially:
tcsh Ana03_ResampleUNetMask.sh
tcsh Ana04_shift2Temp_ANTs.sh
tcsh Ana05_fMRIProcess.sh

7. Run Ana06 in MATLAB for further preprocessing.

8. Run Ana07 and Ana08 for motion correction and scrubbing:
tcsh Ana07_RS_36motionrs.sh
tcsh Ana08_runScrubbing.sh

9. Ana09 and Ana10 are optional steps for manual ICA denoising. If you choose not to perform ICA denoising, you can directly use the following output file for further analysis: *_Func_ANTsWarped_masked_unify5_volreg_blur_despike_detrend_blur_rmmotion36_bandpass015_admean_scrub.nii.gz

10. If you want to perform ICA denoising:
	•	Run Ana09 to perform ICA decomposition.
	•	Run Ana10 to manually select noise components.
You will need to manually specify the noise components in the command line option. (e.g., -f "1,2,3,4,5,6,7,8,9,10")
