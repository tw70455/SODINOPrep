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
4. Run the Ana01, copy all the *_Anat.nii.gz to the IMG folder in Ana02_BrainSeg
tcsh Ana03_ResampleUNetMask.sh

6. Run the python code of 'segEvaluation_Main.py' in Ana02_Brainseg for the brain segmentation, get the result from the IMGMask folder in Ana02_BrainSeg and put all the output image with your image files in the same folder (for example /Volumes/ProcessDisk/fMRIData/).
python segEvaluation_Main.py

7. Run the Ana03, Ana04, and Ana05
tcsh Ana03_ResampleUNetMask.sh
tcsh Ana04_shift2Temp_ANTs.sh
tcsh Ana05_fMRIProcess.sh

9. Run the Ana06 in MATLAB

10. Run the Ana07 and 08
tcsh Ana07_RS_36motionrs.sh
tcsh Ana08_runScrubbing.sh

12. For Ana09 and Ana10, it's for manually ICA denoise, if you don't need it, you can directly use the output files (*_Func_ANTsWarped_masked_unify5_volreg_blur_despike_detrend_blur_rmmotion36_bandpass015_admean_scrub.nii.gz) for further analysis.

    
14. Run the Ana09 for the ICA denoise, and Ana10 for the denoise component selection. You need to manually put the noise component in '-f "1, 2, 3, 4, 5, 6, 7, 8, 9, 10"'. 
