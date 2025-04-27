# Set the path where the anatomical files are located
# Set the subject IDs you want to process

set FILE_PATH="/Volumes/ProcessDisk/fMRIData/"
set NUMS="Subj01 Subj02 Subj03"


# Set base directory (where the script is located)
set BASE_DIR = `dirname $0`


foreach ratNum($NUMS)


	# Step 1: Apply mask to the trimmed data
	3dcalc -a ${FILE_PATH}${ratNum}_Func_ANTsWarped.nii.gz'[5..$]' \
	       -b "${BASE_DIR}/Template/ZTE_Template_mean_Ants_resample_allineate_2ZTE_mask.nii.gz" \
	       -expr 'a*b' \
	       -prefix "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked.nii.gz"


	# Step 2: Calculate mean image after masking
	rm "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_mean.nii.gz"
	3dTstat -mean -prefix "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_mean.nii.gz" "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked.nii.gz"


	# Step 3: Bias field correction (unifize) on mean image
	rm "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_mean_unify5_scale.nii.gz"
	rm "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_mean_unify5.nii.gz"
	3dUnifize -Urad 5 -ssave "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_mean_unify5_scale.nii.gz" \
	        -prefix "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_mean_unify5.nii.gz" "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_mean.nii.gz"


	# Step 4: Intensity scaling by bias correction factor
	rm "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5.nii.gz"
	3dcalc -a "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked.nii.gz" \
	        -b "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_mean_unify5_scale.nii.gz" \
	        -expr 'a*b' -prefix "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5.nii.gz"


	# Step 5: Adjust TR (set repetition time to 2 seconds)
	3drefit -TR 2 "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5.nii.gz"


	# Step 6: Motion correction (volume registration)
	rm "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_parameters"
	rm "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg.nii.gz"
	3dvolreg -tshift 0 -zpad 4 -prefix "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg.nii.gz" -1Dfile "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_parameters" "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5.nii.gz"


	# Step 7: Spatial smoothing (0.6 mm FWHM Gaussian blur)
	rm "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_blur.nii.gz"
	3dmerge -doall -1blur_fwhm 0.6 -prefix "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_blur.nii.gz" "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg.nii.gz"


end
