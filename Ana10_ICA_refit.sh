

set FILE_PATH="/Volumes/ProcessDisk/fMRIData/"
set NUMS="Subj01"



fsl_regfilt -i "${FILE_PATH}${NUMS}_Func_ANTsWarped_masked_unify5_volreg_blur_despike_detrend_blur_rmmotion36_bandpass015_admean_scrub_masked.nii.gz" \
	-d "${FILE_PATH}${NUMS}_Func_ANTsWarped_masked_unify5_volreg_blur_despike_detrend_blur_rmmotion36_bandpass015_admean_scrub_masked_RIA/melodic_mix" \
		-o "${FILE_PATH}${NUMS}_Func_ANTsWarped_masked_unify5_volreg_blur_despike_detrend_blur_rmmotion36_bandpass015_admean_scrub_masked_denoise.nii.gz" \
			-f "1, 2, 3, 4, 5, 6, 7, 8, 9, 10"

