# Set the path where the anatomical files are located
# Set the subject IDs you want to process

set FILE_PATH="/Volumes/ProcessDisk/fMRIData/"
set NUMS="Subj01 Subj02 Subj03"


# Set base directory (where the script is located)
set BASE_DIR = `dirname $0`


foreach ratNum($NUMS) #01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20

rm "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_blur_despike_detrend_blur_rmmotion36_bandpass015_admean_scrub_masked.nii.gz"
3dcalc -a "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_blur_despike_detrend_blur_rmmotion36_bandpass015_admean_scrub.nii.gz"	\
	-b "${BASE_DIR}/Template/ZTE_Template_mean_Ants_resample_allineate_2ZTE_mask.nii.gz"	\
		-expr 'a*b' -prefix "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_blur_despike_detrend_blur_rmmotion36_bandpass015_admean_scrub_masked.nii.gz"

rm "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_blur_despike_detrend_blur_rmmotion36_bandpass015_admean_scrub_masked_RIA.nii.gz"
3dresample -orient RIA -inset "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_blur_despike_detrend_blur_rmmotion36_bandpass015_admean_scrub_masked.nii.gz"	\
	-prefix "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_blur_despike_detrend_blur_rmmotion36_bandpass015_admean_scrub_masked_RIA.nii.gz"

rm -r "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_blur_despike_detrend_blur_rmmotion36_bandpass015_admean_scrub_masked_RIA"
melodic -i "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_blur_despike_detrend_blur_rmmotion36_bandpass015_admean_scrub_masked_RIA.nii.gz" \
 -o "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_blur_despike_detrend_blur_rmmotion36_bandpass015_admean_scrub_masked_RIA" --nobet \
	 -d 35 -a concat --report --Oall

end