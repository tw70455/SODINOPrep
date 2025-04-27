# Set the path where the anatomical files are located
# Set the subject IDs you want to process

set FILE_PATH="/Volumes/ProcessDisk/fMRIData/"
set NUMS="Subj01 Subj02 Subj03"


# Set base directory (where the script is located)
set BASE_DIR = `dirname $0`


foreach ratNum(${NUMS}) # 
	if  ( -e "${FILE_PATH}${ratNum}_Func.nii.gz" ) then

		# Remove previous masked anatomical file if it exists
		rm "${FILE_PATH}${ratNum}_Anat_masked.nii.gz"
		# Create a masked anatomical image using the sum of mask and add file
		3dcalc -a "${FILE_PATH}${ratNum}_Anat.nii.gz" -b "${FILE_PATH}${ratNum}_Anat_mask.nii.gz" -c "${FILE_PATH}${ratNum}_Anat_032_add.nii" -expr 'a*ispositive(b+c)' -prefix "${FILE_PATH}${ratNum}_Anat_masked.nii.gz"
		3dcalc -a "${FILE_PATH}${ratNum}_Anat.nii.gz" -b "${FILE_PATH}${ratNum}_Anat_mask.nii.gz" -expr 'a*ispositive(b)' -prefix "${FILE_PATH}${ratNum}_Anat_masked.nii.gz"


		# Remove previous bias-corrected anatomical file if it exists
		rm "${FILE_PATH}${ratNum}_Anat_masked_unify5.nii.gz"
		# Apply bias field correction (unifize) with radius 5
		3dUnifize -Urad 5 -prefix "${FILE_PATH}${ratNum}_Anat_masked_unify5.nii.gz" "${FILE_PATH}${ratNum}_Anat_masked.nii.gz"
		
		
		
		3drefit -orient RPS "${FILE_PATH}${ratNum}_Anat_masked_unify5.nii.gz"
		
		# Remove previous ANTs registration files if they exist
		rm "${FILE_PATH}${ratNum}_Anat_masked_unify5_Ants*"
		# Register anatomical to template using ANTs (SyN registration)
		antsRegistrationSyN.sh -d 3 -f "${BASE_DIR}/Template/ZTE_Template_mean_Ants_resample_allineate_resample.nii.gz" \
			-m "${FILE_PATH}${ratNum}_Anat_masked_unify5.nii.gz" \
				-o "${FILE_PATH}${ratNum}_Anat_masked_unify5_Ants" -t s -n 4



		3drefit -orient RPS "${FILE_PATH}${ratNum}_Func.nii.gz"

		# Remove previous warped functional file if it exists
		rm "${FILE_PATH}${ratNum}_Func_ANTsWarped.nii.gz"
		# Apply the anatomical-derived transforms to the functional image
		antsApplyTransforms -d 3 -e 3 -v 1 --float \
		  -i "${FILE_PATH}${ratNum}_Func.nii.gz" \
		  -r "${BASE_DIR}/Template/ZTE_Template_mean_Ants_resample_allineate_2ZTE.nii.gz" \
		  -o "${FILE_PATH}${ratNum}_Func_ANTsWarped.nii.gz" \
		  -t "${FILE_PATH}${ratNum}_Anat_masked_unify5_Ants1Warp.nii.gz" \
		  -t "${FILE_PATH}${ratNum}_Anat_masked_unify5_Ants0GenericAffine.mat" \
		  -n Linear


	else
		echo "Warning! ${FILE_PATH}${ratNum}_Func.nii.gz not found."
	endif
end



