# Set the path where the anatomical files are located
# Set the subject IDs you want to process

set FILE_PATH="/Volumes/ProcessDisk/fMRIData/"
set NUMS="Subj01 Subj02 Subj03"


foreach ratNum ($NUMS)
	if  ( -e "${FILE_PATH}${ratNum}_Anat.nii.gz" ) then
		# Remove the previous file that has been resampled to 0.32 mm resolution if it exists
		rm "${FILE_PATH}${ratNum}_Anat_032.nii.gz"
		3dresample -dxyz 0.32 0.32 0.32 -inset "${FILE_PATH}${ratNum}_Anat.nii.gz" -prefix "${FILE_PATH}${ratNum}_Anat_032.nii.gz"
	else
		echo "Warning! ${FILE_PATH}${ratNum}_Anat.nii.gz not found."
	endif
end

