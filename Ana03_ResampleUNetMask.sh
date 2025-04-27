# Set the path where the anatomical files are located
# Set the subject IDs you want to process

set FILE_PATH="/Volumes/ProcessDisk/fMRIData/"
set NUMS="Subj01 Subj02 Subj03"



foreach ratNum($NUMS) 

3drefit -orient LAS -xdel 0.16 -ydel 0.16 -zdel 0.16 -duporigin "${FILE_PATH}${ratNum}_Anat.nii.gz" "${FILE_PATH}${ratNum}_Anat_RPS_mask.nii.gz" 

end
