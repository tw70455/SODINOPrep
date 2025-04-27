# Set the path where the anatomical files are located
# Set the subject IDs you want to process

set FILE_PATH="/Volumes/ProcessDisk/fMRIData/"
set NUMS="Subj01 Subj02 Subj03"



foreach ratNum($NUMS) #01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20

cd "${FILE_PATH}"
tcsh "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_parameters_n_sensor.sh"

end