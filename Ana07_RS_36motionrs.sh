# Set the path where the anatomical files are located
# Set the subject IDs you want to process

set FILE_PATH="/Volumes/ProcessDisk/fMRIData/"
set NUMS="Subj01 Subj02 Subj03"



foreach ratNum($NUMS) #01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20
		
		### despike
		rm "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_blur_despike.nii.gz"
		3dDespike -NEW25 -prefix "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_blur_despike.nii.gz" -nomask "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_blur.nii.gz"
		
		
		#####3ddeconvolve 'motion'
		rm "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_blur_despike_detrend_blur_rmmotion36.nii.gz"
		rm "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_blur_despike_detrend_blur_rmmotion36_fitts_IM*"
		3dDeconvolve -input "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_blur_despike.nii.gz" 	\
		-local_times -polort 13 -jobs 11	-num_stimts 36										\
		-fitts "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_blur_despike_detrend_blur_rmmotion36_fitts_IM"									\
		-errts "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_blur_despike_detrend_blur_rmmotion36.nii.gz"		\
		-stim_file 1 "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_parameters_n"'[0]' 								\
  		-stim_label 1 "roll" 													\
  		-stim_base 1														\
  		-stim_file 2 "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_parameters_n"'[1]' 								\
   		-stim_label 2 "pitch" 													\
   		-stim_base 2														\
		-stim_file 3 "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_parameters_n"'[2]' 								\
    	-stim_label 3 "yaw" 													\
    	-stim_base 3														\
		-stim_file 4 "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_parameters_n"'[3]' 								\
   		-stim_label 4 "DS" 													\
   		-stim_base 4														\
		-stim_file 5 "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_parameters_n"'[4]' 								\
   		-stim_label 5 "DL" 													\
   		-stim_base 5														\
		-stim_file 6 "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_parameters_n"'[5]' 								\
    	-stim_label 6 "DP" 													\
    	-stim_base 6														\
			-stim_file 7 "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_parameters_n"'[6]' 								\
	  		-stim_label 7 "roll1" 													\
	  		-stim_base 7														\
	  		-stim_file 8 "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_parameters_n"'[7]' 								\
	   		-stim_label 8 "pitch1" 													\
	   		-stim_base 8														\
			-stim_file 9 "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_parameters_n"'[8]' 								\
	    	-stim_label 9 "yaw1" 													\
	    	-stim_base 9														\
			-stim_file 10 "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_parameters_n"'[9]' 								\
	   		-stim_label 10 "DS1" 													\
	   		-stim_base 10														\
			-stim_file 11 "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_parameters_n"'[10]' 								\
	   		-stim_label 11 "DL1" 													\
	   		-stim_base 11														\
			-stim_file 12 "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_parameters_n"'[11]' 								\
	    	-stim_label 12 "DP1" 													\
	    	-stim_base 12														\
				-stim_file 13 "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_parameters_n"'[12]' 								\
		  		-stim_label 13 "roll2" 													\
		  		-stim_base 13														\
		  		-stim_file 14 "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_parameters_n"'[13]' 								\
		   		-stim_label 14 "pitch2" 													\
		   		-stim_base 14														\
				-stim_file 15 "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_parameters_n"'[14]' 								\
		    	-stim_label 15 "yaw2" 													\
		    	-stim_base 15														\
				-stim_file 16 "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_parameters_n"'[15]' 								\
		   		-stim_label 16 "DS2" 													\
		   		-stim_base 16														\
				-stim_file 17 "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_parameters_n"'[16]' 								\
		   		-stim_label 17 "DL2" 													\
		   		-stim_base 17														\
				-stim_file 18 "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_parameters_n"'[17]' 								\
		    	-stim_label 18 "DP2" 													\
		    	-stim_base 18														\
					-stim_file 19 "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_parameters_n"'[18]' 								\
			  		-stim_label 19 "roll3" 													\
			  		-stim_base 19														\
			  		-stim_file 20 "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_parameters_n"'[19]' 								\
			   		-stim_label 20 "pitch3" 													\
			   		-stim_base 20														\
					-stim_file 21 "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_parameters_n"'[20]' 								\
			    	-stim_label 21 "yaw3" 													\
			    	-stim_base 21														\
					-stim_file 22 "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_parameters_n"'[21]' 								\
			   		-stim_label 22 "DS3" 													\
			   		-stim_base 22														\
					-stim_file 23 "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_parameters_n"'[22]' 								\
			   		-stim_label 23 "DL3" 													\
			   		-stim_base 23														\
					-stim_file 24 "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_parameters_n"'[23]' 								\
			    	-stim_label 24 "DP3" 													\
			    	-stim_base 24														\
						-stim_file 25 "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_parameters_n"'[24]' 								\
				  		-stim_label 25 "roll4" 													\
				  		-stim_base 25														\
				  		-stim_file 26 "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_parameters_n"'[25]' 								\
				   		-stim_label 26 "pitch4" 													\
				   		-stim_base 26														\
						-stim_file 27 "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_parameters_n"'[26]' 								\
				    	-stim_label 27 "yaw4" 													\
				    	-stim_base 27														\
						-stim_file 28 "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_parameters_n"'[27]' 								\
				   		-stim_label 28 "DS4" 													\
				   		-stim_base 28														\
						-stim_file 29 "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_parameters_n"'[28]' 								\
				   		-stim_label 29 "DL4" 													\
				   		-stim_base 29														\
						-stim_file 30 "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_parameters_n"'[29]' 								\
				    	-stim_label 30 "DP4" 													\
				    	-stim_base 30														\
							-stim_file 31 "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_parameters_n"'[30]' 								\
					  		-stim_label 31 "roll5" 													\
					  		-stim_base 31														\
					  		-stim_file 32 "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_parameters_n"'[31]' 								\
					   		-stim_label 32 "pitch5" 													\
					   		-stim_base 32														\
							-stim_file 33 "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_parameters_n"'[32]' 								\
					    	-stim_label 33 "yaw5" 													\
					    	-stim_base 33														\
							-stim_file 34 "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_parameters_n"'[33]' 								\
					   		-stim_label 34 "DS5" 													\
					   		-stim_base 34														\
							-stim_file 35 "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_parameters_n"'[34]' 								\
					   		-stim_label 35 "DL5" 													\
					   		-stim_base 35														\
							-stim_file 36 "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_parameters_n"'[35]' 								\
					    	-stim_label 36 "DP5" 													\
					    	-stim_base 36														\
								-nobucket
		
		
		#### bandpass
		rm "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_blur_despike_detrend_blur_rmmotion36_bandpass015.nii.gz"
		3dBandpass -prefix "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_blur_despike_detrend_blur_rmmotion36_bandpass015.nii.gz" -band 0.01 0.15 "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_blur_despike_detrend_blur_rmmotion36.nii.gz"
		
		
		rm "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_blur_mean.nii.gz"
		3dTstat -mean -prefix "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_blur_mean.nii.gz" "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_blur.nii.gz"
		
		
		
		#### admean
		rm "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_blur_despike_detrend_blur_rmmotion36_bandpass015_admean.nii.gz"
		3dcalc -a "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_blur_despike_detrend_blur_rmmotion36_bandpass015.nii.gz" \
			-b "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_blur_mean.nii.gz" \
			-expr 'a+b' -prefix "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_blur_despike_detrend_blur_rmmotion36_bandpass015_admean.nii.gz"
		
		
		rm "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_blur_despike_detrend_blur_rmmotion36_fitts_IM+orig.BRIK"
		rm "${FILE_PATH}${ratNum}_Func_ANTsWarped_masked_unify5_volreg_blur_despike_detrend_blur_rmmotion36_fitts_IM+orig.HEAD"
		
end
