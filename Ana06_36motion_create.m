
clear all;

% Set the path where the files are located
FILE_PATH="/Volumes/ProcessDisk/fMRIData/"





Files = dir(fullfile(FILE_PATH,'*_volreg_parameters'));


for subject_count = 1 : length(Files)
	file_name = [Files(subject_count).folder '/' Files(subject_count).name];
	[txt] = textread(file_name);


	%% R2
	for motion_count = 1 : size(txt,2)
	    temp = txt(:,motion_count);
	    temp = temp.^2;
	    R2_motion(:,motion_count) = temp;
	end


	%% Rt-1
	for motion_count = 1 : size(txt,2)
	    temp = txt(:,motion_count);
	    temp = temp(1:end-1);
	    Rt_1_motion(:,motion_count) = [0;temp];
	end



	%% Rt-1^2
	for motion_count = 1 : size(txt,2)
	    temp = txt(:,motion_count);
	    temp = temp(1:end-1);
	    Rt_1square_motion(:,motion_count) = ([0;temp]).^2;
	end


	%% Rt-2
	for motion_count = 1 : size(txt,2)
	    temp = txt(:,motion_count);
	    temp = temp(1:end-2);
	    Rt_2_motion(:,motion_count) = [0;0;temp];
	end

	%% Rt-2^2
	for motion_count = 1 : size(txt,2)
	    temp = txt(:,motion_count);
	    temp = temp(1:end-2);
	    Rt_2square_motion(:,motion_count) = ([0;0;temp]).^2;
	end

	final_motion = cat(2,txt,R2_motion,Rt_1_motion,Rt_1square_motion,Rt_2_motion,Rt_2square_motion);

	format short

	fileID = fopen([file_name '_n'],'w');
	fprintf(fileID,'%3f	%3f	%3f	%3f	%3f	%3f	%3f	%3f	%3f	%3f	%3f	%3f %3f	%3f	%3f	%3f	%3f	%3f	%3f	%3f	%3f	%3f	%3f	%3f %3f	%3f	%3f	%3f	%3f	%3f	%3f	%3f	%3f	%3f	%3f	%3f\n',final_motion');
	fclose(fileID);





	RP = txt;
	RP(:,1:3) = RP(:,1:3)*(pi/180);
	RPDiff=diff(RP);
	RPDiff=[zeros(1,6);RPDiff];
	RPDiffSphere=RPDiff;
	RPDiffSphere(:,1:3)=RPDiffSphere(:,1:3)*5;
	FD_Power=sum(abs(RPDiffSphere),2);
	MeanFD_Power(subject_count)= mean(FD_Power);

	FD_threshold = 0.04;

	rm_volume = find(FD_Power>FD_threshold);
	rm_volume = [rm_volume;rm_volume+1];
	rm_volume = unique(rm_volume);

	keep_volume = setdiff([1:length(FD_Power)],rm_volume);
	command_list1 = ['3dTcat -prefix ' Files(subject_count).name(1:end-11) '_blur_despike_detrend_blur_rmmotion36_bandpass015_admean_scrub.nii.gz ' Files(subject_count).name(1:end-11) '_blur_despike_detrend_blur_rmmotion36_bandpass015_admean.nii.gz"['];
	command_list = [];
	for volume_count = 1 : length(keep_volume)
	    command_list = [command_list num2str(keep_volume(volume_count)-1) ', '];
	end
	disp([command_list1 command_list])

	rm_percentage(subject_count) = length(rm_volume)/length(FD_Power);

	fileID = fopen([file_name '_n_sensor.sh'],'w');
	fprintf(fileID,[command_list1 command_list(1:end-2) ']"']);
	fclose(fileID);

end
