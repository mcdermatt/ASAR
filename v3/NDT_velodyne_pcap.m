% estimates vehicle trajectory given Velodyne pcap file

beep off
clear all
close all

fn = "D:/volpe/2021-03-10-16-43-50_Velodyne-VLP-16-Data_garminSignage.pcap";
veloReader = velodyneFileReader(fn, "VLP16");

ans_cum = [];
i = 1;
fixed = veloReader.readFrame()
moving = veloReader.readFrame()
%remove ground plane
[~,fixed,ground_fixed] = segmentGroundSMRF(fixed,MaxWindowRadius=5,ElevationThreshold=0.1,ElevationScale=0.25);
[~,moving,ground_moving] = segmentGroundSMRF(moving,MaxWindowRadius=5,ElevationThreshold=0.1,ElevationScale=0.25);


% while(hasFrame(veloReader) && (veloReader.CurrentTime < veloReader.StartTime + seconds(10)))
while(hasFrame(veloReader))
    i   
    
    %add noise to each PC
%     noise_scale = 0.02;
%     scan1 = scan1 + noise_scale*randn(size(scan1));
%     scan2 = scan2 + noise_scale*randn(size(scan2));
%     
    
    %NDT---------------------------------------------
    gridstep = 0.5;
    if i == 1
        [tform, movingReg, rmse] = pcregisterndt(moving, fixed, gridstep);%    
    else
        [tform, movingReg, rmse] = pcregisterndt(moving, fixed, gridstep, "InitialTransform", tform, Tolerance=[0.001, 0.005]); %use output of last sim as input for next
    end
        %------------------------------------------------

    ans = [tform.Translation, rotm2eul(tform.Rotation)];
    ans_cum = [ans_cum; ans];
    
    fixed = moving; %make 2nd frame from last registration the new keyframe
    moving = veloReader.readFrame();
    [~,moving,ground_moving] = segmentGroundSMRF(moving,MaxWindowRadius=5,ElevationThreshold=0.1,ElevationScale=0.25); %remove ground plane

%     veloReader.CurrentTime
    i = i + 1;

end

%save to file
fn = "NDT_results_initialguess_signage.txt";
writematrix(ans_cum, fn, 'Delimiter', 'tab')