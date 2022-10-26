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

dSinceLastKey = [0, 0, 0];
magD = 0;
keyArr = [1]
dThresh = 1; %movement threshold required to re-keyframe
% while(hasFrame(veloReader) && (veloReader.CurrentTime < veloReader.StartTime + seconds(10)))
while(hasFrame(veloReader))
    i   
    
    %add noise to each PC
%     noise_scale = 0.02;
%     scan1 = scan1 + noise_scale*randn(size(scan1));
%     scan2 = scan2 + noise_scale*randn(size(scan2));
%     
    
    %NDT---------------------------------------------
    gridstep = 2.5;
%     [tform, movingReg, rmse] = pcregisterndt(moving, fixed, gridstep, Tolerance=[0.001, 0.005], OutlierRatio=0.1);
    if i == 1
        [tform, movingReg, rmse] = pcregisterndt(moving, fixed, gridstep);%    
        ans = [tform.Translation, rotm2eul(tform.Rotation)];
    else
        [tform, movingReg, rmse] = pcregisterndt(moving, fixed, gridstep, "InitialTransform", tform_init, Tolerance=[0.001, 0.005]); %use output of last sim as input for next
        ans = [tform.Translation + tform_init.Translation, rotm2eul(tform.Rotation)];
    end
        %------------------------------------------------

    ans_cum = [ans_cum; ans];
    
    magD = magD + sqrt(tform.Translation(1).^2 + tform.Translation(2).^2);

    if magD > dThresh
        fixed = moving; %make 2nd frame from last registration the new keyframe
        dSinceLastKey = [0, 0, 0];
        magD = 0;
        keyArr = [keyArr, 1];
    else
        dSinceLastKey = dSinceLastKey + [tform.Translation(1), tform.Translation(2), 0]; %ignore z component
        tform_init = tform;
        tform_init.Translation = dSinceLastKey;
        keyArr = [keyArr, 0];
    end
    moving = veloReader.readFrame();
    [~,moving,ground_moving] = segmentGroundSMRF(moving,MaxWindowRadius=5,ElevationThreshold=0.1,ElevationScale=0.25); %remove ground plane

%     veloReader.CurrentTime
    i = i + 1;

end

%save to file
% fn = "NDT_results_pt5m_noRejection_signage.txt";
fn = "NDT_results_v2pt5d1_signage_v2.txt";
writematrix(ans_cum, fn, 'Delimiter', 'tab')

fn = "NDT_keyframes_v2pt5d1_signage_v2.txt";
writematrix(keyArr, fn, 'Delimiter', 'tab')