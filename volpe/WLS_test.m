%test script to get WLS 

clear all

PM_ins = eye(21)*10e-18;
PM_lidar = eye(3)*10e-19;

xHatM_ins_cum = 1e-6*ones(21,1);
xHatM_ins_cum(3) = 0.1;
xHatM_lidar_cum = 2e-6*ones(3,1);
xHatM_lidar_cum(3) = 0.1; 

PM_combined = [PM_ins, zeros(21,3); zeros(3,21), PM_lidar];

W = pinv(PM_combined); %supposed to be this

H = [eye(21); eye(3), zeros(3,18)];

%         should be this???
%update cumulative counts for this segment to include current xHatM estimates 
xHatM_combined = pinv((H.')*W*H)*(H.')*W*[xHatM_ins_cum; xHatM_lidar_cum]; 

W(1,1)
W(22,22)
xHatM_combined(1:2)