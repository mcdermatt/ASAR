%test script to get WLS 

clear all

PM_ins = eye(21)*1e-19;
PM_lidar = eye(3)*1e-19;

xHatM_ins_cum = 1e-6*ones(21,1);
xHatM_ins_cum(3) = 0.1;
xHatM_lidar_cum = 1e-6*ones(3,1);
xHatM_lidar_cum(3) = 0.1; 

% PM_block = [PM_ins, zeros(21,3); zeros(3,21), PM_lidar];
% W = pinv(PM_block);
% H = [eye(21); eye(3), zeros(3,18)];

% (zeros don't make a difference)
PM_block = [PM_ins, zeros(21); 
            zeros(21), [PM_lidar, zeros(3,18); zeros(18,21)]];
% PM_block = [PM_ins, -PM_ins; 
%             -PM_ins, [PM_lidar, zeros(3,18); zeros(18,21)]];
W = pinv(PM_block);
% H = [eye(21); eye(21)];
H = [eye(21); eye(3), zeros(3,18); zeros(18,21)];
xHatM_lidar_cum = [2e-6*ones(3,1); zeros(18,1)] ;


PM_combined = pinv(H.' * W *H);
%update cumulative counts for this segment to include current xHatM estimates 
xHatM_combined = pinv((H.')*W*H)*(H.')*W*[xHatM_ins_cum; xHatM_lidar_cum]; 

test = pinv((H.')*W*H)*(H.')*W;
% test(1,1) %weight of ins
% test(1,22) %weight of lidar

'PM_lidar'
PM_lidar(1,1)
'PM_ins'
PM_ins(1,1)
'PM_combined'
PM_combined(1,1)