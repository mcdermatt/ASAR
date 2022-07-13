%Matt McDermott
% File made to extract ground truth poses from drive 01 dataset

%load GNSS/INS file
fn  = 'Pose-Applanix.log';
POSE = load_pose_applanix(fn);

data = [];
count = 1;

load('E:\Ford\IJRR-Dataset-1\SCANS\Scan1123.mat');
tlast = SCAN.timestamp_laser;

files = dir('*.mat');
for file = files'
    %Load Lidar file
    count
    load(file.name);
    tsl = SCAN.timestamp_laser;
    
    tsl - tlast %~1e5;
    tlast = tsl;

    %find timestamp/ index of POSE.utime that coincides with lidar frame
    closest_idx = find(abs(tsl - POSE.utime) == min(abs(tsl - POSE.utime)));
    closest_idx = closest_idx(1); %just keep the first entry
    
    pos_i = POSE.pos(closest_idx, :);
    vel_i = POSE.vel(closest_idx, :);
    rot_i = quat2eul(POSE.orientation(closest_idx, :));
    drot_i = POSE.rotation_rate(closest_idx, :);
    
    %return data in world frame (ENU??)
%     data_i = [pos_i, vel_i, rot_i, drot_i];

    %only consider magnitude of velocity in forward movement
    vf = sqrt(vel_i(1)^2 + vel_i(2)^2); %forward velocity
%     data_i = [vf, 0, vel_i(3), drot_i];
    data_i = [0, vf, vel_i(3), drot_i];


    %test
%     data_i = [vel_i];

    data = [data; data_i];
    count = count + 1;
end

writematrix(data, "truth.txt", 'Delimiter', 'tab')