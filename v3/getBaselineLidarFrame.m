
%I don't trust the baseline provided by PyKitti for mm level precision,
%   processing the raw OXTS data myself to see if I get the same results

clear all
close all

%Rotation vector from IMU to Velodyne unit
R  = [9.999976e-01 7.553071e-04 -2.035826e-03;
     -7.854027e-04 9.998898e-01 -1.482298e-02;
      2.024406e-03 1.482454e-02 9.998881e-01];

%load OXTS data from file
basedir = 'C:/kitti/2011_09_26/2011_09_26_drive_0005_sync';
oxts = loadOxtsliteData(basedir);
tsdir = 'C:/kitti/2011_09_26/2011_09_26_drive_0005_sync/oxts';
ts = loadTimestamps(tsdir);

pose = convertOxtsToPose(oxts);

transvec = [];
velxvec = [];
velyvec = [];
tsvec = [];
for i = 1:150
    transvec = [transvec, pose{i}(1:3,4)];

    sample = oxts{i};
    velx = sample(9);
    vely = sample(10);

    ts_string = ts{i};
    ts_string_s = ts_string(18:25);
    ts_val = str2double(ts_string_s);
    tsvec = [tsvec, ts_val];

%     velxvec = [velxvec, velx];
%     velyvec = [velyvec, vely];

end


% figure(1)
% plot(velxvec)
% 
% figure(2)
% plot(velyvec)

dtvec = diff(tsvec)

% dtvec = [tsvec(2:end), 0] -tsvec;
% dtvec = dtvec(1:end-1);
velx = diff(transvec(1,:))./dtvec.*0.1;
vely = diff(transvec(2,:))./dtvec.*0.1;
figure()
plot(velx) 
figure()
plot(vely)

vf = sqrt(velx.^2 + vely.^2);
figure()
plot(vf)

writematrix(vf, "vf.txt", "Delimiter", " ")

%test- convert LLA -> ENU from raw OXTS
% lla0 = oxts(1,1);
% lla0 = lla0{1}(1:3);
% lla1 = oxts(1,2);
% lla1 = lla1{1}(1:3);
% 
% ENU = lla2enu(lla1, lla0, 'flat')
% 
% forward = sqrt(ENU(1)^2 + ENU(2)^2)*1.037