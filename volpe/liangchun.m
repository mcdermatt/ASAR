% GPS_INS_SPAN Loosely Coupled
clear all
beep off

% User defined paramaters--------------------------------------------------
pureINS = 0;                    % Set False for GPS/INS fusion
addpath('./Data/SPAN');         % Path for data 
% dat = load('signageData.mat');  % Data file
dat = load('roundaboutData.mat');  
endIndex = 1e6;                 % Example 1e4 is 50 sec of data @ 200 Hz                                % If endIndex exceeds max, then reset to max
% endIndex = 2e5;

%--------------------------------------------------------------------------

% startGPS = %1109;                %ignore useless data before here
startGPS = 1;

% Clean workspace
close all
clc

% Rename variables
rawimusx = dat.rawimusx;
ppptime = dat.ppppos.sow1538;
ppplat = dat.ppppos.ppplat;
ppplon = dat.ppppos.ppplon;
ppphgt = dat.ppppos.ppphgt;
ppplatstd = dat.ppppos.ppplatstd;
ppplonstd = dat.ppppos.ppplonstd;
ppphgtstd = dat.ppppos.ppphgtstd;
gpspos = [ppptime ppplat ppplon ppphgt ppplatstd ppplonstd ppphgtstd];

gpspos = gpspos(startGPS:end, :);
% gpspos(:,1) = gpspos(:,1) - gpspos(1,1);

%remove outages from gpspos
nonzero = find(ppplon);
gpspos = gpspos(nonzero, :);

% Apply transducer constants for IMU
ax = double(rawimusx.rawimuxaccel)*(0.400/65536)*(9.80665/1000)/200;
ay = double(rawimusx.rawimuyaccel)*(0.400/65536)*(9.80665/1000)/200;
az = double(rawimusx.rawimuzaccel)*(0.400/65536)*(9.80665/1000)/200;
gx = double(rawimusx.rawimuxgyro)*0.0151515/65536/200;
gy = double(rawimusx.rawimuygyro)*0.0151515/65536/200;
gz = double(rawimusx.rawimuzgyro)*0.0151515/65536/200;

% Pacakge IMU data
imutime = rawimusx.rawimugnsssow;
%was this
% imuraw = [imutime deg2rad(gy) deg2rad(gx) -deg2rad(gz) ay ax -az];  % imuraw = [imutime gx gy gz ax ay az];
% imuraw = [imutime deg2rad(gx) deg2rad(gy) deg2rad(gz) ax ay az];  % imuraw = [imutime gx gy gz ax ay az];
%changed to this
imuraw = [imutime deg2rad(gy) -deg2rad(gx) -deg2rad(gz) ay -ax -az]; %best soln when not fusing qbn_lidar


[minVal, ind] = min(abs(dat.inspvax.sow1465(1) - imuraw(:,1)));
imuraw = imuraw(ind:end,:);
imutime = imutime(ind:end,:);
[minVal, ind] = min(abs(dat.inspvax.sow1465(1) - gpspos(:,1)));
gpspos = gpspos(ind:end,:);                                         % trim data > start at first GPS sample
ppptime = gpspos(:,1);                    

% Prevent analysis from running past end of data record
maxGPStime = ppptime(end-1);
maxIMUtime = imutime(end);
maxTime = min(maxGPStime,maxIMUtime);
maxIMUindx = max(find(imutime<=maxTime));
if endIndex > maxIMUindx     % Truncate IMU record if necessary
    endIndex = maxIMUindx;  
end
% Compute length of GPS record, whether or not IMU record is truncated
imuEndTime = imutime(endIndex);
gpsEndIndx = max(find(ppptime<=imuEndTime));

% Initialization of Arrays
xHatP = zeros(21,1);            % state estimate
ba = zeros(3,1);                % bias accels
bg = zeros(3,1);                % bias gyros
sa = zeros(3,1);                % scale factor accels
sg = zeros(3,1);                % scale factor gyros
PP = PPinitialize;              % state-error matrix
PP = PP*1e6;                    % Set initial covariance matrix to be large (indicating initial uncertainty)
if pureINS; Ncol = 10; else Ncol = 22; end
msrArr = zeros(7,endIndex-2);        % INS measurement array
resArr = zeros(endIndex-2,Ncol-1);   % Storage array of state-perturbation error results
qbnArr = zeros(endIndex-2,4);        % Storage of quaternion results (relating Body to Inertial)
nedArr = zeros(endIndex-2,3);        % Storage of position results (NED)
gpsMsrArr = zeros(4,gpsEndIndx);    % GPS measurement
gps_res = zeros(gpsEndIndx,4);

% INITIALIZATION of EKF
lla0 = [deg2rad(gpspos(1,2)), deg2rad(gpspos(1,3)), gpspos(1,4)];   % from gps.txt
% lla0 = [deg2rad(dat.inspvax.inslat(1)), deg2rad(dat.inspvax.inslon(1)), dat.inspvax.inshgt(1)];   % from gps.txt
vel0 = [dat.inspvax.insnorthvel(1), dat.inspvax.inseastvel(1), -dat.inspvax.insupvel(1)];
rpy0 = deg2rad([dat.inspvax.insroll(1), dat.inspvax.inspitch(1), dat.inspvax.insazim(1)]); 
% qbn0_enu = eul2quat(rpy0);
rpy0(3) = 2*pi-rpy0(3);
dv0 = 0;

qbn0_enu = eul2quat(rpy0);     %Initialize with RPY 
rpy0_inverse = quat2eul(qbn0_enu);

rpy0 - rpy0_inverse;
qbn0_ned = [qbn0_enu(1) qbn0_enu(3) qbn0_enu(4) -qbn0_enu(2)];
qbn0 = qbn0_ned;
bias = 0;

% Seed initial measurements
msrk = imuraw(1,:);
msrk = msrk';
if pureINS == 0
    gpsmsr = gpspos(1,:);
end

corr_hist = zeros(endIndex, 3); %init matrix to store correction vectors for each timestep

%ONLY KEEP EVERY 3rd GPS MEASUREMNT (FOR SOLUTION SEPERATION DEMO)
% gpspos = gpspos(1:3:end,:);

% Start Loop
gpsCnt = 0;
i = 2;
while i < endIndex  
    if(mod(i,1e3)==2); i-2, end;  % echo time step to screen current step
    
%     msrk1 = [imuraw(i), imuraw(221800 + i, 2:end)];
    msrk1 = imuraw(i + startGPS*200, :);
%     msrk1 = imuraw(i,:);
    msrk1 = msrk1';
    
    if isempty(msrk1) 
        break;
    end
    if pureINS == 0 && isempty(gpsmsr)
        break;
    end  
    
    if i == 1132
        disp("reach here");
    end
       
    dt = msrk1(1) - msrk(1);
    msrk1(7) =  msrk1(7) - bias*dt; 

    if pureINS == 1
        [lla, vel, rpy, dv, qbn] = ins_mechanization_compact(lla0, vel0, rpy0, dv0, qbn0, msrk, msrk1);
        % Store Data
        resArr(i-1,:) = [lla vel rpy];
        qbnArr(i-1,:) = qbn;
    
    else
        % GPS/INS solution
        [lla, vel, rpy, ned, dv, qbn, xHatP, PP, gpsUpdated, gpsCnt, gpsResUpd, corr] = ins_gps(lla0, vel0, rpy0, dv0, qbn0, msrk, msrk1, gpsmsr, xHatP, PP, gpsCnt);
        corr_hist(i,:) = corr; %update correction history
        if gpsUpdated == 1
            gpsmsr = gpspos(gpsCnt+1, :);   
            gps_res(gpsCnt,:) = gpsResUpd;
            xHatP(1:9) = zeros(9,1);        % Reset error states
        end
        % Store Data
        resArr(i-1,:) = [lla vel rpy rad2deg(bg')*3600 (ba')*10^6/9.7803267715 (sg')*10^-6 (sa')*10^-6];
        qbnArr(i-1,:) = qbn;
        nedArr(i-1,:) = ned;
        if gpsUpdated == 1 && ~isempty(gpsmsr)
            gpsMsrArr(:,gpsCnt) = gpsmsr(1:4)';
        end
    end
    
    msrArr(:,i-1) = msrk1;
    lla0 = lla; vel0 = vel; rpy0 = rpy;
    dv0 = dv; qbn0 = qbn;
    msrk = msrk1;    
   
    i = i+1;  
end 

% plots
time = msrArr(1,:);
startTime = time(1);

% % Raw Gyro data
% figure(3)
% subplot(3,1,1);
% plot(time-startTime, rad2deg(msrArr(2,:)), '.-');
% hold on; grid on
% ylabel('{{\Delta}{\theta}_x} (deg)');
% subplot(3,1,2);
% plot(time-startTime, rad2deg(msrArr(3,:)), '.-');
% hold on; grid on
% ylabel('{{\Delta}{\theta}_y} (deg)');
% subplot(3,1,3);
% plot(time-startTime, rad2deg(msrArr(4,:)), '.-');
% hold on; grid on
% ylabel('{{\Delta}{\theta}_z} (deg)');
% xlabel('Time (s)');
% 
% % Raw Accel data
% figure(4)
% subplot(3,1,1);
% plot(time-startTime, msrArr(5,:), '.-');
% hold on; grid on
% ylabel('{{\Delta}v_x} (m/s)');
% subplot(3,1,2);
% plot(time-startTime, msrArr(6,:), '.-');
% hold on; grid on
% ylabel('{{\Delta}v_y} (m/s)');
% subplot(3,1,3);
% plot(time-startTime, msrArr(7,:), '.-');
% hold on; grid on
% ylabel('{{\Delta}v_z} (m/s)');

len = length(time);

% Velocity Plots
figure(5)
subplot(3,1,1);
plot(time-startTime, resArr(:,4));
hold on; grid on
ylabel('velocity x (m/s)');
subplot(3,1,2);
plot(time-startTime, resArr(:,5));
hold on; grid on
ylabel('velocity y (m/s)');
subplot(3,1,3);
plot(time-startTime, resArr(:,6));
hold on; grid on
ylabel('velocity z (m/s)');
xlabel('Time (s)');

% Postion Plots
figure(6)
subplot(3,1,1);
plot(time-startTime, rad2deg(resArr(:,1)));
hold on; grid on
ylabel('latitude (deg)');
subplot(3,1,2);
plot(time-startTime, rad2deg(resArr(:,2)));
hold on; grid on
ylabel('longitude (deg)');
subplot(3,1,3);
plot(time-startTime, resArr(:,3));
hold on; grid on

len = length(time);
ylabel('altitude (m)');
xlabel('Time (s)');

if pureINS == 0
    subplot(3,1,1);
    plot(gpsMsrArr(1, :)-startTime, gpsMsrArr(2, :));
    hold on; grid on
    ylabel('latitude (deg)');
    subplot(3,1,2);
    plot(gpsMsrArr(1, :)-startTime, gpsMsrArr(3, :));
    hold on; grid on
    ylabel('longitude (deg)');
    subplot(3,1,3);
    plot(gpsMsrArr(1, :)-startTime, gpsMsrArr(4, :));
    hold on; grid on
    ylabel('altitude (m)');
    xlabel('Time (s)');
end

lat_lc = rad2deg(resArr(:,1));
lon_lc = rad2deg(resArr(:,2));
alt_lc = resArr(:,3);

%%% IS THIS NEXT CODE SNIPPET (through "len") USED? %%%
kmlFolder = './';

filename = fullfile(kmlFolder,'urbanloco_lc.kml');
% kmlwritepoint(filename,lat_lc,lon_lc,alt_lc, 'IconScale',1);

lat_gps = gpsMsrArr(2, :);
lon_gps = gpsMsrArr(3, :);
alt_gps = gpsMsrArr(4, :);

filename = fullfile(kmlFolder,'urbanloco_gps.kml');
% kmlwritepoint(filename,lat_gps,lon_gps,alt_gps, 'Color','red', 'IconScale',1);

len = length(qbnArr(:,1));

rpyRef = deg2rad([dat.inspvax.insroll, dat.inspvax.inspitch, dat.inspvax.insazim]);
% qbnRef_enu = eul2quat(flip(rpyRef')');     % Initialize with RPY solutoin
% rpyRef_enu = flip(quat2eul(qbnRef_enu)')';
qbnRef_enu = eul2quat(rpyRef);     % Initialize with RPY solutoin
rpyRef_enu = quat2eul(qbnRef_enu);
rpyRefDiff = rpyRef_enu - rpyRef;
zeroInd1 = find(abs(rpyRefDiff(:,3)) < 2);

rpyRef_var = deg2rad([dat.inspvax.insroll, dat.inspvax.inspitch, dat.inspvax.insazim]);
rpyRef_var(:,3) = 2*pi-rpyRef_var(:,3);
% qbnRef_enu_var = eul2quat(flip(rpyRef_var')');     % Initialize with RPY solutoin
% rpyRef_enu_var = flip(quat2eul(qbnRef_enu_var)')';
qbnRef_enu_var = eul2quat(rpyRef_var);     % Initialize with RPY solutoin
rpyRef_enu_var = quat2eul(qbnRef_enu_var);

rpyRefDiff_var = rpyRef_enu_var - rpyRef_var;
zeroInd2 = find(abs(rpyRefDiff_var(:,3)) < 2);

% qbnRef_enu(zeroInd2) = qbnRef_enu_var(zeroInd2);
rpyRef(zeroInd2,:) = rpyRef_var(zeroInd2,:);
% qbnRef_enu = eul2quat(flip(rpyRef')');
% rpyRef_enu = flip(quat2eul(qbnRef_enu)')';
qbnRef_enu = eul2quat(rpyRef);
rpyRef_enu = quat2eul(qbnRef_enu);
rpyRef_enu - rpyRef

% qbnRef_ned = [qbnRef_enu(:,1) qbnRef_enu(:,3) qbnRef_enu(:,2) -qbnRef_enu(:,4)];
qbnRef_ned = [qbnRef_enu(:,1) qbnRef_enu(:,3) qbnRef_enu(:,4) -qbnRef_enu(:,2)];

% for i = 1:length(qbnRef_ned)-1
%     qbnk = qbnRef_ned(i);
%     qbnk1 = qbnRef_ned(i+1);
%     % flip the sign if two quaternions are opposite in sign
%     if (norm((qbnk+qbnk1)/2) < 0.05 && norm((qbnk-qbnk1)/2) > 0.95)
%         qbnRef_ned(i+1) = -qbnk1;
%     end
% end

qbnArr = [qbn0_ned; qbnArr];
qbnRef = qbnRef_ned;
qbnTime = imutime(1:endIndex-1);
qbnRefTime = dat.inspvax.sow1465;
t0 = min(qbnRefTime(1), qbnTime(1));
qbnTime = qbnTime - t0;
qbnRefTime = qbnRefTime - t0;
% Quaternion plot
figure(8)
fg1 = subplot(4,1,1);
plot(qbnTime, qbnArr(:,1));
hold on; grid on
plot(qbnRefTime, qbnRef(:,1));
ylabel('{q_1}');
fg2 = subplot(4,1,2);
plot(qbnTime, qbnArr(:,2));
hold on; grid on
plot(qbnRefTime, qbnRef(:,2));
ylabel('{q_2}');
fg3 = subplot(4,1,3);
plot(qbnTime, qbnArr(:,3));
hold on; grid on
plot(qbnRefTime, qbnRef(:,3));
ylabel('{q_3}');
fg4 = subplot(4,1,4);
plot(qbnTime, qbnArr(:,4));
hold on; grid on
plot(qbnRefTime, qbnRef(:,4));
ylabel('{q_4}');
linkaxes([fg1,fg2,fg3,fg4]);
xlim([0 qbnTime(end)]);

% qbnArr_enu = [qbnArr(:,1) qbnArr(:,3) qbnArr(:,2) -qbnArr(:,4)];
qbnArr_enu = [qbnArr(:,1) qbnArr(:,3) qbnArr(:,4) -qbnArr(:,2)];
% rpyArr = flip(quat2eul(qbnArr_enu)')';
rpyArr = quat2eul(qbnArr_enu);
% Quaternion plot
% figure(9)
% fg1 = subplot(3,1,1);
% plot(qbnTime, rad2deg(rpyArr(:,1)));
% hold on; grid on
% plot(rad2deg(rpyRef(:,1)));
% plot(dat.inspvax.insroll);
% ylabel('{Roll (deg)}');
% fg2 = subplot(3,1,2);
% plot(qbnTime, rad2deg(rpyArr(:,2)));
% hold on; grid on
% plot(rad2deg(rpyRef(:,2)));
% plot(dat.inspvax.inspitch);
% ylabel('{Pitch (deg)}');
% fg3 = subplot(3,1,3);
% plot(qbnTime, rad2deg(rpyArr(:,3)));
% hold on; grid on
% plot(rad2deg(rpyRef(:,3)));
% plot(dat.inspvax.insazim);
% ylabel('{Azimuth (deg)}');
% linkaxes([fg1,fg2,fg3]);
% xlim([0 qbnTime(end)]);

% Quaternion plot
figure(7)
subplot(4,1,1);
plot(qbnArr(:,1));
hold on; grid on
% plot(qbnRef(1:len,4));
ylabel('{q_1}');
subplot(4,1,2);
plot(qbnArr(:,2));
hold on; grid on
% plot(qbnRef(1:len,2));
ylabel('{q_2}');
subplot(4,1,3);
plot(qbnArr(:,3));
hold on; grid on
% plot(qbnRef(1:len,1));
ylabel('{q_3}');
subplot(4,1,4);
plot(qbnArr(:,4));
hold on; grid on
% plot(-qbnRef(1:len,3));
ylabel('{q_4}');


%plot linearized correction factor
figure()
hold on
subplot(3,1,1)
plot(find(corr_hist(:,1)), corr_hist(find(corr_hist(:,1)),1))
ylabel('Lat Correction')
subplot(3,1,2)
plot(find(corr_hist(:,2)), corr_hist(find(corr_hist(:,2)),2))
ylabel('Lon Correction')
subplot(3,1,3)
plot(find(corr_hist(:,3)), corr_hist(find(corr_hist(:,3)),3))
ylabel('Altitude Correction')

% compute rn, rm
WGS84_A = 6378137.0;           % earth semi-major axis (WGS84) (m) 
WGS84_B = 6356752.3142;        % earth semi-minor axis (WGS84) (m) 
e = sqrt(WGS84_A * WGS84_A - WGS84_B * WGS84_B) / WGS84_A;
rn_list = WGS84_A ./ sqrt(1 - e * e * sin(resArr(:,1)).^2);
rm_list = WGS84_A * (1 - e * e) ./ sqrt(power(1 - e * e * sin(resArr(:,1)).^2, 3));



% ins_mechanization_compact.m:
function [lla, vel, rpy, dv, qbn] = ins_mechanization_compact(lla0, vel0, rpy0, dv0, qbn0, msrk, msrk1)

WGS84_A = 6378137.0;           % earth semi-major axis (WGS84) (m) 
WGS84_B = 6356752.3142;        % earth semi-minor axis (WGS84) (m) 
e = sqrt(WGS84_A * WGS84_A - WGS84_B * WGS84_B) / WGS84_A;

% rotational angular velocity of earth
omega_e = 7.2921151467e-5;   

% wgs84 = wgs84Ellipsoid;
dt = msrk1(1) - msrk(1);

% update the velocity
gy0 = msrk(2:4);
ac0 = msrk(5:7);
gy = msrk1(2:4);
ac = msrk1(5:7);
dvfb_k = 1.0/12.0*(cross(gy0,ac) + cross(ac0,gy)) + 0.5*cross(gy,ac);
dvfb_k = ac + dvfb_k;

% compute cbn0
cbn0 = quat2dcm(qbn0);

qne = zeros(1,4);
qne(1) = cos(-pi / 4.0 - lla0(1) / 2.0) * cos(lla0(2) / 2.0);
qne(2) = -sin(-pi / 4.0 - lla0(1) / 2.0) * sin(lla0(2) / 2.0);
qne(3) = sin(-pi / 4.0 - lla0(1) / 2.0) * cos(lla0(2) / 2.0);
qne(4) = cos(-pi / 4.0 - lla0(1) / 2.0) * sin(lla0(2) / 2.0);

qee_h = zeros(1,4); qnn_l = zeros(1,4);

% compute rn, rm
rn = WGS84_A / sqrt(1 - e * e * sin(lla0(1)) * sin(lla0(1)));
rm = WGS84_A * (1 - e * e) / sqrt(power(1 - e * e * sin(lla0(1)) * sin(lla0(1)), 3));

% compute wie
wie(1) = omega_e * cos(lla0(1));
wie(2) = 0;
wie(3) = -omega_e * sin(lla0(1));

% compute wen
wen(1) = vel0(2) / (rn + lla0(3));
wen(2) = -vel0(1) / (rm + lla0(3));
wen(3) = -vel0(2) * tan(lla0(1)) / (rn + lla0(3));
win = wie + wen;

% compute g_l
gl = zeros(1,3);

grav = [9.7803267715, 0.0052790414, 0.0000232718, -0.000003087691089, 0.000000004397731, 0.000000000000721];
sinB = sin(lla0(1));
sinB2 = sinB * sinB;
sinB4 = sinB2 * sinB2;
gl(3) = grav(1) * (1.0 + grav(2) * sinB2 + grav(3) * sinB4) + (grav(4) + grav(5) * sinB2) * lla0(3) + grav(6) * lla0(3) * lla0(3);

zeta_m = dt.*win/2.0;
half_zeta = 0.5*zeta_m;
mold = norm(half_zeta);
sc = sin(mold) / mold;
qnn_l(1) = cos(mold);
qnn_l(2) = sc * half_zeta(1);
qnn_l(3) = sc * half_zeta(2);
qnn_l(4) = sc * half_zeta(3);

wiee = [0 0 omega_e];

xi_m = dt.*wiee/2.0;       % 0.5: half size for extrapolating lla at time(k+1/2) 
half_xi = -0.5*xi_m;
mold = norm(half_xi);
sc = sin(mold) / mold;
qee_h(1) = cos(mold);
qee_h(2) = sc * half_xi(1);
qee_h(3) = sc * half_xi(2);
qee_h(4) = sc * half_xi(3);

qne_l = quatmultiply(qne, qnn_l);
qne_m = quatmultiply(qee_h, qne_l);

lla_m = zeros(1,3);

if qne_m(1) ~= 0
    lla_m(2) = 2 * atan(qne_m(4) / qne_m(1));
    lla_m(1) = 2 * (-pi / 4.0 - atan(qne_m(3) / qne_m(1)));
elseif qne_m(1) == 0 & qne_m(3) == 0
    lla_m(2) = pi;
    lla_m(1) = 2 * (-pi / 4.0 - atan(-qne_m(2) / qne_m(4)));
elseif qne_m(1) == 0 & qne_m(4) == 0
    lla_m(2) = 2 * atan(-qne_m(2) / qne_m(3));
    lla_m(1) = pi / 2.0;
end
    
lla_m(3) = lla0(3) - (vel0(3) * dt) / 2.0;

% extrapolate the speed
vel_m = vel0 + 0.5 * dv0;

% compute the wie_m, wen_m
% compute rn, rm
rn = WGS84_A / sqrt(1 - e * e * sin(lla_m(1)) * sin(lla_m(1)));
rm = WGS84_A * (1 - e * e) / sqrt(power(1 - e * e * sin(lla_m(1)) * sin(lla_m(1)), 3));

% compute wie_m
wie_m(1) = omega_e * cos(lla_m(1));
wie_m(2) = 0;
wie_m(3) = -omega_e * sin(lla_m(1));

% compute wen_m
wen_m(1) = vel_m(2) / (rn + lla_m(3));
wen_m(2) = -vel_m(1) / (rm + lla_m(3));
wen_m(3) = -vel_m(2) * tan(lla_m(1)) / (rn + lla_m(3));
win_m = wie_m + wen_m;

% compute g_l
gl_m = zeros(1,3);
sinB = sin(lla_m(1));
sinB2 = sinB * sinB;
sinB4 = sinB2 * sinB2;
gl_m(3) = grav(1) * (1.0 + grav(2) * sinB2 + grav(3) * sinB4) + (grav(4) + grav(5) * sinB2) * lla0(3) + grav(6) * lla0(3) * lla0(3);

% compute true zeta
zeta = win_m .* dt;
cnn = eye(3,3);

cnn(1,2) = -0.5 * zeta(3);
cnn(1,3) =  0.5 * zeta(2);
cnn(2,1) =  0.5 * zeta(3);
cnn(2,3) = -0.5 * zeta(1);
cnn(3,1) = -0.5 * zeta(2);
cnn(3,2) =  0.5 * zeta(1);
    
% calculate dvfn_k, dvgn_k
% dvfn_k = (cbn0*cnn)'*dvfb_k;              %%%%??????????????????????????????? transpose
dvfn_k = (cbn0'*cnn)*dvfb_k;
dvfn_k = dvfn_k';
dvgn_k = (gl_m-cross((2*wie_m+wen_m), vel_m)).* dt;

% update velocity
dv = dvfn_k + dvgn_k;
vel = vel0 + dv;

% update the position
vel_m = 0.5*(vel0 + vel);

qnn_h = zeros(1,4);
qee_l = zeros(1,4);
    
% recompute the wie_m, wen_m
rn = WGS84_A / sqrt(1 - e * e * sin(lla_m(1)) * sin(lla_m(1)));
rm = WGS84_A * (1 - e * e) / sqrt(power(1 - e * e * sin(lla_m(1)) * sin(lla_m(1)), 3));

% compute wie_m
wie_m(1) = omega_e * cos(lla_m(1));
wie_m(2) = 0;
wie_m(3) = -omega_e * sin(lla_m(1));

% compute wen_m
wen_m(1) = vel_m(2) / (rn + lla_m(3));
wen_m(2) = -vel_m(1) / (rm + lla_m(3));
wen_m(3) = -vel_m(2) * tan(lla_m(1)) / (rn + lla_m(3));
win_m = wie_m + wen_m;

% compute g_l
gl_m = zeros(1,3);
sinB = sin(lla_m(1));
sinB2 = sinB * sinB;
sinB4 = sinB2 * sinB2;
gl_m(3) = grav(1) * (1.0 + grav(2) * sinB2 + grav(3) * sinB4) + (grav(4) + grav(5) * sinB2) * lla0(3) + grav(6) * lla0(3) * lla0(3);

% recompute zeta, xi
zeta_m = win_m.*dt;
half_zeta = 0.5*zeta_m;
mold = norm(half_zeta);
sc = sin(mold) / mold;
qnn_h(1) = cos(mold);
qnn_h(2) = sc * half_zeta(1);
qnn_h(3) = sc * half_zeta(2);
qnn_h(4) = sc * half_zeta(3);

xi_m = wiee.* dt;
half_xi = -0.5*xi_m;
mold = norm(half_xi);
sc = sin(mold) / mold;
qee_l(1) = cos(mold);
qee_l(2) = sc * half_xi(1);
qee_l(3) = sc * half_xi(2);
qee_l(4) = sc * half_xi(3);

% recompute the qnn_h
qne_h = quatmultiply(qne, qnn_h);
qne = quatmultiply(qee_l, qne_h);

lla = zeros(1,3);

if qne(1) ~= 0
    lla(2) = 2 * atan(qne(4) / qne(1));
    lla(1) = 2 * (-pi / 4.0 - atan(qne(3) / qne(1)));
elseif qne(1) == 0 & qne(3) == 0
    lla(2) = pi;
    lla(1) = 2 * (-pi / 4.0 - atan(-qne(2) / qne(4)));
elseif qne(1) == 0 & qne(4) == 0
    lla(2) = 2 * atan(-qne(2) / qne(3));
    lla(1) = pi / 2.0;
end
    
lla(3) = lla0(3) - vel_m(3) * dt;

% update the attitude
qdthe_half = zeros(1,4);
qne0 = zeros(1,4);

% compute qne0
qne0(1) = cos(-pi / 4.0 - lla0(1) / 2.0) * cos(lla0(2) / 2.0);
qne0(2) = -sin(-pi / 4.0 - lla0(1) / 2.0) * sin(lla0(2) / 2.0);
qne0(3) = sin(-pi / 4.0 - lla0(1) / 2.0) * cos(lla0(2) / 2.0);
qne0(4) = cos(-pi / 4.0 - lla0(1) / 2.0) * sin(lla0(2) / 2.0);

qneo_inv = quatinv(qne0);

qdthe = quatmultiply(qneo_inv, qne)';

vec = zeros(1,3);

if qdthe(1) ~= 0
    phi_m = atan(sqrt(qdthe(2)*qdthe(2)+qdthe(3)*qdthe(3)+qdthe(4)*qdthe(4))/qdthe(1));
    f = 0.5 * sin(phi_m) / phi_m;
    vec(1) = qdthe(2) / f;
    vec(2) = qdthe(3) / f;
    vec(3) = qdthe(4) / f;
else
    vec(1:3) = pi * qdthe(2:4);
end

vec_half = 0.5 * vec;

v = 0.5 * vec_half;
mold = norm(v, 3);
sc = sin(mold) / mold;
qdthe_half(1) = cos(mold);
qdthe_half(2) = sc * v(1);
qdthe_half(3) = sc * v(2);
qdthe_half(4) = sc * v(3);
    
qne_m = quatmultiply(qne0, qdthe_half);        % communication law
lla_m = zeros(1,3);

if qne_m(1) ~= 0
    lla_m(2) = 2 * atan(qne_m(4) / qne_m(1));
    lla_m(1) = 2 * (-pi / 4.0 - atan(qne_m(3) / qne_m(1)));
elseif qne_m(1) == 0 & qne_m(3) == 0
    lla_m(2) = pi;
    lla_m(1) = 2 * (-pi / 4.0 - atan(-qne_m(2) / qne_m(4)));
elseif qne_m(1) == 0 & qne_m(4) == 0
    lla_m(2) = 2 * atan(-qne_m(2) / qne_m(3));
    lla_m(1) = pi / 2.0;
end

lla_m(3) = (lla0(3) + lla(3)) / 2.0;

% compute rn, rm
rn = WGS84_A / sqrt(1 - e * e * sin(lla_m(1)) * sin(lla_m(1)));
rm = WGS84_A * (1 - e * e) / sqrt(power(1 - e * e * sin(lla_m(1)) * sin(lla_m(1)), 3));

% compute wie_m
wie_m(1) = omega_e * cos(lla_m(1));
wie_m(2) = 0;
wie_m(3) = -omega_e * sin(lla_m(1));

% compute wen_m
wen_m(1) = vel_m(2) / (rn + lla_m(3));
wen_m(2) = -vel_m(1) / (rm + lla_m(3));
wen_m(3) = -vel_m(2) * tan(lla_m(1)) / (rn + lla_m(3));
win_m = wie_m + wen_m;

% compute g_l
gl_m = zeros(1,3);
sinB = sin(lla_m(1));
sinB2 = sinB * sinB;
sinB4 = sinB2 * sinB2;
gl_m(3) = grav(1) * (1.0 + grav(2) * sinB2 + grav(3) * sinB4) + (grav(4) + grav(5) * sinB2) * lla0(3) + grav(6) * lla0(3) * lla0(3);

% recompute zeta, xi
phi = gy+1/12*cross(gy0, gy);
half_phi = 0.5*phi;
mold = norm(half_phi);
sc = sin(mold) / mold;
qbb(1) = cos(mold);
qbb(2) = sc * half_phi(1);
qbb(3) = sc * half_phi(2);
qbb(4) = sc * half_phi(3);

zeta = win_m.*dt;
half_zeta = -0.5*zeta;
mold = norm(half_zeta);
sc = sin(mold) / mold;
qnn(1) = cos(mold);
qnn(2) = sc * half_zeta(1);
qnn(3) = sc * half_zeta(2);
qnn(4) = sc * half_zeta(3);

tmp_q = quatmultiply(qbn0, qbb);
qbn = quatmultiply(qnn, tmp_q);

% normalization
% e_q = sumsqr(qbn);
% e_q = 1 - 0.5 * (e_q - 1);
% qbn = e_q.*qbn;
qbn = quatnormalize(qbn);

% rpy = flip(quat2eul(qbn));
rpy = quat2eul(qbn);
end


% ins_mechanization_compact.m:
function [lla, vel, rpy, ned, dv, qbn, xHatP, PP, gpsUpdated, gpsCnt, gpsResUpd, corr] = ins_gps(lla0, vel0, rpy0, dv0, qbn0, msrk, msrk1, gpsmsr, xHatP, PP, gpsCnt)

WGS84_A = 6378137.0;           % earth semi-major axis (WGS84) (m) 
WGS84_B = 6356752.3142;        % earth semi-minor axis (WGS84) (m) 
e = sqrt(WGS84_A * WGS84_A - WGS84_B * WGS84_B) / WGS84_A;

% rotational angular velocity of earth
omega_e = 7.2921151467e-5;   

% wgs84 = wgs84Ellipsoid;
dt = msrk1(1) - msrk(1);

% update the velocity
gy0 = msrk(2:4);
ac0 = msrk(5:7);
gy = msrk1(2:4);
ac = msrk1(5:7);
dvfb_k = 1.0/12.0*(cross(gy0,ac) + cross(ac0,gy)) + 0.5*cross(gy,ac);
dvfb_k = ac + dvfb_k;

% compute cbn0
cbn0 = quat2dcm(qbn0);

qne = zeros(1,4);
qne(1) = cos(-pi / 4.0 - lla0(1) / 2.0) * cos(lla0(2) / 2.0);
qne(2) = -sin(-pi / 4.0 - lla0(1) / 2.0) * sin(lla0(2) / 2.0);
qne(3) = sin(-pi / 4.0 - lla0(1) / 2.0) * cos(lla0(2) / 2.0);
qne(4) = cos(-pi / 4.0 - lla0(1) / 2.0) * sin(lla0(2) / 2.0);

qee_h = zeros(1,4); qnn_l = zeros(1,4);

% compute rn, rm
rn = WGS84_A / sqrt(1 - e * e * sin(lla0(1)) * sin(lla0(1)));
rm = WGS84_A * (1 - e * e) / sqrt(power(1 - e * e * sin(lla0(1)) * sin(lla0(1)), 3));

% compute wie
wie(1) = omega_e * cos(lla0(1));
wie(2) = 0;
wie(3) = -omega_e * sin(lla0(1));

% compute wen
wen(1) = vel0(2) / (rn + lla0(3));
wen(2) = -vel0(1) / (rm + lla0(3));
wen(3) = -vel0(2) * tan(lla0(1)) / (rn + lla0(3));
win = wie + wen;

% compute g_l
gl = zeros(1,3);

grav = [9.7803267715, 0.0052790414, 0.0000232718, -0.000003087691089, 0.000000004397731, 0.000000000000721];
sinB = sin(lla0(1));
sinB2 = sinB * sinB;
sinB4 = sinB2 * sinB2;
gl(3) = grav(1) * (1.0 + grav(2) * sinB2 + grav(3) * sinB4) + (grav(4) + grav(5) * sinB2) * lla0(3) + grav(6) * lla0(3) * lla0(3);

zeta_m = dt.*win/2.0;
half_zeta = 0.5*zeta_m;
mold = norm(half_zeta);
sc = sin(mold) / mold;
qnn_l(1) = cos(mold);
qnn_l(2) = sc * half_zeta(1);
qnn_l(3) = sc * half_zeta(2);
qnn_l(4) = sc * half_zeta(3);

wiee = [0 0 omega_e];

xi_m = dt.*wiee/2.0;       % 0.5: half size for extrapolating lla at time(k+1/2) 
half_xi = -0.5*xi_m;
mold = norm(half_xi);
sc = sin(mold) / mold;
qee_h(1) = cos(mold);
qee_h(2) = sc * half_xi(1);
qee_h(3) = sc * half_xi(2);
qee_h(4) = sc * half_xi(3);

qne_l = quatmultiply(qne, qnn_l);
qne_m = quatmultiply(qee_h, qne_l);

lla_m = zeros(1,3);

if qne_m(1) ~= 0
    lla_m(2) = 2 * atan(qne_m(4) / qne_m(1));
    lla_m(1) = 2 * (-pi / 4.0 - atan(qne_m(3) / qne_m(1)));
elseif qne_m(1) == 0 & qne_m(3) == 0
    lla_m(2) = pi;
    lla_m(1) = 2 * (-pi / 4.0 - atan(-qne_m(2) / qne_m(4)));
elseif qne_m(1) == 0 & qne_m(4) == 0
    lla_m(2) = 2 * atan(-qne_m(2) / qne_m(3));
    lla_m(1) = pi / 2.0;
end
    
lla_m(3) = lla0(3) - (vel0(3) * dt) / 2.0;

% extrapolate the speed
vel_m = vel0 + 0.5 * dv0;

% compute the wie_m, wen_m
% compute rn, rm
rn = WGS84_A / sqrt(1 - e * e * sin(lla_m(1)) * sin(lla_m(1)));
rm = WGS84_A * (1 - e * e) / sqrt(power(1 - e * e * sin(lla_m(1)) * sin(lla_m(1)), 3));

% compute wie_m
wie_m(1) = omega_e * cos(lla_m(1));
wie_m(2) = 0;
wie_m(3) = -omega_e * sin(lla_m(1));

% compute wen_m
wen_m(1) = vel_m(2) / (rn + lla_m(3));
wen_m(2) = -vel_m(1) / (rm + lla_m(3));
wen_m(3) = -vel_m(2) * tan(lla_m(1)) / (rn + lla_m(3));
win_m = wie_m + wen_m;

% compute g_l
gl_m = zeros(1,3);
sinB = sin(lla_m(1));
sinB2 = sinB * sinB;
sinB4 = sinB2 * sinB2;
gl_m(3) = grav(1) * (1.0 + grav(2) * sinB2 + grav(3) * sinB4) + (grav(4) + grav(5) * sinB2) * lla0(3) + grav(6) * lla0(3) * lla0(3);

% compute true zeta
zeta = win_m .* dt;
cnn = eye(3,3);

cnn(1,2) = -0.5 * zeta(3);
cnn(1,3) =  0.5 * zeta(2);
cnn(2,1) =  0.5 * zeta(3);
cnn(2,3) = -0.5 * zeta(1);
cnn(3,1) = -0.5 * zeta(2);
cnn(3,2) =  0.5 * zeta(1);
    
% calculate dvfn_k, dvgn_k
% dvfn_k = (cbn0*cnn)'*dvfb_k;              %%%%??????????????????????????????? transpose
dvfn_k = (cbn0'*cnn)*dvfb_k;
dvfn_k = dvfn_k';
dvgn_k = (gl_m-cross((2*wie_m+wen_m), vel_m)).* dt;

% update velocity
dv = dvfn_k + dvgn_k;
vel = vel0 + dv;

% update the position
vel_m = 0.5*(vel0 + vel);

qnn_h = zeros(1,4);
qee_l = zeros(1,4);
    
% recompute the wie_m, wen_m
rn = WGS84_A / sqrt(1 - e * e * sin(lla_m(1)) * sin(lla_m(1)));
rm = WGS84_A * (1 - e * e) / sqrt(power(1 - e * e * sin(lla_m(1)) * sin(lla_m(1)), 3));

% compute wie_m
wie_m(1) = omega_e * cos(lla_m(1));
wie_m(2) = 0;
wie_m(3) = -omega_e * sin(lla_m(1));

% compute wen_m
wen_m(1) = vel_m(2) / (rn + lla_m(3));
wen_m(2) = -vel_m(1) / (rm + lla_m(3));
wen_m(3) = -vel_m(2) * tan(lla_m(1)) / (rn + lla_m(3));
win_m = wie_m + wen_m;

% compute g_l
gl_m = zeros(1,3);
sinB = sin(lla_m(1));
sinB2 = sinB * sinB;
sinB4 = sinB2 * sinB2;
gl_m(3) = grav(1) * (1.0 + grav(2) * sinB2 + grav(3) * sinB4) + (grav(4) + grav(5) * sinB2) * lla0(3) + grav(6) * lla0(3) * lla0(3);

% recompute zeta, xi
zeta_m = win_m.*dt;
half_zeta = 0.5*zeta_m;
mold = norm(half_zeta);
sc = sin(mold) / mold;
qnn_h(1) = cos(mold);
qnn_h(2) = sc * half_zeta(1);
qnn_h(3) = sc * half_zeta(2);
qnn_h(4) = sc * half_zeta(3);

xi_m = wiee.* dt;
half_xi = -0.5*xi_m;
mold = norm(half_xi);
sc = sin(mold) / mold;
qee_l(1) = cos(mold);
qee_l(2) = sc * half_xi(1);
qee_l(3) = sc * half_xi(2);
qee_l(4) = sc * half_xi(3);

% recompute the qnn_h
qne_h = quatmultiply(qne, qnn_h);
qne = quatmultiply(qee_l, qne_h);

lla = zeros(1,3);

if qne(1) ~= 0
    lla(2) = 2 * atan(qne(4) / qne(1));
    lla(1) = 2 * (-pi / 4.0 - atan(qne(3) / qne(1)));
elseif qne(1) == 0 & qne(3) == 0
    lla(2) = pi;
    lla(1) = 2 * (-pi / 4.0 - atan(-qne(2) / qne(4)));
elseif qne(1) == 0 & qne(4) == 0
    lla(2) = 2 * atan(-qne(2) / qne(3));
    lla(1) = pi / 2.0;
end
    
lla(3) = lla0(3) - vel_m(3) * dt;

% update the attitude
qdthe_half = zeros(1,4);
qne0 = zeros(1,4);

% compute qne0
qne0(1) = cos(-pi / 4.0 - lla0(1) / 2.0) * cos(lla0(2) / 2.0);
qne0(2) = -sin(-pi / 4.0 - lla0(1) / 2.0) * sin(lla0(2) / 2.0);
qne0(3) = sin(-pi / 4.0 - lla0(1) / 2.0) * cos(lla0(2) / 2.0);
qne0(4) = cos(-pi / 4.0 - lla0(1) / 2.0) * sin(lla0(2) / 2.0);

qneo_inv = quatinv(qne0);

qdthe = quatmultiply(qneo_inv, qne)';

vec = zeros(1,3);

if qdthe(1) ~= 0
    phi_m = atan(sqrt(qdthe(2)*qdthe(2)+qdthe(3)*qdthe(3)+qdthe(4)*qdthe(4))/qdthe(1));
    f = 0.5 * sin(phi_m) / phi_m;
    vec(1) = qdthe(2) / f;
    vec(2) = qdthe(3) / f;
    vec(3) = qdthe(4) / f;
else
    vec(1:3) = pi * qdthe(2:4);
end

vec_half = 0.5 * vec;

v = 0.5 * vec_half;
mold = norm(v, 3);
sc = sin(mold) / mold;
qdthe_half(1) = cos(mold);
qdthe_half(2) = sc * v(1);
qdthe_half(3) = sc * v(2);
qdthe_half(4) = sc * v(3);
    
qne_m = quatmultiply(qne0, qdthe_half);        % communication law
lla_m = zeros(1,3);

if qne_m(1) ~= 0
    lla_m(2) = 2 * atan(qne_m(4) / qne_m(1));
    lla_m(1) = 2 * (-pi / 4.0 - atan(qne_m(3) / qne_m(1)));
elseif qne_m(1) == 0 & qne_m(3) == 0
    lla_m(2) = pi;
    lla_m(1) = 2 * (-pi / 4.0 - atan(-qne_m(2) / qne_m(4)));
elseif qne_m(1) == 0 & qne_m(4) == 0
    lla_m(2) = 2 * atan(-qne_m(2) / qne_m(3));
    lla_m(1) = pi / 2.0;
end

lla_m(3) = (lla0(3) + lla(3)) / 2.0;

% compute rn, rm
rn = WGS84_A / sqrt(1 - e * e * sin(lla_m(1)) * sin(lla_m(1)));
rm = WGS84_A * (1 - e * e) / sqrt(power(1 - e * e * sin(lla_m(1)) * sin(lla_m(1)), 3));

% compute wie_m
wie_m(1) = omega_e * cos(lla_m(1));
wie_m(2) = 0;
wie_m(3) = -omega_e * sin(lla_m(1));

% compute wen_m
wen_m(1) = vel_m(2) / (rn + lla_m(3));
wen_m(2) = -vel_m(1) / (rm + lla_m(3));
wen_m(3) = -vel_m(2) * tan(lla_m(1)) / (rn + lla_m(3));
win_m = wie_m + wen_m;

% compute g_l
gl_m = zeros(1,3);
sinB = sin(lla_m(1));
sinB2 = sinB * sinB;
sinB4 = sinB2 * sinB2;
gl_m(3) = grav(1) * (1.0 + grav(2) * sinB2 + grav(3) * sinB4) + (grav(4) + grav(5) * sinB2) * lla0(3) + grav(6) * lla0(3) * lla0(3);

% recompute zeta, xi
phi = gy+1/12*cross(gy0, gy);
half_phi = 0.5*phi;
mold = norm(half_phi);
sc = sin(mold) / mold;
qbb(1) = cos(mold);
qbb(2) = sc * half_phi(1);
qbb(3) = sc * half_phi(2);
qbb(4) = sc * half_phi(3);

zeta = win_m.*dt;
half_zeta = -0.5*zeta;
mold = norm(half_zeta);
sc = sin(mold) / mold;
qnn(1) = cos(mold);
qnn(2) = sc * half_zeta(1);
qnn(3) = sc * half_zeta(2);
qnn(4) = sc * half_zeta(3);

tmp_q = quatmultiply(qbn0, qbb);
qbn = quatmultiply(qnn, tmp_q);

% normalization
% e_q = sumsqr(qbn);
% e_q = 1 - 0.5 * (e_q - 1);
% qbn = e_q.*qbn;
qbn = quatnormalize(qbn);

% % flip the sign if two quaternions are opposite in sign
% if (qbn(1)*qbn0(1)<0 && qbn(2)*qbn0(2)<0 && qbn(3)*qbn0(3)<0 && qbn(4)*qbn0(4)<0 && norm(qbn-qbn0) > 0.05)
%     qbn = -qbn;
% end

% rpy = flip(quat2eul(qbn));
rpy = quat2eul(qbn);

% ------------------ Phi-angle model GPS/INS LC algorithm -------------- %
gpstime = gpsmsr(1);
gpsmsr(2:3) = deg2rad(gpsmsr(2:3));

% prediction
% compute rn, rm
rn = WGS84_A / sqrt(1 - e * e * sin(lla(1)) * sin(lla(1)));
rm = WGS84_A * (1 - e * e) / sqrt(power(1 - e * e * sin(lla(1)) * sin(lla(1)), 3));

c1 = 0; c2 = 1;
    
lla_in = c1*lla0 + c2*lla;
vel_in = c1*vel0 + c2*vel;

Arr = zeros(3,3);
Arr(1,3) = -vel_in(1)/(rm + lla_in(3))^2;
Arr(2,1) = vel_in(2)*sin(lla_in(1))/((rn + lla_in(3))*(cos(lla_in(1)))^2);
Arr(2,3) = -vel_in(2)/((rn + lla_in(3))^2*cos(lla_in(1)));

Arv = zeros(3,3);
Arv(1,1) = 1/(rm + lla_in(3));
Arv(2,2) = 1/((rn + lla_in(3))*cos(lla_in(1)));
Arv(3,3) = -1;

Avr(1,1) = -2*vel_in(2)*omega_e*cos(lla_in(1))-(vel_in(2))^2/((rn + lla_in(3))*(cos(lla_in(1)))^2);
Avr(1,3) = -vel_in(1)*vel_in(3)/(rm + lla_in(3))^2+(vel_in(2))^2*tan(lla_in(1))/(rn + lla_in(3))^2;
Avr(2,1) = 2*omega_e*(vel_in(1)*cos(lla_in(1))-vel_in(3)*sin(lla_in(1)))+(vel_in(2)*vel_in(1))/((rn + lla_in(3))*(cos(lla_in(1)))^2);
Avr(2,3) = (-vel_in(2)*vel_in(3)-vel_in(1)*vel_in(2)*tan(lla_in(1)))/(rn + lla_in(3))^2;
Avr(3,1) = 2*vel_in(2)*omega_e*sin(lla_in(1));
Rmn = sqrt(rm*rn);
Avr(3,3) = (vel_in(2))^2/(rn + lla_in(3))^2+(vel_in(1))^2/(rm + lla_in(3))^2-2*gl(3)/(Rmn + lla_in(3));

Aer = zeros(3,3);
Aer(1,1) = -omega_e*sin(lla_in(1));
Aer(1,3) = -vel_in(2)/(rn + lla_in(3))^2;
Aer(2,3) = vel_in(1)/(rm+lla_in(3))^2;
Aer(3,1) = -omega_e*cos(lla_in(1))-vel_in(2)/((rn + lla_in(3))*(cos(lla_in(1)))^2);
Aer(3,3) = vel_in(2)*tan(lla_in(1))/(rn + lla_in(3))^2;

Avv = zeros(3,3);
Avv(1,1) = vel_in(3)/(rm + lla_in(3));
Avv(1,2) = -2*omega_e*sin(lla_in(1))-2*vel_in(2)*tan(lla_in(1))/(rn + lla_in(3));
Avv(1,3) = vel_in(1)/(rm + lla_in(3));
Avv(2,1) = 2*omega_e*sin(lla_in(1))+vel_in(2)*tan(lla_in(1))/(rn + lla_in(3));
Avv(2,2) = (vel_in(3)+vel_in(1)*tan(lla_in(1)))/(rn + lla_in(3));
Avv(2,3) = 2*omega_e*cos(lla_in(1))+vel_in(2)/(rn + lla_in(3));
Avv(3,1) = -2*vel_in(1)/(rm+lla_in(3));
Avv(3,2) = -2*omega_e*cos(lla_in(1))-2*vel_in(2)/(rn + lla_in(3));

Aev = zeros(3,3);
Aev(1,2) = 1/(rn + lla_in(3));
Aev(2,1) = -1/(rm + lla_in(3));
Aev(3,2) = -tan(lla_in(1))/(rn + lla_in(3));

% compute cbn
cbn = quat2dcm(qbn);
cbn = cbn';

fn = cbn*(ac/dt);

Tba = 4*3600;
Tbg = 4*3600;
Tsa = 4*3600;
Tsg = 4*3600;

fn_cross = [0 -fn(3) fn(2); fn(3) 0 -fn(1); -fn(2) fn(1) 0];
win_cross = [0 -win(3) win(2); win(3) 0 -win(1); -win(2) win(1) 0];
f_diag = diag(ac/dt);
w_diag = diag(gy/dt);

A = [Arr, Arv, zeros(3,3), zeros(3,3), zeros(3,3), zeros(3,3), zeros(3,3); ...
    Avr, Avv, fn_cross, cbn, zeros(3,3), cbn*f_diag, zeros(3,3); ...
    Aer, Aev, -win_cross, zeros(3,3), -cbn, zeros(3,3), -cbn*w_diag; ...
    zeros(3,3), zeros(3,3), zeros(3,3), diag([-1.0/Tba,-1.0/Tba, -1.0/Tba]), zeros(3,3), zeros(3,3), zeros(3,3); ...
    zeros(3,3),zeros(3,3),zeros(3,3),zeros(3,3), diag([-1.0/Tbg,-1.0/Tbg, -1.0/Tbg]), zeros(3,3), zeros(3,3); ...
    zeros(3,3), zeros(3,3), zeros(3,3), zeros(3,3), zeros(3,3), diag([-1.0/Tsa,-1.0/Tsa, -1.0/Tsa]), zeros(3,3); ...
    zeros(3,3), zeros(3,3), zeros(3,3), zeros(3,3), zeros(3,3), zeros(3,3), diag([-1.0/Tsg,-1.0/Tsg, -1.0/Tsg])];

B = zeros(21,18);
B(4:6,1:3) = cbn;
B(7:9,4:6) = -cbn;
B(10:12,7:9) = eye(3);
B(13:15,10:12) = eye(3);
B(16:18,13:15) = eye(3);
B(19:21,16:18) = eye(3);

F = eye(21)+A*dt;

% set up Q
% qg = (deg2rad(0.005)/3600)^2;
% qa = (25*10^-6*gl(3))^2;
% qbg = 2*(deg2rad(0.0022)/60)^2/Tbg;
% qba = 2*(0.00075/60)^2/Tba;
% qsg = 2*(10*10^-6)^2/Tsg;
% qsa = 2*(10*10^-6)^2/Tsa;
qg = (deg2rad(0.2))^2;              % Not tuned >> Assumes 
qa = 0.05^2;
qbg = 2*(deg2rad(0.0022)/60)^2/Tbg;
qba = 2*(0.00075/60)^2/Tba;
qsg = 2*(10*10^-6)^2/Tsg;
qsa = 2*(10*10^-6)^2/Tsa;
Q = zeros(18,18);
Q(1,1) = qa; Q(2,2) = qa; Q(3,3) = qa; 
Q(4,4) = qg; Q(5,5) = qg; Q(6,6) = qg;
Q(7,7) = qba; Q(8,8) = qba; Q(9,9) = qba; 
Q(10,10) = qbg; Q(11,11) = qbg; Q(12,12) = qbg;
Q(13,13) = qsa; Q(14,14) = qsa; Q(15,15) = qsa; 
Q(16,16) = qsg; Q(17,17) = qsg; Q(18,18) = qsg;
Qk = 0.5*(F*B*Q*B'+B*Q*B'*F')*dt;
% Qk = (F*B*Q*B'*F')*dt;

% Extended Kalman Filter 
xHatM = F*xHatP;                    % Forward Euler integration (A priori)
PM = F*PP*F'+Qk;                    % cov. Estimate

rn = WGS84_A / sqrt(1 - e * e * sin(lla_in(1)) * sin(lla_in(1)));
rm = WGS84_A * (1 - e * e) / sqrt(power(1 - e * e * sin(lla_in(1)) * sin(lla_in(1)), 3));
coeff = diag([(rm+lla_in(3)); ((rn+lla_in(3))*cos(lla_in(1))); -1]);

% rn1 = WGS84_A / sqrt(1 - e * e * sin(lla_in(1)) * sin(lla_in(1)));
% rm1 = WGS84_A * (1 - e * e) / sqrt(power(1 - e * e * sin(lla_in(1)) * sin(lla_in(1)), 3));
% coeff1 = diag([(rm+lla_in(3)); ((rn+lla_in(3))*cos(lla_in(1))); -1]);

% rn = WGS84_A / sqrt(1 - e * e * sin(gpsmsr(2)) * sin(gpsmsr(2)));
% rm = WGS84_A * (1 - e * e) / sqrt(power(1 - e * e * sin(gpsmsr(2)) * sin(gpsmsr(2)), 3));
% coeff = diag([(rm+gpsmsr(4)); ((rn+gpsmsr(4))*cos(gpsmsr(2))); -1]);

% drn = rn1 - rn;
% drm = rm1 - rm;
% dcoeff = coeff1 - coeff;

% update with GPS measurements
if gpstime <= msrk1(1) && gpstime > msrk(1)  % If GPS time between two inertial measurement times, do an update %
    
    % R = 0.01*diag(gpsmsr(5:7).^2);  % R matrix -- note coefficient of 0.01!!!
    R = diag(gpsmsr(5:7).^2);  % R matrix
    
    % dr = [2.52;0.794;-0.468];  % LEVER ARM in m (XYZ body frame, YX(-Z) aligns with NED)
    % dr = [0.794;2.52;0.468];
    dr = [1e-9;1e-9;1e-9];  % LEVER ARM in m (XYZ body frame, YX(-Z) aligns with NED)
    lla_gps_corr = coeff*(gpsmsr(2:4))' - cbn*dr;

    y = coeff*lla_in' - lla_gps_corr;
    
    gpsResUpd = [gpstime, lla_gps_corr'];
    
    H = zeros(3,21);
    H(1:3, 1:3) = coeff;
    
    L = PM*H'*inv(H*PM*H'+R);
    yHat = H*xHatM;
    xHatP = xHatM + L*(y-yHat);         % a posteriori estimate
    PP = (eye(size(F))-L*H)*PM*(eye(size(F))-L*H)'+L*R*L';
    
    lla = lla - (xHatP(1:3))';
    corr = - (xHatP(1:3))';
    
    xi = xHatP(7:9);
    E = [0 -xi(3) xi(2); xi(3) 0 -xi(1); -xi(2) xi(1) 0];
    cbn = (eye(3)+E)*cbn;
    
    vel = vel - (xHatP(4:6))'; %commenting out for debug 3/7/22
    
%     comment out below to ignore correction step- not ideal but helps
    %convergence??
%   qbn = dcm2quat(cbn');
    
    % normalization
%     e_q = sumsqr(qbn);
%     e_q = 1 - 0.5 * (e_q - 1);
%     qbn = e_q.*qbn;
    qbn = quatnormalize(qbn);
    
    % flip the sign if two quaternions are opposite in sign
    if (norm((qbn+qbn0)/2) < 0.15 && norm((qbn-qbn0)/2) > 0.85)
        qbn = -qbn;
    end
    
    % rpy = flip(quat2eul(qbn));
    rpy = quat2eul(qbn);
    
    dv = vel - vel0;
    
    gpsUpdated = 1;
    gpsCnt = gpsCnt + 1;
else
    xHatP = xHatM;
    PP = PM;
    
    gpsResUpd = [];
    gpsUpdated = 0;
    corr = [0 0 0];
end

    ned = (coeff*lla')';

end

function PP = PPinitialize
% Initialize state-error array
    PP = zeros(21,21);              
    PP(1,1) = (deg2rad(1))^2;
    PP(2,2) = (deg2rad(1))^2;
    PP(3,3) = 1^2;
    PP(4,4) = 0.002^2;
    PP(5,5) = 0.002^2;
    PP(6,6) = 0.002^2;
    PP(7,7) = (deg2rad(0.01))^2;
    PP(8,8) = (deg2rad(0.01))^2;
    PP(9,9) = (deg2rad(0.01))^2;
    PP(10,10) = (deg2rad(0.01)/3600)^2;
    PP(11,11) = (deg2rad(0.01)/3600)^2;
    PP(12,12) = (deg2rad(0.01)/3600)^2;
    PP(13,13) = (10^-4)^2;
    PP(14,14) = (10^-4)^2;
    PP(15,15) = (10^-4)^2;
    PP(16,16) = (10^-5)^2;
    PP(17,17) = (10^-5)^2;
    PP(18,18) = (10^-5)^2;
    PP(19,19) = (10^-5)^2;
    PP(20,20) = (10^-5)^2;
    PP(21,21) = (10^-5)^2;
end