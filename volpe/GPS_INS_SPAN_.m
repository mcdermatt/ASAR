%GPS_INS_SPAN Loosely Coupled - version 11/22/2021

%TODO: Tune INS sensor noise covariance per datasheet
%TODO: get rid of all *_combined vars, instead overwrite INS data
%TODO: remap inputs and outputs of main func into data structure
%TODO: debug timestep issues between subsequent lidar scans (!= 1/200s)
%TODO: finish implementing WLS

clear all

% User defined paramaters
addpath('./Data/SPAN');         % Path for data 
dat = load('signageData.mat');  % Data file
endIndx = 2e4; %was 1e6                  % Example 1e4 is 50 sec of data @ 200 Hz % If endIndex exceeds max, then reset to max
fuse_lidar = 1;                 % combine xHat_ins with xHat_lidar at each GPS measurement                         
fuse_gps = 0;                   % if 0, just INS and Lidar are fused at GPS measurement timestamps

% Clean workspace
close all
clc
beep off 
% feature('SetPrecision', 64)

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
rpyRef = deg2rad([dat.inspvax.insroll, dat.inspvax.inspitch, dat.inspvax.insazim]);
qbnRef = eul2quat(flip(rpyRef')');   
qbnTime = dat.inspvax.sow1465;


%import lidar data and transform to relative estimates in lla frame
%--------------------------------------------------------------------------
pos_lidar_enu = -load('pos_lidar_enu.mat').pos_lidar_enu;
% position: [2851×7 double]:
    % Columns 1-3: Translation vector (x,y,z) (in meters)
    % Columns 4-7: Rotational vector (qw,qx,qy,qz) (in rad)
 
[lat_lidar, lon_lidar, hgt_lidar] = enu2geodetic(pos_lidar_enu(:,1), pos_lidar_enu(:,2), ... 
        pos_lidar_enu(:,3), ppplat(1), ppplon(1), ppphgt(1), wgs84Ellipsoid('m'));

t_lidar = load('lidar_time.mat').t;
t_lidar = t_lidar - t_lidar(1); %make this start at 0

%convert lidar lat lon hgt to relative estimates
dlat_lidar = lat_lidar - [lat_lidar(2:end).' 0].';
dlat_lidar = -dlat_lidar(1:end-1);
dlon_lidar = lon_lidar - [lon_lidar(2:end).' 0].';
dlon_lidar = -dlon_lidar(1:end-1);
%bug here?? going to replace with OG hgt from pos_lidar_enu
% dhgt_lidar = hgt_lidar - [hgt_lidar(2:end).' 0].';
dhgt_lidar = pos_lidar_enu(:,3) - [pos_lidar_enu(2:end, 3).' 0].'  ;
dhgt_lidar = dhgt_lidar(1:end-1); %TODO- still needs arbitrary scaling?

%remove first time_step to keep consistant with dlat, dlon, dhgh_lidar
t_lidar = t_lidar(2:end);

%starting with just position estimates- will move on to rotations
%eventaully...
dpos_lidar = [(t_lidar + ppptime(1)) dlat_lidar dlon_lidar dhgt_lidar];

%DEBUG 1/3/22 : keep dpos_lidar in NED
dE = pos_lidar_enu(:,2) - [pos_lidar_enu(2:end,2).' 0].';
dN = pos_lidar_enu(:,1) - [pos_lidar_enu(2:end,1).' 0].';
dU = pos_lidar_enu(:,3) - [pos_lidar_enu(2:end,3).' 0].';
dpos_lidar = [(t_lidar + ppptime(1)), -dN(1:end-1), -dE(1:end-1), -dU(1:end-1)];

%debug: get change in velocity from lidar
ddE = dE(:) - [dE(2:end).' 0].';
ddN = dN(:) - [dN(2:end).' 0].';
ddU = dU(:) - [dU(2:end).' 0].';
dvel_lidar = [ddN(1:end-1), ddE(1:end-1), ddU(1:end-1)];
%--------------------------------------------------------------------------

% sync and interpolate GPS readings so they occur at the same time as Lidar
% measurements 
%NOTE: we need to do this because the lidar did not begin recording at the 
%same time as the GPS/INS system. Also the lidar unit records at
%inconsistant frequencies: look at a plot of t_lidar vs time
%--------------------------------------------------------------------------
startGPS = 1109; %cut off every GPS point before here to sync datasets
gps_time = ppptime - ppptime(1);
gps_time = gps_time(startGPS:end)-startGPS;

%create new time vector for GPS that:
%   1) Samples at coinciding with every 10th lidar point
%   2) Interpolates in regions with Lidar data outages
t_gps_new = t_lidar(1:10:end);
t_gps_new = [t_gps_new(1:112); linspace(139.5, 185.5, 47).'; t_gps_new(113:122); ...
    linspace(196.3,244.3,48).'; t_gps_new(124:138); linspace(259.4, 303.4, 44).'; ...
    t_gps_new(140:151); linspace(316.9, 363.9,47).';  t_gps_new(152:161);...
    linspace(375.2, 429.2, 54).'; t_gps_new(163:end) ];

%for debug
% t_gps_new = t_lidar;

lon_gps_lidartime = interp1(gps_time, ppplon(startGPS:end), t_gps_new);
lat_gps_lidartime = interp1(gps_time, ppplat(startGPS:end), t_gps_new);
hgt_gps_lidartime = interp1(gps_time, ppphgt(startGPS:end), t_gps_new);

%interpolate GPS stds
lon_gps_lidartime_std = interp1(gps_time, ppplonstd(startGPS:end), t_gps_new);
lat_gps_lidartime_std = interp1(gps_time, ppplatstd(startGPS:end), t_gps_new);
hgt_gps_lidartime_std = interp1(gps_time, ppphgtstd(startGPS:end), t_gps_new);

%set all NaN lat/lon/hgt values at beginning of vectors to first non-nan
%value
nan_idx = find(isnan(lon_gps_lidartime)); %pts with nan vals

lon_gps_lidartime(nan_idx) = lon_gps_lidartime(nan_idx(end) + 1);
lat_gps_lidartime(nan_idx) = lat_gps_lidartime(nan_idx(end) + 1);
hgt_gps_lidartime(nan_idx) = hgt_gps_lidartime(nan_idx(end) + 1);

lon_gps_lidartime_std(nan_idx) = lon_gps_lidartime_std(nan_idx(end) + 1);
lat_gps_lidartime_std(nan_idx) = lat_gps_lidartime_std(nan_idx(end) + 1);
hgt_gps_lidartime_std(nan_idx) = hgt_gps_lidartime_std(nan_idx(end) + 1);

%add time offset back in to t_lidar so that IMU data can be truncated
%properly
% t_lidar = t_lidar + ppptime(1);
t_gps_new = t_gps_new + ppptime(1);

% %was this
% gpspos = [ppptime ppplat ppplon ppphgt ppplatstd ppplonstd ppphgtstd];
%now this
gpspos = [t_gps_new lat_gps_lidartime lon_gps_lidartime hgt_gps_lidartime ...
    lat_gps_lidartime_std lon_gps_lidartime_std hgt_gps_lidartime_std];
%--------------------------------------------------------------------------

% Apply transducer constants for IMU
ax = double(rawimusx.rawimuxaccel)*(0.400/65536)*(9.80665/1000)/200;    % m/s
ay = double(rawimusx.rawimuyaccel)*(0.400/65536)*(9.80665/1000)/200;    % m/s
az = double(rawimusx.rawimuzaccel)*(0.400/65536)*(9.80665/1000)/200;    % m/s
gx = double(rawimusx.rawimuxgyro)*0.0151515/65536/200;                  % deg
gy = double(rawimusx.rawimuygyro)*0.0151515/65536/200;                  % deg
gz = double(rawimusx.rawimuzgyro)*0.0151515/65536/200;                  % deg

% Pacakge IMU data
imutime = rawimusx.rawimugnsssow;
imuraw = [imutime deg2rad(gy) deg2rad(gx) -deg2rad(gz) ay ax -az];  % imuraw = [imutime gx gy gz ax ay az];
imuIndx = imutime >= gpspos(1,1)-0.05;
% trim data > start at first GPS sample
imuraw = imuraw(imuIndx,:);          %was this (incorrect??)
imutime = imutime(imuIndx);
%test -> trying to sync start of INS
% imuraw = imuraw(1109*200:end, :);       
% imutime = imutime(1109*200:end);

% Prevent analysis from running past end of data record
% maxGPStime = ppptime(end-1); %was this
maxGPStime = t_gps_new(end - 1);
maxIMUtime = imutime(end);
maxTime = min(maxGPStime,maxIMUtime);
maxIMUindx = max(find(imutime<=maxTime));
if endIndx > maxIMUindx     % Truncate IMU record if necessary
    endIndx = maxIMUindx;  
end
% Compute length of GPS record, whether or not IMU record is truncated
imuEndTime = imutime(endIndx);
gpsEndIndx = max(find(ppptime<=imuEndTime));

%Initialization of Arrays
xHatM_ins = double(zeros(21,1));            % state estimate
xHatM_lidar = double(zeros(3,1));
xHatM_ins_hist = double(zeros(gpsEndIndx, 21)); %cumulative vars for summing between GPS keyframes
xHatM_lidar_hist = double(zeros(gpsEndIndx, 3));
xHatM_combined_hist = double(zeros(gpsEndIndx, 21));
ba = zeros(3,1);                % bias accels
bg = zeros(3,1);                % bias gyros
sa = zeros(3,1);                % scale factor accels
sg = zeros(3,1);                % scale factor gyros
PP = PPinitialize;              % state-error matrix
PP = PP*1e6;                    % Set initial covariance matrix to be large (indicating initial uncertainty)
Ncol = 22;
msrArr = zeros(7,endIndx-2);        % INS measurement array
resArr = zeros(endIndx-2,Ncol-1);   % Storage array of state-perturbation error results
qbnArr = zeros(endIndx-2,4);        % Storage of quaternion results (relating Body to Inertial)
gpsMsrArr = zeros(4,gpsEndIndx);    % GPS measurement
gps_res = zeros(gpsEndIndx,4);

%INITIALIZATION of EKF
lla0_ins = [deg2rad(ppplat(1)), deg2rad(ppplon(1)), ppphgt(1)];   % from gps.txt
[minVal, ind] = min(abs(dat.inspvax.sow1465 - imuraw(1,1)));
% vel0 = [dat.inspvax.insnorthvel(ind), dat.inspvax.inseastvel(ind), -dat.inspvax.insupvel(ind)]; 
vel0 = [0, 0, 0];
rpy0 = deg2rad([dat.inspvax.insroll(ind), dat.inspvax.inspitch(ind), dat.inspvax.insazim(ind)]);
% rpy0 = deg2rad([1.367381982628118, 2.131468301661621, 184.2637214777763]);
dv0 = [0 0 0];
qbn0 = eul2quat(flip(rpy0));     %Initialize with RPY solutoin
bias = 0;

% Seed initial measurements
msrk = imuraw(1,:);
msrk = msrk';
gpsmsr = gpspos(1,:);
lidarmsr = dpos_lidar(1, :);
ned0_lidar = [0, 0, 0]; %lla0_ins;
lla0_combined = lla0_ins;
lla_ins_last = lla0_ins; %lla_in the last time GPS was called
lla_combined_hist = zeros(3, gpsEndIndx); %array to hold lla_combined data
t_last_lidar = msrk(1);
xHatM_ins_cum = zeros(21,1);
xHatM_lidar_cum = zeros(3,1);

%init matrices for storing PM info
PM_hist_ins = zeros(endIndx,3);
PM_hist_lidar = zeros(endIndx, 3);
PM_hist_combined = zeros(endIndx, 3);
PM_lidar_last = 0*eye(3); %init uncertainty for lidar

%PP initialize does not work correctly if fuse_lidar == 0
if fuse_gps == 0
    PP(1:3,1:3) = eye(3)*1e-17;
end

%initialize var containing product of Fs
F200 = zeros(21,21);

% Start Loop
gpsCnt = 0;
lidarCnt = 0;
i = 2;
while i < endIndx  
    
    %TODO: bring in lidar rotations to correct INS data???
    qbn_lidar = pos_lidar_enu(lidarCnt+1, 4:7);
    
    if(mod(i,1e3)==2); i-2, end  % echo time step to screen current step

    %old (doesn't timeshift INS data)
%     msrk1 = imuraw(i,:);
%     msrk1 = msrk1';

    %timeshifted to match GPS/Lidar
    %can't just cut off the first 1109*200 elements since we need the
    %timestamps from imuraw intact
%     221745 % from manual inspection of LX code...
    msrk1 = [imuraw(i), imuraw(221800 + i, 2:end)].';
    
    if isempty(msrk1) 
        break;
    end
    if isempty(gpsmsr)
        break;
    end  
       
    dt = msrk1(1) - msrk(1);
    msrk1(7) =  msrk1(7) - bias*dt; 
        
    % GPS/INS solution
    [lla_ins, ned_lidar, lla_combined, vel, rpy, dv, qbn, xHatM_ins, xHatM_lidar, xHatM_combined, PP, gpsUpdated, lidarUpdated, gpsCnt, lidarCnt, gpsResUpd, PM_ins, PM_lidar, PM_combined, F200] = ...
        ins_gps(lla0_ins, ned0_lidar, lla0_combined, lla_ins_last, vel0, rpy0, dv0, qbn0, qbn_lidar, msrk, msrk1, gpsmsr, lidarmsr, xHatM_ins, xHatM_lidar, xHatM_ins_cum, xHatM_lidar_cum, PP, gpsCnt, lidarCnt, t_last_lidar, PM_lidar_last, fuse_lidar, fuse_gps, F200);

%     vel %TODO why is this exploding???

%     xHatM_ins(1) %stuck at 0
    
    %save sigmas in units of (m)
    PM_hist_ins(i-1,:) = [sqrt(PM_ins(1,1))*6.36e6 sqrt(PM_ins(2,2))*4.97e6, sqrt(PM_ins(3,3))];
    PM_hist_lidar(i-1,:) = [sqrt(PM_lidar(1,1))*6.36e6 sqrt(PM_lidar(2,2))*4.97e6, sqrt(PM_lidar(3,3))];
    PM_hist_combined(i-1,:) = [sqrt(PM_combined(1,1))*6.36e6 sqrt(PM_combined(2,2))*4.97e6, sqrt(PM_combined(3,3))];

    if lidarUpdated == 1
        t_last_lidar = lidarmsr(1);
        lidarmsr = dpos_lidar(lidarCnt+1, :);%update lidar measurement
        ned0_lidar = ned_lidar;
        vel = dpos_lidar(lidarCnt, 2:4)*10; %-> BIG improvement
        if gpsUpdated == 1
            xHatM_lidar_hist(gpsCnt,:) = xHatM_lidar_hist(gpsCnt,:) + xHatM_lidar.';
            lla_ins_last = lla_ins;
        else
            xHatM_lidar_hist(gpsCnt+1,:) = xHatM_lidar_hist(gpsCnt+1,:) + xHatM_lidar.';
        end
    end
    
    if gpsUpdated == 1
        xHatM_ins_hist(gpsCnt,:) = xHatM_ins_hist(gpsCnt,:) + xHatM_ins.';
        gpsmsr = gpspos(gpsCnt+1, :);   
        gps_res(gpsCnt,:) = gpsResUpd;
        xHatM_ins(1:9) = zeros(9,1);        % Reset error states %was xHatP_ins
        xHatM_combined_hist(gpsCnt,:) = xHatM_combined;
        lla_combined_hist(:,gpsCnt) = lla_combined;
        lla0_combined = lla_combined;
    else
        xHatM_ins_hist(gpsCnt+1,:) = xHatM_ins_hist(gpsCnt+1,:) + xHatM_ins.';
    end

    % Store Data
    resArr_ins(i-1,:) = [lla_ins vel rpy rad2deg(bg')*3600 (ba')*10^6/9.7803267715 (sg')*10^-6 (sa')*10^-6];
    resArr_lidar(i-1,:) = [ned_lidar vel rpy rad2deg(bg')*3600 (ba')*10^6/9.7803267715 (sg')*10^-6 (sa')*10^-6];
    qbnArr(i-1,:) = qbn;
    if gpsUpdated == 1 && ~isempty(gpsmsr)
        gpsMsrArr(:,gpsCnt) = gpsmsr(1:4)';
    end
    
%     size(PM_lidar)
    PM_lidar_last = PM_lidar;
    
    %for debug
%     lla_ins(1) - lla0_ins(1)
%     (1) %<- currently exploding...

    msrArr(:,i-1) = msrk1;
    lla0_ins = lla_ins; 
    vel0 = vel; 
    rpy0 = rpy;
    dv0 = dv; 
    qbn0 = qbn;
    msrk = msrk1;
    
    xHatM_ins_cum = xHatM_ins_hist(gpsCnt+1,:);
    xHatM_lidar_cum = xHatM_lidar_hist(gpsCnt+1,:);
    
    %test
    PP = PM_ins;
    
    i = i+1;  
end 


% (((more plots in Liangchun's file)))

time = msrArr(1,:);
startTime = time(1);

%convert GPS measurements array to ENU
[Ngps, Egps, Ugps] = geodetic2enu(gpsMsrArr(2,:), gpsMsrArr(3,:), gpsMsrArr(4,:), ...
    gpsMsrArr(2,1), gpsMsrArr(3,1), gpsMsrArr(4,1), wgs84Ellipsoid);
%find timesteps where gpsMsrArr != 0
nonzero = find( gpsMsrArr(1,:) ~= 0); %nonzero elements of GPS
nonzero_combined = find(lla_combined_hist(1,:) ~= 0); %nonzero elements of INS+Lidar fused estiamtes (occur on GPS timesteps)

[Eins, Nins, Uins] = geodetic2enu(resArr_ins(:,1), resArr_ins(:,2), resArr_ins(:,3), ...
    resArr_ins(1,1), resArr_ins(1,2), resArr_ins(1,3), wgs84Ellipsoid, 'radians');

%plot summed xHats between GPS frames for INS and Lidar
figure()
title('cumulative xHat')
subplot(3,1,1)
hold on
xlabel('timestep')
%TODO: make sure this is still lon/ lat after ENU->NED swap
ylabel('estimated change in lon per GPS frame (m)')
plot(xHatM_ins_hist(:,2))
plot(xHatM_lidar_hist(:,2))
plot(xHatM_combined_hist(:,1))
dEgps = -Egps(nonzero);
dEgps = dEgps - [dEgps(2:end) 0];
plot([0 dEgps(1:end-1)])
legend('xHat- INS', 'xHat- Lidar', 'xHat- combined', 'gps baseline')

subplot(3,1,2)
hold on
xlabel('timestep')
ylabel('estimated change in lat per GPS frame (m)')
plot(xHatM_ins_hist(:,1))
plot(xHatM_lidar_hist(:,1))
plot(xHatM_combined_hist(:,2))
dNgps = -Ngps(nonzero);
dNgps = dNgps - [dNgps(2:end) 0];
plot([0 dNgps(1:end-1)])
legend('xHat- INS', 'xHat- Lidar', 'xHat- combined', 'gps baseline')


%covaraince plots (new)
figure(5)
title('GPS Corrected Lidar/ INS')
% sgtitle('Lidar corrected INS')
% sgtitle('Lidar and INS, no correction')
% hold on %putting this here prevents logy axis
skip_start = 2500;
semilogy(time(skip_start:end) - startTime, PM_hist_ins(skip_start:end-2,1), ...
    time(skip_start:end) - startTime, PM_hist_lidar(skip_start:end-2,1), ...
    time(skip_start:end) - startTime, PM_hist_combined(skip_start:end-2,1))
xlim([50, 60])
grid on
legend('\sigma_{lon,ins}', '\sigma_{lon,lidar}', '\sigma_{lon,combined}')
xlabel('time (s)')
ylabel('standard deviation of error (m)')

figure(6)
% Postion Plots ----------------------
subplot(3,1,1);
hold on; grid on
% plot(time-startTime, rad2deg(resArr_ins(:,1))); %was this with ins in lla
plot(time-startTime, Eins)
% plot(time-startTime, rad2deg(resArr_lidar(:,1))); %was this with lidar in lla
plot(time-startTime, resArr_lidar(:,2));
% plot(time-startTime, rad2deg(lla_combined_hist(1,1:end-1))); %not working

% plot(gpsMsrArr(1, nonzero)-startTime, gpsMsrArr(2, nonzero)); %used for lat/lon
plot(gpsMsrArr(1, nonzero)-startTime, Egps(nonzero));
% plot(gpsMsrArr(1,nonzero_combined)-startTime, rad2deg(lla_combined_hist(1,nonzero_combined)))
plot(gpsMsrArr(1,nonzero_combined)-startTime, lla_combined_hist(1,nonzero_combined))
legend('LLA INS', 'LLA Lidar Dead Reckoning', 'pure GPS', 'Lidar and INS only')
% legend('LLA INS', 'LLA Lidar Dead Reckoning', 'pure GPS') %temp
hold on; grid on
ylabel('East (m)');


subplot(3,1,2);
hold on; grid on
% plot(time-startTime, rad2deg(resArr_ins(:,2)));
plot(time-startTime, Nins)
% plot(time-startTime, rad2deg(resArr_lidar(:,2)));
plot(time-startTime, resArr_lidar(:,1));
ylabel('longitude (deg)');
% plot(gpsMsrArr(1, nonzero)-startTime, gpsMsrArr(3, nonzero));
plot(gpsMsrArr(1, nonzero)-startTime, Ngps(nonzero));
% plot(gpsMsrArr(1,nonzero_combined)-startTime, rad2deg(lla_combined_hist(2,nonzero_combined)))
plot(gpsMsrArr(1,nonzero_combined)-startTime, lla_combined_hist(2,nonzero_combined))
legend('LLA INS', 'LLA Lidar Dead Reckoning', 'pure GPS', 'Lidar and INS only')
% legend('LLA INS', 'LLA Lidar Dead Reckoning', 'pure GPS')
hold on; grid on
ylabel('North (m)');

subplot(3,1,3);
hold on; grid on
plot(time-startTime, resArr_ins(:,3));
plot(time-startTime, resArr_lidar(:,3));
%     plot(gpsMsrArr(1, :)-startTime, gpsMsrArr(4, :));
plot(gpsMsrArr(1, nonzero)-startTime, gpsMsrArr(4, nonzero));  
legend('LLA INS', 'LLA Lidar Dead Reckoning', 'pure GPS')
hold on; grid on
ylabel('altitude (m)');
xlabel('Time (s)');


len = length(time);
ylabel('altitude (m)');
xlabel('Time (s)');

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

%Quaternion plot
% figure(7)
% subplot(4,1,1);
% plot(imutime(3:endIndx)-startTime,qbnArr(:,1));
% hold on; grid on
% plot(qbnTime(1:gpsEndIndx)-startTime,qbnRef(1:gpsEndIndx,4));
% ylabel('{q_1}');
% subplot(4,1,2);
% plot(imutime(3:endIndx)-startTime,qbnArr(:,2));
% hold on; grid on
% plot(qbnTime(1:gpsEndIndx)-startTime,qbnRef(1:gpsEndIndx,2));
% ylabel('{q_2}');
% subplot(4,1,3);
% plot(imutime(3:endIndx)-startTime,qbnArr(:,3));
% hold on; grid on
% plot(qbnTime(1:gpsEndIndx)-startTime,qbnRef(1:gpsEndIndx,1));
% ylabel('{q_3}');
% subplot(4,1,4);
% plot(imutime(3:endIndx)-startTime,qbnArr(:,4));
% hold on; grid on
% plot(qbnTime(1:gpsEndIndx)-startTime,-qbnRef(1:gpsEndIndx,3));
% ylabel('{q_4}');
% legend('EKF','Ref')

% compute rn, rm
WGS84_A = 6378137.0;           % earth semi-major axis (WGS84) (m) 
WGS84_B = 6356752.3142;        % earth semi-minor axis (WGS84) (m) 
e = sqrt(WGS84_A * WGS84_A - WGS84_B * WGS84_B) / WGS84_A;
rn_list = WGS84_A ./ sqrt(1 - e * e * sin(resArr(:,1)).^2);
rm_list = WGS84_A * (1 - e * e) ./ sqrt(power(1 - e * e * sin(resArr(:,1)).^2, 3));

function [lla_ins, ned_lidar, lla_combined, vel, rpy, dv, qbn, xHatM_ins, xHatM_lidar, xHatM_combined, PP, gpsUpdated, lidarUpdated, gpsCnt, lidarCnt, gpsResUpd, PM_ins, PM_lidar, PM_combined, F200] = ...
    ins_gps(lla0_ins, ned0_lidar, lla0_combined, lla_ins_last, vel0, rpy0, dv0, qbn0, qbn_lidar, msrk, msrk1, gpsmsr, lidarmsr, xHatM_ins, xHatM_lidar, xHatM_ins_cum, xHatM_lidar_cum, PP, gpsCnt, lidarCnt, t_last_lidar, PM_lidar_last, fuse_lidar, fuse_gps, F200)

WGS84_A = 6378137.0;           % earth semi-major axis (WGS84) (m) 
WGS84_B = 6356752.3142;        % earth semi-minor axis (WGS84) (m) 
e = sqrt(WGS84_A * WGS84_A - WGS84_B * WGS84_B) / WGS84_A; %0.0818

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
qne(1) = cos(-pi / 4.0 - lla0_ins(1) / 2.0) * cos(lla0_ins(2) / 2.0);
qne(2) = -sin(-pi / 4.0 - lla0_ins(1) / 2.0) * sin(lla0_ins(2) / 2.0);
qne(3) = sin(-pi / 4.0 - lla0_ins(1) / 2.0) * cos(lla0_ins(2) / 2.0);
qne(4) = cos(-pi / 4.0 - lla0_ins(1) / 2.0) * sin(lla0_ins(2) / 2.0);

qee_h = zeros(1,4); qnn_l = zeros(1,4);

% compute rn, rm
rn = WGS84_A / sqrt(1 - e * e * sin(lla0_ins(1)) * sin(lla0_ins(1)));
rm = WGS84_A * (1 - e * e) / sqrt(power(1 - e * e * sin(lla0_ins(1)) * sin(lla0_ins(1)), 3));

% compute wie
wie(1) = omega_e * cos(lla0_ins(1));
wie(2) = 0;
wie(3) = -omega_e * sin(lla0_ins(1));

% compute wen
wen(1) = vel0(2) / (rn + lla0_ins(3));
wen(2) = -vel0(1) / (rm + lla0_ins(3));
wen(3) = -vel0(2) * tan(lla0_ins(1)) / (rn + lla0_ins(3));
win = wie + wen;

% compute g_l
gl = zeros(1,3);

grav = [9.7803267715, 0.0052790414, 0.0000232718, -0.000003087691089, 0.000000004397731, 0.000000000000721];
sinB = sin(lla0_ins(1));
sinB2 = sinB * sinB;
sinB4 = sinB2 * sinB2;
gl(3) = grav(1) * (1.0 + grav(2) * sinB2 + grav(3) * sinB4) + (grav(4) + grav(5) * sinB2) * lla0_ins(3) + grav(6) * lla0_ins(3) * lla0_ins(3);

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
    
lla_m(3) = lla0_ins(3) - (vel0(3) * dt) / 2.0;

% extrapolate the speed
vel_m = vel0 + 0.5 * dv0;

% compute the wie_m, wen_m - ANGULAR VELOCITIES
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
gl_m(3) = grav(1) * (1.0 + grav(2) * sinB2 + grav(3) * sinB4) + (grav(4) + grav(5) * sinB2) * lla0_ins(3) + grav(6) * lla0_ins(3) * lla0_ins(3);

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
dvfn_k = (cbn0'*cnn)*dvfb_k; %was this in Liangchun's code
% dvfn_k = (cbn0'*cnn)*dvfb_k .* dt; % I think Liangchun forgot to multiply by dt here???
dvfn_k = dvfn_k';
dvgn_k = (gl_m-cross((2*wie_m+wen_m), vel_m)).* dt;

% update velocity
dv = dvfn_k + dvgn_k;
vel = vel0 + dv;

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
gl_m(3) = grav(1) * (1.0 + grav(2) * sinB2 + grav(3) * sinB4) + (grav(4) + grav(5) * sinB2) * lla0_ins(3) + grav(6) * lla0_ins(3) * lla0_ins(3);

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

lla_ins = zeros(1,3);

if qne(1) ~= 0
    lla_ins(2) = 2 * atan(qne(4) / qne(1));
    lla_ins(1) = 2 * (-pi / 4.0 - atan(qne(3) / qne(1)));
elseif qne(1) == 0 & qne(3) == 0
    lla_ins(2) = pi;
    lla_ins(1) = 2 * (-pi / 4.0 - atan(-qne(2) / qne(4)));
elseif qne(1) == 0 & qne(4) == 0
    lla_ins(2) = 2 * atan(-qne(2) / qne(3));
    lla_ins(1) = pi / 2.0;
end

lla_ins(3) = lla0_ins(3) - vel_m(3) * dt;

% update the attitude
qdthe_half = zeros(1,4);
qne0 = zeros(1,4);

% compute qne0
qne0(1) = cos(-pi / 4.0 - lla0_ins(1) / 2.0) * cos(lla0_ins(2) / 2.0);
qne0(2) = -sin(-pi / 4.0 - lla0_ins(1) / 2.0) * sin(lla0_ins(2) / 2.0);
qne0(3) = sin(-pi / 4.0 - lla0_ins(1) / 2.0) * cos(lla0_ins(2) / 2.0);
qne0(4) = cos(-pi / 4.0 - lla0_ins(1) / 2.0) * sin(lla0_ins(2) / 2.0);

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

lla_m(3) = (lla0_ins(3) + lla_ins(3)) / 2.0;

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
gl_m(3) = grav(1) * (1.0 + grav(2) * sinB2 + grav(3) * sinB4) + (grav(4) + grav(5) * sinB2) * lla0_ins(3) + grav(6) * lla0_ins(3) * lla0_ins(3);

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

rpy = flip(quat2eul(qbn));

% ------------------ Phi-angle model GPS/INS LC algorithm -------------- %
gpstime = gpsmsr(1);
gpsmsr(2:3) = deg2rad(gpsmsr(2:3));

% prediction
% compute rn, rm
rn = WGS84_A / sqrt(1 - e * e * sin(lla_ins(1)) * sin(lla_ins(1)));
rm = WGS84_A * (1 - e * e) / sqrt(power(1 - e * e * sin(lla_ins(1)) * sin(lla_ins(1)), 3));

c1 = 0; c2 = 1;
    
lla_in = c1*lla0_ins + c2*lla_ins;
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

%I think this has to do with "Correlation time"?
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

A = double(A);

B = zeros(21,18);
B(4:6,1:3) = cbn;
B(7:9,4:6) = -cbn;
B(10:12,7:9) = eye(3);
B(13:15,10:12) = eye(3);
B(16:18,13:15) = eye(3);
B(19:21,16:18) = eye(3);

F = eye(21)+A*dt;

% set up Q
qg = (deg2rad(0.2))^2;   %initial bias error         
% qg = (deg2rad(0.5))^2; 
qa = 0.05^2;             %non-orthogonality  - not provided for new PwrPak7
% qa = 0.06^2;

%Gyro white noise = 0.0022 deg/sqrt(h) per Liangchun's pdf
% qbg = 2*(deg2rad(0.0022)/60)^2/Tbg; %with old unit
qbg = 2*(deg2rad(0.06)/60)^2/Tbg; %ours is 0.06
% Liangchun's accel White noise = 0.00075 (m/s/sqrt(hr))
% qba = 2*(0.00075/60)^2/Tba; %with old unit
qba = 2*(0.025/60)^2/Tba; %again, much higher on the new unit

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

% Extended Kalman Filter --------
% was this (only uses INS stuff)
% xHatM_ins(1:3) = lla_ins - lla0_ins; %old
% xHatM_ins(4:6) = vel - vel0;         %old       
xHatM_ins = F*xHatM_ins;          % Forward Euler integration (A priori)
PM = F*PP*F'+Qk;                    % cov. Estimate
F200 = F*F200; %keep product of sequential F matrices for use in Lidar fusion

%transform xHatM_ins into ENU here before combining with lidar...
%DEBUG -> why is xHatM_ins(1:3) exploding???
[xHatM_ins_E, xHatM_ins_N, xHatM_ins_U] = geodetic2enu(lla_ins(1), lla_ins(2), lla_ins(3), lla0_ins(1), lla0_ins(2), lla0_ins(3), wgs84Ellipsoid, 'radians');
xHatM_ins(1:3) = [xHatM_ins_N, xHatM_ins_E, xHatM_ins_U];

%set noise covariance matrix for lidar
%     Qk_lidar = 1e4*ones(3,3); %init arbitrarily high values
Qk_lidar = zeros(3,3); %test w/ zeros
%set useful parts of Qk_lidar based on values determined experimentally over the summer
Qk_lidar(3,3) = 0.0001;%sigmaz**2 (meters)
Qk_lidar(1,1) = 1.4e-17; %sigmalon**2 (deg)
Qk_lidar(2,2) = 1.4e-17; %sigmalat**2 (deg)

% from Final_Slides.pptx
%TODO: these are currently in VEHICLE BODY FRAME
% Qk_lidar(1,1) = 0.000784; %(m) 
% Qk_lidar(2,2) = 0.000484; %(m)
% Qk_lidar(3,3) = 0.005625; %(m)

lidartime = lidarmsr(1);
if lidartime <= msrk1(1) && lidartime > msrk(1)
    %only using lidar translation estimates:
       %[dxyz dvxyz quats ...] -> 21x1 vec    
   
    %need to scale by amount of time elapsed since last measurement since
    %it is different than INS measurements
    dt_lidar = lidartime - t_last_lidar;
%     denu_lidar_scaled = lidarmsr(2:end)/dt_lidar*dt;
%     denu_lidar_scaled = denu_lidar_scaled/2.54; %DEBUG- not sure why I need to scale in -> cm ...
%     xHatM_lidar = [denu_lidar_scaled].'; %was this -just looking at changes in pos (no delta_rotation)

    xHatM_lidar = lidarmsr(2:4).'; %better
       
    %PM = F*PP*F' + Q
    PM_lidar = PM_lidar_last + Qk_lidar;

    lidarUpdated = 1;
    lidarCnt = lidarCnt + 1;
    
    ned_lidar = ned0_lidar + (xHatM_lidar(1:3).');
    
    %reset ins heading to lidar estimated heading
%     qbn = qbn_lidar; %wow this is helping a lot!!!
    
    else
        lidarUpdated = 0;
        ned_lidar = ned0_lidar;
        PM_lidar = PM_lidar_last; %TODO - figure out a cleaner way to do this
    end

PM_ins = PM;
%e == 0.0818
rn = WGS84_A / sqrt(1 - e * e * sin(lla_in(1)) * sin(lla_in(1)));
rm = WGS84_A * (1 - e * e) / sqrt(power(1 - e * e * sin(lla_in(1)) * sin(lla_in(1)), 3));
coeff = diag([(rm+lla_in(3)); ((rn+lla_in(3))*cos(lla_in(1))); -1]);

% update with GPS measurements
if gpstime <= msrk1(1) && gpstime > msrk(1)  % If GPS time between two inertial measurement times, do an update %

    if fuse_lidar == 1
        
%         dt_lidar = gpstime - t_lidar;
        
        %new ------------------------------------------------------

        %  NOTE: all Ps are (+) here but that doesn't matter for now without correction
        
        % W = ([P_k         , F_big*P_{k-1};
        %       P_{k-1}*F_big^T , P_{k-1} + Qk_lidar ])^-1
        % P_k = P_bar + P_prime
        % P_bar = ??
        
        %PP currently holds the value of P_{k-1} 
        %       INCORRECT (since pp is only 0.005s behind (instead of 1 GPS
        %       timestep back ~1s))
        %TODO - create new var with different name to store this PP_last
        %TODO - do I need Qk_lidar_cum??
        
        W = pinv([PM, F200*PP;
                  PP*(F200.'), PP + [Qk_lidar, zeros(3,18); zeros(18,21)]]);  
              
        H = [eye(21);  eye(3), zeros(3,18); zeros(18,21)];
%         H = [eye(3), zeros(3,18); zeros(18,21);  eye(3), zeros(3,18); zeros(18,21)];
%         PM_ins = pinv(H.' * W * H);
% 
%         PM_ins(1,1) %why is this exploding??
%         PM_lidar(1,1)
%         PM_ins

        %lla_lidar = (lla_ins at last fusion timestamp) + (xHatM_lidar in LLA)
        %update cumulative counts for this segment to include current xHatM estimates 
%         xHatM_ins_cum = xHatM_ins + xHatM_ins_cum.';
        xHatM_lidar_cum = xHatM_lidar + xHatM_lidar_cum.';
        lla_lidar = lla_ins_last + [xHatM_lidar_cum(1)/coeff(1,1), xHatM_lidar_cum(2)/coeff(2,2), xHatM_lidar_cum(3)];
                
%         lla_ins = pinv((H.')*W*H)*(H.')*W*[lla_ins.'; zeros(18,1); lla_lidar.'; zeros(18,1)];
% %         %resize to 3x1
%         lla_ins = lla_ins(1:3).';
% %         lla_ins
        
%          ~~~~~~~~~~~~~~~
        lla_ins = lla_lidar; %temp
        qbn = qbn_lidar;
%          ~~~~~~~~~~~~~~~
%         [xHatM_ins_E, xHatM_ins_N, xHatM_ins_U] = geodetic2enu(lla_ins(1), lla_ins(2), lla_ins(3), lla0_ins(1), lla0_ins(2), lla0_ins(3), wgs84Ellipsoid, 'radians');
%         xHatM_ins(1:3) = [xHatM_ins_N, xHatM_ins_E, -xHatM_ins_U];
       
        %NOTE: WLS seems to be favoring ins over lidar...
%         weights = pinv((H.')*W*H)*(H.')*W;
%         weights(1,1)  %~0.95
%         weights(1,22) %~0.05
        
        %----------------------------------------------------------
        
        %test --------------
%         PM = PM_ins;
%         PM_ins = zeros(21);
%         PM_ins = eye(21)*gps_std;
%         PM_ins(4:end,4:end) = 0;
        PM_combined = PM_ins; %temp placeholder
        
        %TODO - need to convert xHatM_combined back to lla
%         lla_combined = lla0_combined + xHatM_combined(1:3).'; %wrong...
%         lla_combined(1)
        %--------------------
        
        F200 = F; %reset product
        xHatM_combined = xHatM_ins;
        
    else
        lla_combined = lla_ins;
        xHatM_combined = xHatM_ins;
    end
%     xHatM = xHatM_ins; % not sure if correct...
    
    %R == [std_gps_lat, std_gps_lon, std_gps_hgt]
    R = 0.01*diag(gpsmsr(5:7).^2);  % R matrix -- note coefficient of 0.01!!!
    dr = [0;0;0];  % LEVER ARM in m (XYZ body frame, YX(-Z) aligns with NED)
    lla_gps_corr = coeff*(gpsmsr(2:4))' - cbn*dr; %from Liangchun
    
%     'lla_in'
%     lla_in(1) % ~=0.6782 rad
%     lla_gps_corr
%     lla_in(1) - gpsmsr(2)

    gpsResUpd = [gpstime, lla_gps_corr'];
    if fuse_gps == 1
        %TODO DEBUG: lla_in (rad), gpsmsr (rad)
        y = coeff*lla_in' - lla_gps_corr; %this is in enu   
        
        %Kalman Correction Step --------------------------
        %FROM GPS DATA
        H = zeros(3,21);
        %H transforms (lla IN RADIANS!!) -> (m)
        H(1:3, 1:3) = coeff;

        L = PM_ins*H'*pinv(H*PM*H'+R, 1e-20); %added tol to help with singular inversion

        yHat = H*xHatM; %was this
%         yHat = xHatM(1:3); % using this since xHatM is already in (m)
        xHatP = xHatM + L*(y-yHat);         % a posteriori estimate (m) 
        
%         c = L*(y-yHat);
%         c(7:end)        

%         coeff*lla_in'
%         lla_gps_corr

        xHatM_ins = xHatP;
                
        PP = (eye(size(F))-L*H)*PM*(eye(size(F))-L*H)'+L*R*L';
        %-------------------------------------------------

    %     L(1,1) % 1.5e-7 == 1/coeff(1,1)
    %     coeff(1,1) = 6.36e6, coeff(2,2) = 4.97e6
    %     'y' %debug -> currently oscillating and too big
    %     y
    %     'yHat'
    %     yHat
    %     lla_ins(1)


        %debug here - be careful with coeff, it only works with rad!!
%         lla_ins = lla_ins - (xHatP(1:3))'; %was this, incorrect since xHatP is in enu
        
        %trying this
%         xHatP_enu = [xHatP(1)*L(1,1), xHatP(2)*L(2,2), xHatP(3)];
%         xHatP_enu(1)
%         lla_ins = lla_ins - xHatP_enu;
        
        %DEBUG ONLY!! (drags lla ins to be on gpsmsr) ---------
%         lla_ins = gpsmsr(2:4);
        %------------------------------------------------------
        
        %reset PM_lidar
%         PM_lidar = eye(3)*gps_std;
    end
        
%     xi = rad2deg(xHatM_ins(7:9));
%     E = [0 -xi(3) xi(2); xi(3) 0 -xi(1); -xi(2) xi(1) 0];
%     cbn = (eye(3)+E)*cbn;
    %TODO: DEBUG THIS ---------------------------------------------
    %apply velocity correction
    %NOTE: looks like this is removing almost all velocity (xHatP is too big)
    vel = vel - (xHatM_ins(4:6))'; % was this (causes stepping behavior)
%     vel = xHatM_ins(4:6).';
    %---------------------------------------------------------------
    
    %comment out below to ignore correction step- not ideal but helps
    %convergence?? -LX
%     qbn = dcm2quat(cbn');

    % normalization
%     e_q = sumsqr(qbn);
%     e_q = 1 - 0.5 * (e_q - 1);
%     qbn = e_q.*qbn;
    qbn = quatnormalize(qbn);

    % flip the sign if two quaternions are opposite in sign
    if (qbn(1)*qbn0(1)<0)
        qbn = -qbn;
    end

    rpy = flip(quat2eul(qbn));

    dv = vel - vel0;

    gpsUpdated = 1; % flag for INS fusion, regardless if GPS is actually used
    gpsCnt = gpsCnt + 1;
    
    %--------------------
    
    lla_combined = lla_ins(1:3); %temp
    
else
    lla_combined = lla0_combined;
%     xHatP = xHatM; %TODO-- figure out if I should still output this...
    PP = PM;
    xHatM_combined = xHatM_ins; %placeholder
    
    gpsResUpd = [];
    gpsUpdated = 0;
end

PM_combined = PM_ins; %placeholder

end

function PP = PPinitialize
%function PP = PPinitialize
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