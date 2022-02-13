%GPS_INS_SPAN Loosely Coupled - version 11/22/2021

%TODO: x_ins is currently holding full changes in state per timestep-
%       (rather than accumulated bias errors as specified in the doc)
%       this is incorrect and we need a new variable to hold changes in x
%TODO: get rid of *_combined vars, instead overwrite INS data
%TODO: remap inputs and outputs of main func into data structure
%TODO: while no new lidar updates, set WLS to entirely INS

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
pos_lidar_enu = load('pos_lidar_enu.mat').pos_lidar_enu;
% position: [2851×7 double]:
    % Columns 1-3: Translation vector (x,y,z) (in meters)
    % Columns 4-7: Rotational vector (qw,qx,qy,qz) (in rad)
 
%DEBUG: is this actually used?? -no
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
dpos_lidar = [(t_lidar + ppptime(1)), dN(1:end-1), dE(1:end-1), -dU(1:end-1)];

%debug: get change in velocity from lidar
ddE = dE(:) - [dE(2:end).' 0].';
ddN = dN(:) - [dN(2:end).' 0].';
ddU = dU(:) - [dU(2:end).' 0].';
dvel_lidar = [ddN(1:end-1), ddE(1:end-1), ddU(1:end-1)]; %not used
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
%   1) Samples at coincide with every 10th lidar point
%   2) Interpolates in regions with Lidar data outages
t_gps_new = t_lidar(1:10:end);
t_gps_new = [t_gps_new(1:112); linspace(139.5, 185.5, 47).'; t_gps_new(113:122); ...
    linspace(196.3,244.3,48).'; t_gps_new(124:138); linspace(259.4, 303.4, 44).'; ...
    t_gps_new(140:151); linspace(316.9, 363.9,47).';  t_gps_new(152:161);...
    linspace(375.2, 429.2, 54).'; t_gps_new(163:end) ];

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

%using gps time basis
% gpspos = [ppptime ppplat ppplon ppphgt ppplatstd ppplonstd ppphgtstd];
%using lidar time basis
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
% imuraw = [imutime deg2rad(gy) deg2rad(gx) -deg2rad(gz) ay ax -az];  % imuraw = [imutime gx gy gz ax ay az];
% imuraw = [imutime deg2rad(gy) deg2rad(gx) deg2rad(gz) ay ax -az];  % best when fusing with qbn_lidar??
% imuraw = [imutime -deg2rad(gy) deg2rad(gx) -deg2rad(gz) -ay ax -az]; %flips both lat and lon estimates...
imuraw = [imutime deg2rad(gy) -deg2rad(gx) -deg2rad(gz) ay -ax -az]; %best soln when not fusing qbn_lidar
% imuraw = [imutime deg2rad(gy) deg2rad(gx) deg2rad(gz) ay -ax -az]; %test


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
x_ins = double(zeros(21,1));            % state estimate
x_lidar = double(zeros(3,1));
x_ins_hist = double(zeros(gpsEndIndx, 21)); %cumulative vars for summing between GPS keyframes
x_lidar_hist = double(zeros(gpsEndIndx, 3));
xHatM_combined_hist = double(zeros(gpsEndIndx, 21));
ba = zeros(3,1);                % bias accels
bg = zeros(3,1);                % bias gyros
sa = zeros(3,1);                % scale factor accels
sg = zeros(3,1);                % scale factor gyros
PM_ins = PPinitialize;              % state-error matrix
PP_ins_last = PM_ins;               % init matrix that stores value of PP the last time fusion was performed
PM_combined = PM_ins;               % init PM_combined value to PP
Ncol = 22;
msrArr = zeros(7,endIndx-2);        % INS measurement array
resArr = zeros(endIndx-2,Ncol-1);   % Storage array of state-perturbation error results
resArr_lidar = zeros(endIndx-2,Ncol-1);  
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

%set noise covariance matrix for lidar
Qk_lidar = zeros(3,3); %test w/ zeros
% from Final_Slides.pptx - experimentally determined values from the summer
% Qk_lidar(1,1) = 0.000784; %(m) 
% Qk_lidar(2,2) = 0.000484; %(m)
% Qk_lidar(3,3) = 0.005625; %(m)
Qk_lidar(1,1) = 1.4e-17; %sigmalon**2 (deg)
Qk_lidar(2,2) = 1.4e-17; %sigmalat**2 (deg)
Qk_lidar(3,3) = 0.0001;%sigmaz**2 (meters)

% Seed initial measurements
msrk = imuraw(1,:);
msrk = msrk';
gpsmsr = gpspos(1,:);
lidarmsr = dpos_lidar(1, :);
ned0_lidar = [0, 0, gpspos(1,4)]; %lla0_ins;
lla0_combined = lla0_ins;
lla_ins_last = lla0_ins; %lla_in the last time GPS was called
ned_lidar_last = [0, 0, 0];
lla_combined_hist = zeros(3, gpsEndIndx); %array to hold lla_combined data
t_last_fuse = msrk(1);
x_ins_cum = zeros(21,1);
x_lidar_cum = zeros(3,1);

%init matrices for storing PM info
PM_hist_ins = zeros(endIndx,3);
PM_hist_lidar = zeros(endIndx, 3);
PM_hist_combined = zeros(endIndx, 3);
PM_lidar_last = 0*eye(3); %init uncertainty for lidar
PM_ins = zeros(21);
PM_ins(1:3,1:3) = eye(3)*1e-17;

%init F
F = zeros(21,21);
F200 = zeros(21);
Qk200 = zeros(21,21);

%init ins qbn to lidar
qbn0 = pos_lidar_enu(1, 4:7);

% Start Loop
gpsCnt = 0;
lidarCnt = 0;
i = 2;
while i < endIndx  
    
    if(mod(i,1e3)==2); i-2, end  % echo time step to screen current step
    
    %bring in lidar rotations to correct INS data
    qbn_lidar = pos_lidar_enu(lidarCnt+1, 4:7);
    %Lidar qbn frame is upside down- need to flip sign of yaw
    lidar_rpy = flip(quat2eul(qbn_lidar));
    lidar_rpy(3) = -lidar_rpy(3);
    lidar_rpy(1) = 0; %remove roll and pitch components of lidar (should be 0)
    lidar_rpy(2) = 0;
    qbn_lidar = eul2quat(flip(lidar_rpy));
%     ins_rpy = flip(quat2eul(qbn0)) %for debug
%     lidar_rpy %for debug
    
    %timeshifted to match GPS/Lidar
    %can't just cut off the first 1109*200 elements since we need the
    %timestamps from imuraw intact
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
    [lla_ins, ned_lidar, lla_combined, vel, rpy, dv, qbn, xHatM_ins, x_ins, x_lidar, xHatM_combined, gpsUpdated, lidarUpdated, gpsCnt, lidarCnt, gpsResUpd, PM_ins, PM_lidar, PM_combined, F, Qk200] = ...
        EKF(lla0_ins, ned0_lidar, lla0_combined, lla_ins_last, vel0, dv0, qbn0, msrk, msrk1, gpsmsr, lidarmsr, x_ins, x_lidar, x_lidar_cum, PM_ins, PP_ins_last, PM_combined, gpsCnt, lidarCnt, PM_lidar_last, fuse_lidar, fuse_gps, Qk200);   
    
     %only considering diagonal elements fixes negative eig bug
    %considering time since last lidar reset
    W = pinv([diag(diag(PM_ins)), diag(diag(F200*PP_ins_last));
          diag(diag(PP_ins_last*(F200.'))), diag(diag(PP_ins_last)) + [10*(msrk1(1)-t_last_fuse)*Qk_lidar, zeros(3,18); zeros(18,21)]], 1e-20);

    H = [eye(21);  eye(3), zeros(3,18); zeros(18,21)];
    PM_combined = pinv(H.' * W * H);   
    
%     interval = 2e3; %10s
%     interval = 1e3; %5s
    interval = 2e2; %1s
    if fuse_lidar == 1
        if  mod(i,interval) == 0
            
            %reset INS and Lidar covariance to combined estiamte
            PM_ins(1:3,1:3) = PM_combined(1:3,1:3);
            PM_lidar(1:3,1:3) = PM_combined(1:3,1:3);
%             weights = pinv((H.')*W*H, 1e-20)*(H.')*W;
%             'ins'
%             weights(1,1)  %ins
%             'lidar'
%             weights(1,22) %lidar
            
            %Correct position estimates of INS ----------------------------
            %WLS LLA -> not correct: we want to do WLS on CHANGES in position since last update
%             [Lat0, Lon0, Alt0] = enu2geodetic(resArr_lidar(i-2,2), resArr_lidar(i-2,1), -resArr_lidar(i-2,3), ...
%                                     ppplat(1), ppplon(1), ppphgt(1), wgs84Ellipsoid('m'), 'degrees');
%             lla_lidar = [deg2rad(Lat0), deg2rad(Lon0), resArr_lidar(i-2,3)];            
%             lla0_ins = pinv((H.')*W*H)*(H.')*W*[lla_ins.'; zeros(18,1); lla_lidar.'; zeros(18,1)];
%             lla0_ins = lla0_ins(1:3).'; %reshape to 3x1

%             %WLS on changes in LLA~~~~~~~
%             [lat_lidar, lon_lidar, h_lidar] = enu2geodetic(ned_lidar(2), ned_lidar(1), -ned_lidar(3), ...
%                                     ppplat(1), ppplon(1), ppphgt(1), wgs84Ellipsoid('m'), 'degrees');
%             lla_lidar = [lat_lidar, lon_lidar, h_lidar];
%             dlla_lidar = lla_lidar - lla_lidar_last; %not working -> scaling issue??

            [Lat_delta, Lon_delta, Alt_delta] = enu2geodetic(ned_lidar(2) - ned_lidar_last(2), ned_lidar(1) - ned_lidar_last(1), -ned_lidar(3) + ned_lidar_last(3), ...
                                    ppplat(1), ppplon(1), ppphgt(1), wgs84Ellipsoid('m'), 'degrees');
            dlla_lidar = [deg2rad(Lat_delta), deg2rad(Lon_delta), Alt_delta] - [deg2rad(ppplat(1)), deg2rad(ppplon(1)), ppphgt(1)];                                
            dlla_ins = lla_ins - lla_ins_last;                                
            
            wls_dlla = pinv((H.')*W*H, 1e-20)*(H.')*W*[dlla_ins.'; zeros(18,1); dlla_lidar.'; zeros(18,1)];
            lla_ins = lla_ins_last + wls_dlla(1:3).';
            lla0_ins = lla_ins;
            
            ned_lidar_last = ned_lidar;
            lla_ins_last = lla_ins;
%             %~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
              %Naive: simply reset lla_ins to lidar
%             lla0_ins = lla_lidar;            
            %--------------------------------------------------------------

            %Correct velocity estimates of INS-----------------------------                               
            dt_lidar = dpos_lidar(lidarCnt+1,1) - dpos_lidar(lidarCnt,1);
            vel_lidar = [dpos_lidar(lidarCnt, 3)/dt_lidar, dpos_lidar(lidarCnt, 2)/dt_lidar, -dpos_lidar(lidarCnt, 4)/dt_lidar];
            
            %WLS absolute vel update
%             vel0 = pinv((H.')*W*H, 1e-20)*(H.')*W*[vel.'; zeros(18,1); vel_lidar.'; zeros(18,1)];
%             vel0 = vel0(1:3).';

            % Naive: reset ins velocity to lidar velocity estimate
            vel0 = vel_lidar; 

            % TODO: do WLS for CHANGE in velocity between subsequent fusion periods

            %--------------------------------------------------------------

            
            %reset INS heading to lidar
            qbn0 = qbn_lidar;             
            
            %reset INS absolute position covariance to Lidar
            PM_ins = diag(diag(PM_ins)); %remove all non-diagonal elements
            PM_ins(1:3,1:3) = PM_lidar; %reset positon covariance
            %reset INS velocity covariance
            PM_ins(4,4) = PM_lidar(1,1);
            PM_ins(5,5) = PM_lidar(2,2);
            PM_ins(6,6) = PM_lidar(3,3);    
            
            %update baseline
            PP_ins_last = PM_ins;
            F200 = F;

            t_last_fuse = msrk1(1);

        else
            F200 = F200*F;
            lla0_ins = lla_ins;
            qbn0 = qbn;
            vel0 = vel; 
        end
    end
        
    %save sigmas in units of (m)
    PM_hist_lidar(i-1,:) = [sqrt(PM_lidar(1,1))*6.36e6 sqrt(PM_lidar(2,2))*4.97e6, sqrt(PM_lidar(3,3))];
    PM_hist_ins(i-1,:) = [sqrt(PM_ins(1,1))*6.36e6 sqrt(PM_ins(2,2))*4.97e6, sqrt(PM_ins(3,3))]; %was this, thinking it should be in (m) though
    PM_hist_combined(i-1,:) = [sqrt(PM_combined(1,1))*6.36e6 sqrt(PM_combined(2,2))*4.97e6, sqrt(PM_combined(3,3))];

    if lidarUpdated == 1
        lidarmsr = dpos_lidar(lidarCnt+1, :);%update lidar measurement
        ned0_lidar = ned_lidar;
        if gpsUpdated == 1
            x_lidar_hist(gpsCnt,:) = x_lidar_hist(gpsCnt,:) + x_lidar.';
%             lla_ins_last = lla_ins;
        else
            x_lidar_hist(gpsCnt+1,:) = x_lidar_hist(gpsCnt+1,:) + x_lidar.';
        end
    end
    
    if gpsUpdated == 1
        x_ins_hist(gpsCnt,:) = x_ins_hist(gpsCnt,:) + x_ins.';
        gpsmsr = gpspos(gpsCnt+1, :);   
        gps_res(gpsCnt,:) = gpsResUpd;
        x_ins(1:9) = zeros(9,1);        % Reset error states %was xHatP_ins
        xHatM_combined_hist(gpsCnt,:) = xHatM_combined;
        lla_combined_hist(:,gpsCnt) = lla_combined;
        lla0_combined = lla_combined;       
    else
        x_ins_hist(gpsCnt+1,:) = x_ins_hist(gpsCnt+1,:) + x_ins.';
    end

    % Store Data
    resArr_ins(i-1,:) = [lla_ins vel rpy rad2deg(bg')*3600 (ba')*10^6/9.7803267715 (sg')*10^-6 (sa')*10^-6];
    resArr_lidar(i-1,:) = [ned_lidar vel rpy rad2deg(bg')*3600 (ba')*10^6/9.7803267715 (sg')*10^-6 (sa')*10^-6];
    qbnArr(i-1,:) = qbn;
    if gpsUpdated == 1 && ~isempty(gpsmsr)
        gpsMsrArr(:,gpsCnt) = gpsmsr(1:4)';
    end
    
    PM_lidar_last = PM_lidar;

    msrArr(:,i-1) = msrk1;
    rpy0 = rpy;
    dv0 = dv; 
    msrk = msrk1;
    
    x_lidar_cum = x_lidar_hist(gpsCnt+1,:);
    
    i = i+1;  
end 

% (((more plots available in Liangchun's file)))

time = msrArr(1,:);
startTime = time(1);

%convert GPS measurements array to ENU
[Ngps, Egps, Ugps] = geodetic2enu(gpsMsrArr(2,:), gpsMsrArr(3,:), gpsMsrArr(4,:), ...
    gpsMsrArr(2,1), gpsMsrArr(3,1), gpsMsrArr(4,1), wgs84Ellipsoid);
% Ngps = -Ngps;

%find timesteps where gpsMsrArr != 0
nonzero = find( gpsMsrArr(1,:) ~= 0); %nonzero elements of GPS
nonzero_combined = find(lla_combined_hist(1,:) ~= 0); %nonzero elements of INS+Lidar fused estiamtes (occur on GPS timesteps)

[Eins, Nins, Uins] = geodetic2enu(resArr_ins(:,1), resArr_ins(:,2), resArr_ins(:,3), ...
    resArr_ins(1,1), resArr_ins(1,2), resArr_ins(1,3), wgs84Ellipsoid, 'radians');

dEgps = Egps(nonzero);
dEgps = [dEgps(2:end) 0] - dEgps;
dNgps = Ngps(nonzero);
dNgps = [dNgps(2:end) 0] - dNgps;
dUgps = Ugps(nonzero);
dUgps = [dUgps(2:end) 0] - dUgps;

%plot velocity ----------------------------------------------------
figure()
subplot(3,1,1)
title('Estimated Velocity')
hold on
xlabel('time (s)')
ylabel('East (m/s)')
plot(x_ins_hist(:,1))
plot(x_lidar_hist(:,1))
% plot(xHatM_combined_hist(:,1))
plot([0 dEgps(1:end-1)])
% legend('INS', 'Lidar', 'xHat- combined', 'GPS')
legend('INS', 'Lidar', 'GPS')

subplot(3,1,2)
hold on
xlabel('time (s)')
%TODO: make sure this is still lon/ lat after ENU->NED swap
ylabel('North (m/s)')
plot(x_ins_hist(:,2))
plot(x_lidar_hist(:,2))
% plot(xHatM_combined_hist(:,2))
plot([0 dNgps(1:end-1)])
% legend('xHat- INS', 'xHat- Lidar', 'xHat- combined', 'gps baseline')
legend('INS', 'Lidar', 'GPS')

subplot(3,1,3)
hold on
xlabel('time (s)')
%TODO: make sure this is still lon/ lat after ENU->NED swap
ylabel('Up (m/s)')
plot(x_ins_hist(:,3))
plot(x_lidar_hist(:,3))
% plot(xHatM_combined_hist(:,3))
plot([0 dUgps(1:end-1)])
% legend('xHat- INS', 'xHat- Lidar', 'xHat- combined', 'gps baseline')
legend('INS', 'Lidar', 'GPS')
%--------------------------------------------------------------------

%covaraince plots ---------------------------------------------------
figure(5)
title('Covariance')
% sgtitle('Lidar corrected INS')
% sgtitle('Lidar and INS, no correction')
% hold on %putting this here prevents logy axis
skip_start = 1; %2500;

%log--
semilogy(time(skip_start:end) - startTime, PM_hist_ins(skip_start:end-2,1), ...
    time(skip_start:end) - startTime, PM_hist_lidar(skip_start:end-2,1), ...
    time(skip_start:end) - startTime, PM_hist_combined(skip_start:end-2,1))
%-----
%standard--
% hold on
% plot(time(skip_start:end) - startTime, PM_hist_ins(skip_start:end-2,1))
% plot(time(skip_start:end) - startTime, PM_hist_lidar(skip_start:end-2,1))
% plot(time(skip_start:end) - startTime, PM_hist_combined(skip_start:end-2,1))
%----------

xlim([50, 60])
grid on
legend('\sigma_{lon,ins}', '\sigma_{lon,lidar}', '\sigma_{lon,combined}')
% legend('\sigma_{lon,ins}', '\sigma_{lon,lidar}')
xlabel('time (s)')
ylabel('standard deviation of error (m)')
%--------------------------------------------------------------------

figure(6)
% Postion Plots -------------------------------------------------------
subplot(3,1,1);
title('Estimated Position')
hold on; grid on
% plot(time-startTime, rad2deg(resArr_ins(:,1))); %was this with ins in lla
plot(time-startTime, Nins)
% plot(time-startTime, rad2deg(resArr_lidar(:,1))); %was this with lidar in lla
plot(time-startTime, resArr_lidar(:,1));
% plot(time-startTime, rad2deg(lla_combined_hist(1,1:end-1))); %not working
% plot(gpsMsrArr(1, nonzero)-startTime, gpsMsrArr(2, nonzero)); %used for lat/lon
plot(gpsMsrArr(1, nonzero)-startTime, Egps(nonzero));
% plot(gpsMsrArr(1,nonzero_combined)-startTime, lla_combined_hist(1,nonzero_combined))
% legend('LLA INS', 'LLA Lidar Dead Reckoning', 'pure GPS', 'Lidar and INS only')
legend('INS', 'Lidar', 'GPS')
hold on; grid on
ylabel('East (m)');

subplot(3,1,2);
hold on; grid on
% plot(time-startTime, rad2deg(resArr_ins(:,2)));
plot(time-startTime, Eins)
% plot(time-startTime, rad2deg(resArr_lidar(:,2)));
plot(time-startTime, resArr_lidar(:,2));
ylabel('longitude (deg)');
% plot(gpsMsrArr(1, nonzero)-startTime, gpsMsrArr(3, nonzero));
plot(gpsMsrArr(1, nonzero)-startTime, Ngps(nonzero));
% plot(gpsMsrArr(1,nonzero_combined)-startTime, lla_combined_hist(2,nonzero_combined))
% legend('LLA INS', 'LLA Lidar Dead Reckoning', 'pure GPS', 'Lidar and INS only')
% legend('LLA INS', 'LLA Lidar Dead Reckoning', 'pure GPS')
legend('INS', 'Lidar', 'GPS')
hold on; grid on
ylabel('North (m)');

subplot(3,1,3);
hold on; grid on
plot(time-startTime, resArr_ins(:,3));
plot(time-startTime, resArr_lidar(:,3));
%     plot(gpsMsrArr(1, :)-startTime, gpsMsrArr(4, :));
plot(gpsMsrArr(1, nonzero)-startTime, gpsMsrArr(4, nonzero));  
% legend('LLA INS', 'LLA Lidar Dead Reckoning', 'pure GPS')
legend('INS', 'Lidar', 'GPS')
hold on; grid on
ylabel('altitude (m)');
xlabel('Time (s)');
%-----------------------------------------------------------------------

% %Quaternion plot -----------------------------------------------------
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
% %----------------------------------------------------------------------


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