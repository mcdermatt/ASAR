% d1_imu.m:

clear all
close all
clc

% TM- This switch determines whether you're using the pure INS function or
%     the INS/GPS fusion function. 1 is pure INS, 0 is fusion. 
pureINS = 1;

msrArr = zeros(7,1);
gpsMsrArr = zeros(4,1);
xHatP = zeros(21,1);
ba = zeros(3,1);
bg = zeros(3,1);
sa = zeros(3,1);
sg = zeros(3,1);
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

qbnArr = zeros(1,4);
qbnRef = zeros(1,4);

% TM- These next lines read and store input variables from the data files

readoem7memmap_update

timeData = sow1462; %rawimugnsssow;
timeData = timeData - timeData(1);
startInsData = [rawimuxaccel rawimuyaccel rawimuzaccel rawimuxgyro rawimuygyro rawimuzgyro];
gpsData = [inslat inslon inshgt]; % Just a placeholder for now
rpyRef = [insroll inspitch insazim]; % THIS IS IN DEGREES
pvaxTime = sow1465 - sow1465(1);

%% Messing with variables

indexFound = false;
startIndex = -1;
i = 1;
while i < 1200 && ~ indexFound
    disp(cast(timeData(i), 'int32'))
    if cast(timeData(i), 'int32') == 1100
        startIndex = i;
        indexFound = true;
    end
    i = i + 1;
end

fprintf('START INDEX: %i', startIndex)


%qbnRef = load('./Data/qbn.txt');
qbnStart = zeros(length(insroll), 4);

%{
rpy1 = rpyRef(1,:);
quat1 = eul2quat(rpy1, 'XYZ');
rpy2 = quat2eul(quat1, 'XYZ'); 

disp(rpy1)
disp(quat1)
disp(rpy2)
%}

for i = startIndex:length(insroll)
    qbnStart(i,:) = eul2quat(flip(rpyRef(i,:)));
end

loopLimit = startIndex + 90; %length(timeData);
factor = cast(loopLimit/1694, 'int32') + 1;
fprintf("Array Scale factor: %i", factor)

qbnRef = stretchArray(qbnStart, factor);
% TM- This sets constants needed for both mechanization functions
%lla0 = [deg2rad(37.8025488), deg2rad(-122.4156203), -11.048];   % from gps.txt
lla0 = [lat, lon, h];
vel0 = [0.022, 0.180, 0.026];                                   % from nav_pvt.txt

if pureINS == 1
    Ncol = 10;
    resArr = zeros(1,9);
    trueFile = gpsData;
    %trueFile = fopen(gpsFn);
else
    Ncol = 22;
    resArr = zeros(1,21);
    nedArr = zeros(1,3);
    trueFile = gpsData;
    %trueFile = fopen(gpsFn);
end


true_res = zeros(1,Ncol);
gps_res = zeros(1,4);


rpy0 = rpyRef(1,:);
dv0 = 0;
qbn0 = eul2quat(flip(rpy0));

bias = 0;

qbnArr = [qbnArr; qbn0];

if pureINS == 0
    %gpsmsr = cell2mat(textscan(gpsFile, '%f %f %f %f %f %f %f\n',1));
    gpsmsr = [1.5670*1.0e+09, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
end

gpsCnt = 0;

% TM- This while loop iterates through the entire data file, which creates
%     the updates and calculations of sensor measurement 

msrk = getMsrkVector(startInsData(startIndex,:), timeData(startIndex));

i = startIndex + 1;
while i < loopLimit% size(insData, 1)
%while ~feof(insFile)
% while ~feof(insFile) && i < 500
    %msrk1 = fread(insFile,7,'double');
    
    msrk1 = getMsrkVector(startInsData(i,:), timeData(i));
    
    if isempty(msrk1) 
        break;
    
    end
    
    if pureINS == 0 && isempty(gpsmsr)
        break;
    end
    
    dt = msrk1(1) - msrk(1);
    msrk1(7) =  msrk1(7) - bias*dt;
        
    % TM- This if/else determines whether the pure INS or GPS aided INS 
    %     solution will be used 
    
    if pureINS == 1
        [lla, vel, rpy, dv, qbn, win] = ins_mechanization_compact(lla0, vel0, rpy0, dv0, qbn0, msrk, msrk1);
        
        resArr = [resArr; lla vel rpy];
        qbnArr = [qbnArr; qbn];
    
    else
    
        % GPS/INS solution
        [lla, vel, rpy, ned, dv, qbn, xHatP, PP, gpsUpdated, gpsCnt, gps_res] = ins_gps(lla0, vel0, rpy0, dv0, qbn0, msrk, msrk1, gpsmsr, xHatP, PP, gpsCnt, gps_res);

        if gpsUpdated == 1
            %gpsmsr = cell2mat(textscan(gpsFile, '%f %f %f %f %f %f %f\n',1));      
            gpsmsr = [1.5670*1.0e+09, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    
            xHatP(1:9) = zeros(9,1);
            
        end
        
        resArr = [resArr; lla vel rpy rad2deg(bg')*3600 (ba')*10^6/9.7803267715 (sg')*10^-6 (sa')*10^-6];
        
        if pureINS == 0
            nedArr = [nedArr; ned]; % TM- This notation adds the newest ned and qbn as a new row in the matrices nedArr and qbnArr 
            qbnArr = [qbnArr; qbn];
            
            if i == 10186
                a = 0;
            end
        end
    
    end
    
    msrArr = [msrArr msrk1];
    
    if pureINS == 0 && gpsUpdated == 1 && ~isempty(gpsmsr)
        gpsMsrArr = [gpsMsrArr gpsmsr(1:4)'];
    end
    
    % TM- At the end of each iteration of the loop, all the final product
    %     variables of the current calculation are reestablished as the 
    %     initial variables of the next calculation
    i = i+1;
    
    lla0 = lla; vel0 = vel; rpy0 = rpy;
    dv0 = dv; qbn0 = qbn;
    msrk = msrk1;    
end 

msrArr = msrArr(:,2:end);
qbnArr = qbnArr(2:end,:);

if pureINS == 0
    gpsMsrArr = gpsMsrArr(:,2:end);
end

if pureINS == 0
    gps_res = gps_res(2:end,:);
end

resArr = resArr(2:end,:);

if pureINS == 0
    nedArr = nedArr(2:end,:);
end

% plots
% TM- The following lines graph the final data after all the calculations
%     are complete
time = msrArr(1,:);
startTime = time(1);

% ppplatData = reshape(ppplat, );

disp("BIG SIZES: ")
disp(size(msrArr))
disp(size(resArr))


%% Delta Theta (XYZ) Graphs

limit = length(time);

figure
subplot(3,1,1);
plot(time-startTime, rawimuxgyro(1:limit));
hold on; grid on
ylabel('{{\Delta}{\theta}_x} (deg)');
xlabel('Time (s)');
title('Raw IMU Rotation Values')
subplot(3,1,2);
plot(time-startTime, rawimuygyro(1:limit));
hold on; grid on
ylabel('{{\Delta}{\theta}_y} (deg)');
xlabel('Time (s)');
subplot(3,1,3);
plot(time-startTime, rawimuzgyro(1:limit));
hold on; grid on
ylabel('{{\Delta}{\theta}_z} (deg)');
xlabel('Time (s)');

figure
subplot(3,1,1);
plot(time-startTime, rad2deg(msrArr(2,:)));
hold on; grid on
ylabel('{{\Delta}{\theta}_x} (deg)');
xlabel('Time (s)');
title('MSRK Input Rotation Values')
subplot(3,1,2);
plot(time-startTime, rad2deg(msrArr(3,:)));
hold on; grid on
ylabel('{{\Delta}{\theta}_y} (deg)');
xlabel('Time (s)');
subplot(3,1,3);
plot(time-startTime, rad2deg(msrArr(4,:)));
hold on; grid on
ylabel('{{\Delta}{\theta}_z} (deg)');
xlabel('Time (s)');

%% Delta Velocity (XYZ) Graphs
figure
subplot(3,1,1);
plot(time-startTime, msrArr(5,:));
hold on; grid on
ylabel('{{\Delta}v_x} (m/s)');
xlabel('Time (s)');
title('MSRK Input Velocity Values')
subplot(3,1,2);
plot(time-startTime, msrArr(6,:));
hold on; grid on
ylabel('{{\Delta}v_y} (m/s)');
xlabel('Time (s)');
subplot(3,1,3);
plot(time-startTime, msrArr(7,:));
hold on; grid on
ylabel('{{\Delta}v_z} (m/s)');
xlabel('Time (s)');

figure
subplot(3,1,1);
plot(time-startTime, rawimuxaccel(1:limit));
hold on; grid on
ylabel('{{\Delta}v_x} (m/s)');
xlabel('Time (s)');
title('Raw IMU Velocity Values')
subplot(3,1,2);
plot(time-startTime, rawimuyaccel(1:limit));
hold on; grid on
ylabel('{{\Delta}v_y} (m/s)');
xlabel('Time (s)');
subplot(3,1,3);
plot(time-startTime, rawimuzaccel(1:limit));
hold on; grid on
ylabel('{{\Delta}v_z} (m/s)');
xlabel('Time (s)');

%fclose(insFile);

%% Velocity (XYZ) Graphs
%len = length(time);

gnssLen = length(gnssvelhorspd);

GNSSTruthX = zeros(gnssLen, 1);
GNSSTruthY = zeros(gnssLen, 1);
GNSSTruthZ = zeros(gnssLen, 1);

for i = 1:gnssLen
    GNSSTruthX(i) = gnssvelhorspd(i) * cos(gnssveltrkgnd(i));
    GNSSTruthY(i) = gnssvelhorspd(i) * sin(gnssveltrkgnd(i));
    GNSSTruthZ(i) = gnssvelvertspd(i) * -1;
end

disp("Done")

figure
subplot(3,1,1);
plot(time-startTime, resArr(:,4));
hold on; grid on
ylabel('Algorithm Output (m/s)');
xlabel('Time (s)')
title('Velocity X')
subplot(3,1,2);
plot(pvaxTime, insnorthvel);
hold on; grid on
ylabel('INSPVAX (m/s)');
xlabel('Time (s)')
subplot(3,1,3);
plot(pvaxTime, GNSSTruthX);
hold on; grid on
ylabel('Both (m/s)');
xlabel('Time (s)');


figure
subplot(3,1,1);
plot(time-startTime, resArr(:,5));
hold on; grid on
ylabel('Algorithm Output (m/s)');
title('Velocity Y')
subplot(3,1,2);
plot(pvaxTime, inseastvel);
hold on; grid on
ylabel('INSPVAX (m/s)');
subplot(3,1,3);
plot(pvaxTime, GNSSTruthY);
hold on; grid on
ylabel('GNSSVel (m/s)');
xlabel('Time (s)');

figure
subplot(3,1,1);
plot(time-startTime, resArr(:,6));
hold on; grid on
ylabel('Algorithm Output (m/s)');
xlabel('Time (s)');
title('Velocity Z')
subplot(3,1,2);
plot(pvaxTime, insupvel);
hold on; grid on
ylabel('INSPVAX (m/s)');
xlabel('Time (s)');
subplot(3,1,3);
plot(pvaxTime, GNSSTruthZ);
hold on; grid on
ylabel('GNSSVel (m/s)');
xlabel('Time (s)');
xlabel('Time (s)');

disp('done2')

%% Latitude/Longitude/Height Graphs

%latLength = length(resArr(:,1));
%lonLength = length(resArr(:,2));
%hgtLength = length(resArr(:,3));

figure
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

kmlFolder = './';

filename = fullfile(kmlFolder,'urbanloco_lc.kml');
kmlwritepoint(filename,lat_lc,lon_lc,alt_lc, 'IconScale',1);

lat_gps = gpsMsrArr(2, :);
lon_gps = gpsMsrArr(3, :);
alt_gps = gpsMsrArr(4, :);

filename = fullfile(kmlFolder,'urbanloco_gps.kml');
kmlwritepoint(filename,lat_gps,lon_gps,alt_gps, 'Color','red', 'IconScale',1);

%% Quaternion Graphs
%len = length(qbnArr(:,1));
len = length(qbnRef(:,1));

figure
subplot(4,1,1);
plot(qbnArr(:,1));
hold on; grid on
plot(qbnRef(1:len,4));
ylabel('{q_1}');
subplot(4,1,2);
plot(qbnArr(:,2));
hold on; grid on
plot(qbnRef(1:len,2));
ylabel('{q_2}');
subplot(4,1,3);
plot(qbnArr(:,3));
hold on; grid on
plot(qbnRef(1:len,1));
ylabel('{q_3}');
subplot(4,1,4);
plot(qbnArr(:,4));
hold on; grid on
plot(-qbnRef(1:len,3));
ylabel('{q_4}');


figure
subplot(4,1,1);
hold on; grid on
plot(qbnStart(:,4));
ylabel('{q_1}');
subplot(4,1,2);
hold on; grid on
plot(qbnStart(:,2));
ylabel('{q_2}');
subplot(4,1,3);
hold on; grid on
plot(qbnStart(:,1));
ylabel('{q_3}');
subplot(4,1,4);
hold on; grid on
plot(-qbnStart(:,3));
ylabel('{q_4}');

%% Atan2 RPY Comparison

atan2Angle = atan2(msrArr(6,:), msrArr(5,:));

figure
subplot(2, 1, 1)
plot(time-startTime, atan2Angle)
hold on; grid on;
ylabel('Atan2 INS Angle (deg)')
xlabel('Time (s)')
title('COMPARISON: Atan2 and Yaw')
subplot(2, 1, 2)
plot(pvaxTime, insazim)
hold on; grid on;
ylabel('Actual Yaw (deg)');
xlabel('Time (s)');

%% INSPVAX Input Angle Graphs
figure
subplot(3,1,1);
plot(pvaxTime, insroll);
hold on; grid on
ylabel('Actual Roll (deg)');
xlabel('Time (s)');
title('INSPVAX RPY Values- original')
subplot(3,1,2);
plot(pvaxTime, inspitch);
hold on; grid on
ylabel('Actual Pitch (deg)');
xlabel('Time (s)');
subplot(3,1,3);
plot(pvaxTime, insazim);
hold on; grid on
ylabel('Actual Yaw (deg)');
xlabel('Time (s)');


% compute rn, rm
WGS84_A = 6378137.0;           % earth semi-major axis (WGS84) (m) 
WGS84_B = 6356752.3142;        % earth semi-minor axis (WGS84) (m) 
e = sqrt(WGS84_A * WGS84_A - WGS84_B * WGS84_B) / WGS84_A;
rn_list = WGS84_A ./ sqrt(1 - e * e * sin(resArr(:,1)).^2);
rm_list = WGS84_A * (1 - e * e) ./ sqrt(power(1 - e * e * sin(resArr(:,1)).^2, 3));

%% Functions Used

function newArray = stretchArray(oldArray, factor)
newArray = zeros(size(oldArray, 1)*factor, size(oldArray, 2));

%disp(size(oldArray))
%disp(size(newArray))

j = 1;
for i = 1:length(newArray)
    newArray(i,:) = oldArray(j,:);
    if mod(i, factor) == 0
        j = j + 1;
    end
end
end

function msrk = getMsrkVector(imuVector, time)
acclFactor = (0.0151515 / 65536) / 200;
gyroFactor  = ((0.4 / 65536) * (9.80665 / 1000)) / 200;

imuData = reshape(cast(imuVector, 'double'), 6, 1);
for i = 1:6
    if i <= 3
        imuData(i) = imuData(i) * acclFactor; % Adjust with Accelerometer Scale Factor
    else
        imuData(i) = imuData(i) * gyroFactor; % Adjust with Gyroscope Scale Factor
        imuData(i) = deg2rad(imuData(i));
    end
end

msrk = [time(1); imuData];
end

% ins_mechanization_compact.m:
function [lla, vel, rpy, dv, qbn, win] = ins_mechanization_compact(lla0, vel0, rpy0, dv0, qbn0, msrk, msrk1)
% TM- INPUTS AND OUTPUTS
%   -  lla0, lla: [latitude (deg), longitude (deg), altitude (m)]
%   -  vel0, vel: [v_x (m/s), v_y (m/s), v_z (m/s)] 
%   -  rpy0, rpy: Euler angles representing the attitude of the object
%   -   dv0,  dv: Overall change in velocity from previous step (m/s)
%   -  qbn0, qbn: A quaternion representing the attitude of the object
%   -  msrk: [t (s), w_x (rad/s), w_y (rad/s), w_z (rad/s), a_x (m/s^2), a_y (m/s^2), a_z (m/s^2)]
%   - msrk1: Array with the same components as msrk, but all are advanced by one timestep


WGS84_A = 6378137.0;           % earth semi-major axis (WGS84) (m) 
WGS84_B = 6356752.3142;        % earth semi-minor axis (WGS84) (m) 
e = sqrt(WGS84_A * WGS84_A - WGS84_B * WGS84_B) / WGS84_A;

% rotational angular velocity of earth
omega_e = 7.2921151467e-5;   

%fprintf('%i: %i\n', i, insData(i,1))

% TM- These next few lines unpack the input variables
dt = msrk1(1) - msrk(1);
gy0 = msrk(2:4);  % TM- delta theta k-1
ac0 = msrk(5:7);  % TM- delta v^b_f,k-1 - Integrated specific force (m/s?)
gy = msrk1(2:4);  % TM- delta theta k
ac = msrk1(5:7);  % TM- delta v^b_f,k  - Integrated specific force (m/s?)


% This big chunk checks the values and sizes of the starting vars
%{
fprintf("gy0: %d x %d\n", size(gy0, 1), size(gy0, 2))
disp(gy0)
fprintf("ac0: %d x %d\n", size(ac0, 1), size(ac0, 2))
disp(ac0)
fprintf("gy: %d x %d\n", size(gy, 1), size(gy, 2))
disp(gy)
fprintf("ac: %d x %d\n", size(ac, 1), size(ac, 2))
disp(ac)
disp("---------------------------------------------")
%}

% update the velocity
% TM- The following calculations are parts of Shin 2.3.3

% TM- These next few lines correspond to Shin (2.48b)- one component of 
%     the velocity update due to the specific force. 
dvfb_k = 1.0/12.0*(cross(gy0,ac) + cross(ac0,gy)) + 0.5*cross(gy,ac);
dvfb_k = ac + dvfb_k; % TM- Final calc- this is delta v^b(k-1)_f, k

% compute cbn0
% TM- Based on this line here, it is a direction cosine matrix representing the attitude
cbn0 = quat2dcm(qbn0);

% TM- qne seems to be a quaternion representing the inital orientation
%     based on latitude and longitude
qne = zeros(1,4);
qne(1) = cos(-pi / 4.0 - lla0(1) / 2.0) * cos(lla0(2) / 2.0);
qne(2) = -sin(-pi / 4.0 - lla0(1) / 2.0) * sin(lla0(2) / 2.0);
qne(3) = sin(-pi / 4.0 - lla0(1) / 2.0) * cos(lla0(2) / 2.0);
qne(4) = cos(-pi / 4.0 - lla0(1) / 2.0) * sin(lla0(2) / 2.0);

% TM- These next lines are the creation of the quaternion qnn_l: Shin Equation 2.51c 
[zeta_m, wie, wen] = get_zeta(dt, omega_e, lla0, vel0, WGS84_A, e, 1);
win = wie + wen;

qnn_l = make_quaternion(zeta_m, 0.5);
% TM- qnn_l here corresponds to the quaternion
%     q^n(k-1)_n(k-1/2), the result of 2.51c


% TM- With a very similar process as 2.51c, these next nine lines create qee_h: Shin Equation 2.51d
wiee = [0 0 omega_e];
xi_m = dt.*wiee/2.0;       % 0.5: half size for extrapolating lla at time(k+1/2) 
qee_h = make_quaternion(xi_m, -0.5); % TM- is the xi used in Shin 2.51d
% TM- The final product here, is qee_h, or Shin's quaternion
%     q^e(k-1/2)_e(k-1), the result of 2.51d


% TM- These next two lines use the twe previous quaternions qnn_l and
%     qee_h, as well as the quaternion defined way above with some lla trig
qne_l = quatmultiply(qne, qnn_l);     % TM- This is Shin 2.51a
qne_m = quatmultiply(qee_h, qne_l);   % TM- This is Shin 2.51b


% TM- This function uses the calculated quaternions to create an 
%     updated lla vector halfway through the timestep

lla_m = quatToLla(qne_m);
lla_m(3) = lla0(3) - (vel0(3) * dt) / 2.0;

% extrapolate the speed
% TM- This is Shin 2.52a
vel_m = vel0 + 0.5 * dv0;

gl_m = get_gl(lla0, lla_m);

% TM- Using the updated lla vector from above, calculate new wie and wen 
%     values for the time midpoint: _m
% compute true zeta
[zeta, wie_m, wen_m] = get_zeta(dt, omega_e, lla_m, vel_m, WGS84_A, e, 0);

cnn = eye(3,3); % TM- This creates a 3x3 identity matrix

% TM- This fills the non-diagonals of the matrix with true zeta multiples
cnn(1,2) = -0.5 * zeta(3);
cnn(1,3) =  0.5 * zeta(2);
cnn(2,1) =  0.5 * zeta(3);
cnn(2,3) = -0.5 * zeta(1);
cnn(3,1) = -0.5 * zeta(2);
cnn(3,2) =  0.5 * zeta(1);

% This chunk displays the values and sizes of the matrices calculated above
%{
disp(size(cbn0'))
disp(cbn0');
disp(size(cnn))
disp(cnn);
disp(size(dvfb_k))
disp(dvfb_k);
%}

% TM- These three lines combine all of the previous calculations into the
%     final variables needed to update the velocity
dvfn_k = (cbn0'*cnn)*dvfb_k;
dvfn_k = dvfn_k';
dvgn_k = (gl_m-cross((2*wie_m+wen_m), vel_m)).* dt; % TM- Shin 2.53

% update velocity
% TM- This is the big one, combining the two terms to create an overall
%     delta v, and then updating the overall velocity
%     One term is from gravity/coriolis, the other from specific force
dv = dvfn_k + dvgn_k;
vel = vel0 + dv;

% update the position
% TM- This corresponds to Shin Section 2.3.4
vel_m = 0.5*(vel0 + vel);

% recompute zeta, xi
[zeta_m, ~, ~] = get_zeta(dt, omega_e, lla_m, vel_m, WGS84_A, e, 0);

qnn_h = make_quaternion(zeta_m, 0.5);

xi_m = wiee.* dt;
qee_l = make_quaternion(xi_m, -0.5);

% recompute the qnn_h
qne_h = quatmultiply(qne, qnn_h);
qne = quatmultiply(qee_l, qne_h);

lla = quatToLla(qne);
lla(3) = lla0(3) - vel_m(3) * dt;

% update the attitude
% TM- This corresponds to Section 2.3.5
qdthe_half = zeros(1,4);
qne0 = zeros(1,4);

% compute qne0
qne0(1) = cos(-pi / 4.0 - lla0(1) / 2.0) * cos(lla0(2) / 2.0);
qne0(2) = -sin(-pi / 4.0 - lla0(1) / 2.0) * sin(lla0(2) / 2.0);
qne0(3) = sin(-pi / 4.0 - lla0(1) / 2.0) * cos(lla0(2) / 2.0);
qne0(4) = cos(-pi / 4.0 - lla0(1) / 2.0) * sin(lla0(2) / 2.0);

qneo_inv = quatinv(qne0);

% TM- This resulting quaternion is the product of the inverse of the
%     quaternion determined above, and the initial attitude quaternion from
%     the beginning of the program
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

lla_m = quatToLla(qne_m);
lla_m(3) = (lla0(3) + lla(3)) / 2.0;

% recompute zeta, xi
phi = gy+1/12*cross(gy0, gy);
qbb = make_quaternion(phi, 0.5);

[zeta, ~, ~] = get_zeta(dt, omega_e, lla_m, vel_m, WGS84_A, e, 0);

qnn = make_quaternion(zeta, -0.5);

tmp_q = quatmultiply(qbn0, qbb);
qbn = quatmultiply(qnn, tmp_q);

% normalization
e_q = sumsqr(qbn);
e_q = 1 - 0.5 * (e_q - 1);
qbn = e_q.*qbn;

rpy = flip(quat2eul(qbn));

end


function [zeta, wie, wen] = get_zeta(dt, omega_e, lla, vel, WGS84_A, e, first)
% Compute the wie, wen, and zeta
rn = WGS84_A / sqrt(1 - e * e * sin(lla(1)) * sin(lla(1)));
rm = WGS84_A * (1 - e * e) / sqrt(power(1 - e * e * sin(lla(1)) * sin(lla(1)), 3));

% compute wie
wie(1) = omega_e * cos(lla(1));
wie(2) = 0;
wie(3) = -omega_e * sin(lla(1));

% compute wen
wen(1) = vel(2) / (rn + lla(3));
wen(2) = -vel(1) / (rm + lla(3));
wen(3) = -vel(2) * tan(lla(1)) / (rn + lla(3));
win = wie + wen;

%disp(class(dt))
%disp(class(win))

if first == 1
    zeta = dt.*win/2.0;
else
    zeta = win.*dt;
end

end

function quat = make_quaternion(var, factor)
% Turn a value (zeta, phi, xi) into a quaternion
half_var = factor*var; 
mold = norm(half_var);
sc = sin(mold) / mold;

quat = zeros(1,4);
quat(1) = cos(mold);
quat(2) = sc * half_var(1);
quat(3) = sc * half_var(2);
quat(4) = sc * half_var(3);

end

function gl = get_gl(lla0, lla)
% compute g_l
grav = [9.7803267715, 0.0052790414, 0.0000232718, -0.000003087691089, 0.000000004397731, 0.000000000000721];
gl = zeros(1,3);
sinB = sin(lla(1));
sinB2 = sinB * sinB;
sinB4 = sinB2 * sinB2;
gl(3) = grav(1) * (1.0 + grav(2) * sinB2 + grav(3) * sinB4) + (grav(4) + grav(5) * sinB2) * lla0(3) + grav(6) * lla0(3) * lla0(3);
end

function lla = quatToLla(quat)
% Create a 
lla = zeros(1,3);

if quat(1) ~= 0
    lla(2) = 2 * atan(quat(4) / quat(1));
    lla(1) = 2 * (-pi / 4.0 - atan(quat(3) / quat(1)));
elseif quat(1) == 0 && quat(3) == 0
    lla(2) = pi;
    lla(1) = 2 * (-pi / 4.0 - atan(-quat(2) / quat(4)));
elseif quat(1) == 0 && quat(4) == 0
    lla(2) = 2 * atan(-quat(2) / quat(3));
    lla(1) = pi / 2.0;
end
end


% ins_mechanization_compact.m:
function [lla, vel, rpy, ned, dv, qbn, xHatP, PP, gpsUpdated, gpsCnt, gps_res] = ins_gps(lla0, vel0, rpy0, dv0, qbn0, msrk, msrk1, gpsmsr, xHatP, PP, gpsCnt, gps_res)
% TM- As far as I can tell right now, the code from here to the comment
%     "Phi-Error Model" is exactly the same as the last function without GPS 

WGS84_A = 6378137.0;           % earth semi-major axis (WGS84) (m) 
WGS84_B = 6356752.3142;        % earth semi-minor axis (WGS84) (m) 
e = sqrt(WGS84_A * WGS84_A - WGS84_B * WGS84_B) / WGS84_A;

% rotational angular velocity of earth
omega_e = 7.2921151467e-5;   

dt = msrk1(1) - msrk(1);

gy = msrk1(2:4);
ac = msrk1(5:7);

gl = get_gl(lla0, lla0);

[lla, vel, rpy, dv, qbn, win] = ins_mechanization_compact(lla0, vel0, rpy0, dv0, qbn0, msrk, msrk1);

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


% TM- All the matricies starting with A are used to create the ultimate A
%     matrix below
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

% TM- This A matrix is used to calculate the F matrix, which includes the
%     coefficients of the error states of delta x
A = [Arr, Arv, zeros(3,3), zeros(3,3), zeros(3,3), zeros(3,3), zeros(3,3); ...
    Avr, Avv, fn_cross, cbn, zeros(3,3), cbn*f_diag, zeros(3,3); ...
    Aer, Aev, -win_cross, zeros(3,3), -cbn, zeros(3,3), -cbn*w_diag; ...
    zeros(3,3), zeros(3,3), zeros(3,3), diag([-1.0/Tba,-1.0/Tba, -1.0/Tba]), zeros(3,3), zeros(3,3), zeros(3,3); ...
    zeros(3,3),zeros(3,3),zeros(3,3),zeros(3,3), diag([-1.0/Tbg,-1.0/Tbg, -1.0/Tbg]), zeros(3,3), zeros(3,3); ...
    zeros(3,3), zeros(3,3), zeros(3,3), zeros(3,3), zeros(3,3), diag([-1.0/Tsa,-1.0/Tsa, -1.0/Tsa]), zeros(3,3); ...
    zeros(3,3), zeros(3,3), zeros(3,3), zeros(3,3), zeros(3,3), zeros(3,3), diag([-1.0/Tsg,-1.0/Tsg, -1.0/Tsg])];


% TM- The following calculations, up until Qk = ... correspond 
%     to Shin section 3.3.1- Discrete Time System, under the heading 
%     The Extended Kalman Filter Design


% TM- B here is written as G in the thesis, the noise mapping input matrix
%     at time tk and is used to calculate Qk below
B = zeros(21,18);
B(4:6,1:3) = cbn;
B(7:9,4:6) = -cbn;
B(10:12,7:9) = eye(3);
B(13:15,10:12) = eye(3);
B(16:18,13:15) = eye(3);
B(19:21,16:18) = eye(3);

% TM- This is the uppercase Phi_k, the state transition matrix
%     at time tk, used in the Qk calculation
F = eye(21)+A*dt;

qg = (deg2rad(0.2))^2;
qa = 0.05^2;
qbg = 2*(deg2rad(0.0022)/60)^2/Tbg;
qba = 2*(0.00075/60)^2/Tba;
qsg = 2*(10*10^-6)^2/Tsg;
qsa = 2*(10*10^-6)^2/Tsa;
% TM- This Q is the spectral density matrix at time tk
Q = zeros(18,18);
Q(1,1) = qa; Q(2,2) = qa; Q(3,3) = qa; 
Q(4,4) = qg; Q(5,5) = qg; Q(6,6) = qg;
Q(7,7) = qba; Q(8,8) = qba; Q(9,9) = qba; 
Q(10,10) = qbg; Q(11,11) = qbg; Q(12,12) = qbg;
Q(13,13) = qsa; Q(14,14) = qsa; Q(15,15) = qsa; 
Q(16,16) = qsg; Q(17,17) = qsg; Q(18,18) = qsg;
Qk = 0.5*(F*B*Q*B'+B*Q*B'*F')*dt;
% TM- This corresponds to Shin 3.47, both equations
%     involve Qk and 1/2, but the other letters are slightly different:
%     Phi, G, and T in the thesis are F, B, and prime here 

% TM- The next two lines of code correspond to Shin 3.77a and 3.77b
%     This means that F (like above) is Phi_(k-1), and 
%     xHatM is delta x_(k|k-1), and xHatP is delta x_(k-1|k-1)

% TM- These lines correspond to equations (23) and (24) in the Rife thesis
%     An M (xHatM, PM) represents a superscript minus and the subscript 
%     k+1 in the text, indicating the result of the current prediction 
%     step. The P (xHatP, PP) represents the superscript minus and the 
%     subscript k, indicating the result of the previous step's corrected 
%     calculation value

% TM- This next equation is under Shin 3.3.3- Linearized Kalman Filter,
%     and is the estimate and error covariance projections of the time
%     update stage

% Extended Kalman Filter 
xHatM = F*xHatP;                    % Forward Euler integration (A priori)
PM = F*PP*F'+Qk;                    % cov. Estimate

rn = WGS84_A / sqrt(1 - e * e * sin(lla_in(1)) * sin(lla_in(1)));
rm = WGS84_A * (1 - e * e) / sqrt(power(1 - e * e * sin(lla_in(1)) * sin(lla_in(1)), 3));
coeff = diag([(rm+lla_in(3)); ((rn+lla_in(3))*cos(lla_in(1))); -1]);

% update with GPS measurements
% TM- This is still Shin 3.3.3, and is the measurement update
if gpstime <= msrk1(1) && gpstime > msrk(1)
    
    R = 0.01*diag(gpsmsr(5:7).^2);
    
    dr = [0.571210;0.223760;-0.270500];
    lla_gps_corr = coeff*(gpsmsr(2:4))' - cbn*dr;

    y = coeff*lla_in' - lla_gps_corr;
    
    gps_res = [gps_res; gpstime, lla_gps_corr'];
    
    % TM- H here is the matrix H_k constructed in Rife (27)
    H = zeros(3,21);
    H(1:3, 1:3) = coeff;
    
    % TM- The next chunk of lines correspond to more of Shin 3.3.3
    %     These are the big calculations from the measurement update
    
    % TM- This is a combination of Shin 3.78a and 3.78b
    %     (H*PM*H'+R) is 3.78a, and 3.78b takes that inverse, and 
    %     multiplies it by PM*H'
    %     This line directly corresponds to Rife (28)
    L = PM*H'*inv(H*PM*H'+R);
    
    % TM- This sets up the next equaton, Shin 3.78c
    %     This is the same as Rife (26), but without adding sensor noise
    yHat = H*xHatM;
    
    % TM- This line is Shin 3.78c, solving for delta xHat_(k|k-1)
    %     This is also Rife (29)
    xHatP = xHatM + L*(y-yHat);         % a  posteriori estimate
    
    % TM- This is Shin 3.78d and Rife (30)
    PP = (eye(size(F))-L*H)*PM*(eye(size(F))-L*H)'+L*R*L';
    
    % TM- VARIABLE TRANSLATIONS: 
    %   Code   ->   Thesis     ->      Meaning
    %  ----------------------------------------
    %    L        K_k                  Kalman gain        
    %    PM       P_(k|k-1)            Covariance of predicted estimate minus   
    %    H        H_k                  Relates sensor outputs to states in a linearized sense
    %    R        R_k                  Measurement noise covariance matrix
    %   xHatM     delta xHat_(k|k-1)   Predicted estimate minus
    %   yHat      H_k * xHatM          Intermediate step of 3.78c
    %    y        delta z_k            Vector measurement
    %    PP       P_(k|k)              Updated solution plus
    %    I        eye(size)            Identity matrix 
    %    F        Phi_k                State transition matrix
    
    % NOTE: A prime indicates a superscript of T in the thesis
    %       Ex: H', L', etc
    %
    %       Additionally, an M or P at the end of a variable name indicates
    %       a plus or minus. A minus is the result of the previous 
    %       prediction step, while a plus is the result of the current 
    %       correction step
    
    
    lla = lla - (xHatP(1:3))';
 
    xi = xHatP(7:9);
    E = [0 -xi(3) xi(2); xi(3) 0 -xi(1); -xi(2) xi(1) 0];
    cbn = (eye(3)+E)*cbn;
    
    vel = vel - (xHatP(4:6))';
    
    qbn = dcm2quat(cbn');
    
    % normalization
    e_q = sumsqr(qbn);
    e_q = 1 - 0.5 * (e_q - 1);
    qbn = e_q.*qbn;
    
    % flip the sign if two quaternions are opposite in sign
    if (qbn(1)*qbn0(1)<0)
        qbn = -qbn;
    end
    
    rpy = flip(quat2eul(qbn));
    
    dv = vel - vel0;
    
    gpsUpdated = 1;
    gpsCnt = gpsCnt + 1;
else
    % TM- At the end of the calculations, rename the final (+) values of  
    %     the corrected steps as the starting (-) values for the next 
    %     iteration's prediction step
    xHatP = xHatM;
    PP = PM;
    
    gpsUpdated = 0;
end

    ned = (coeff*lla')';

end