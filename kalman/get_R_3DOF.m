%file used to generate R matrix from GPS Data

% % %DIRECT FROM GPS DATA -----------------------------------------------------
%import data file
filename = '2021-03-10-16-43-50_Velodyne-VLP-16-Data_garminSignage-position.csv';
% opts = detectImportOptions(filename);
data = readmatrix(filename);
lat = -data(:,1);
lon = -data(:,2);
h = -data(:,14);
t_bp = data(:,4);
%"lat","lon","gpstime","time","accel1x","accel1y","accel2x","accel2y",
%"accel3x","accel3y","gyro1","gyro2","gyro3","heading","temp1","temp2",
%"temp3","Points:0","Points:1","Points:2"

timeRef = t_bp(1); % units sec 
timeVec = t_bp;    % units sec

%get idx where data is zero
idx = find(lon);
lat = lat(idx);
lon = lon(idx);
h = h(idx);
t_bp = t_bp(idx);

%convert to ENU
gps_enu = Wgslla2enu(bplat, bplon, bphgt, bplat(1), bplon(1), bphgt(1));


%adjust h for continuous rotation
h(444:end) = h(444:end) + 180;
h(4395:end) = h(4395:end) + 360; 
h(5160:end) = h(5160:end) - 360;

%fix timescale in h

%DEBUG: apply moving average
% h = movmean(h, 25);
% lat = movmean(lat, 2);
% lon = movmean(lon, 2);
% % 
% % % %--------------------------------------------------------------------------

%FROM BESTPOS--------------------------------------------------------------
% lon = x_gps.';
% lat = y_gps.';
% % t_bp = nxrx; %broken
% t_bp = 1:max(size(x_gps));
% %Need to get h from kalman demo script...
% 
% timeRef = t(1); % units sec 
% timeVec = t;    % units sec
% -------------------------------------------------------------------------


% % %Matt's old Interpolation Method ------------------------------------------
% interpts_GPS = 300;
% method_GPS = 'cubic';
% 
% 
% latq = linspace(1,max(size(lat)),interpts_GPS);
% interp_lat = interp1(lat, latq, method_GPS);
% latq2 = linspace(1,interpts_GPS, max(size(lat)));
% interp_lat2 = interp1(interp_lat, latq2, method_GPS);
% 
% lonq = linspace(1,max(size(lon)),interpts_GPS);
% interp_lon = interp1(lon, lonq, method_GPS);
% lonq2 = linspace(1,interpts_GPS, max(size(lon)));
% interp_lon2 = interp1(interp_lon, lonq2, method_GPS); 
% 
% headingq = linspace(1,max(size(h)),interpts_GPS);
% interp_heading = interp1(h, headingq, method_GPS);
% 
% headingq2 = linspace(1,interpts_GPS, max(size(h)));
% interp_heading2 = interp1(interp_heading, headingq2, method_GPS); 
% 
% interp_lon2 = interp_lon2.'; %for debug
% interp_lat2 = interp_lat2.'; %for debug
% interp_heading2 = interp_heading2.'; %for debug
% 
% % %--------------------------------------------------------------------------


%Jason's Cubic fit method -----------------------------
% Build window
windowHalfWidth = 6;  % User defined parameter
windowFullWidth = 2*windowHalfWidth+1;
% Re-name data read by read0em7memmap_update.m
timeIntoExperiment = timeVec-timeRef;

% Initialize fit
interp_lon2 = zeros(size(lon))*NaN;  % Initialize fit to values of "Not a Number"
dlon = interp_lon2;

interp_lat2 = zeros(size(lat))*NaN;  % Initialize fit to values of "Not a Number"
dlat = interp_lat2;

interp_heading2 = zeros(size(h))*NaN;  % Initialize fit to values of "Not a Number"
dheading = interp_heading2;

% Cycle through each time
lengthData = length(lon);   
for indx = windowHalfWidth+1:lengthData-windowHalfWidth % scan over window center points
    windowedData = lon(indx-windowHalfWidth:indx+windowHalfWidth);
    centerData = lon(indx);
    tCenter = timeIntoExperiment(indx);
    t_window = timeIntoExperiment(indx-windowHalfWidth:indx+windowHalfWidth)-tCenter;
    oneVec = ones(windowFullWidth,1);
    A = [oneVec t_window t_window.^2 t_window.^3]; % If sample rate is consistent, then this matrix is constant; 
                             %  but computing this matrix at each time step
                             %  allows us to deal with jumps in the sample
                             %  time
    coef = A\windowedData; % Compute cubic coefficients
    interp_lon2(indx) = A(windowHalfWidth+1,:)*coef;
%     dx(indx) = centerData-interp_x(indx); % subtract fit from the data at the center point

    windowedData = lat(indx-windowHalfWidth:indx+windowHalfWidth);
    centerData = lat(indx);
    coef = A\windowedData; % Compute cubic coefficients
    interp_lat2(indx) = A(windowHalfWidth+1,:)*coef;

    windowedData = h(indx-windowHalfWidth:indx+windowHalfWidth);
    centerData = h(indx);
    coef = A\windowedData; % Compute cubic coefficients
    interp_heading2(indx) = A(windowHalfWidth+1,:)*coef;

end
%get rid of NaN values at beginning and end of interp
interp_lon2(1:windowHalfWidth) = lon(1:windowHalfWidth);
interp_lon2(end-windowHalfWidth:end) = lon(end-windowHalfWidth:end);
dx = lon - interp_lon2;

interp_lat2(1:windowHalfWidth) = lat(1:windowHalfWidth);
interp_lat2(end-windowHalfWidth:end) = lat(end-windowHalfWidth:end);
dy = lat - interp_lat2;

interp_heading2(1:windowHalfWidth) = h(1:windowHalfWidth);
interp_heading2(end-windowHalfWidth:end) = h(end-windowHalfWidth:end);
dheading = h - interp_heading2;

%------------------------------------------------------



%plot lat -----------------------------------------------------------------
figure()
hold on
% interp_lat2 = interp_lat2.';
plot(t_bp,lat)
%interp_lat2 and lat are in different time scales
%this results in data that is horizontally shifted (not good)
% t_bp = linspace(min(t),max(t),max(size(lat))); 
plot(t_bp, interp_lat2);

legend('actual','interpolated')
title('lat');
xlabel('timestep')
ylabel('lat (m)')
hold off
%  
%plot lon------------------------------------------------------------------
figure()
hold on
plot(t_bp,lon)
plot(t_bp, interp_lon2);
legend('actual','interpolated')
title('lon')
xlabel('timestep')
ylabel('lon (m)')

hold off

% plot heading ------------------------------------------------------------
figure()
hold on
% plot(t_bp, interp_heading2); %not working...
plot(t_bp, h);
plot(t_bp, interp_heading2)
legend('actual','interpolated')
title('heading')
xlabel('timestep')
ylabel('heading (deg)')
hold off

% % put everything together to get final R matrix----------------------------
% figure()
% hold on
% plot(lon, lat);
% plot(interp_lon2, interp_lat2);
% legend('actual','interpolated')
% title('Path (GPS data)')
% xlabel('lon (deg)')
% ylabel('lat (deg)')
% hold off

dlat = lat - interp_lat2;
dlon = lon - interp_lon2;
dheading = h - interp_heading2;

%in m already if using lat/ lon == x_gps/ y_gps from kalman_demo file
std_lat = std(dlat)
std_lon = std(dlon)
std_heading = std(dheading)

% % std_heading = deg2rad(std_heading)
% %plot heading error -------------------------------------------------------
figure()
hold on
plot(dheading)
title("heading error: mean = " + num2str(mean(dheading)) +  " std heading = " + num2str(std_heading))
xlabel("timestep (s)")
ylabel("error (deg)")
hold off

R = zeros(3);
R(1,1) = std_lon^2;
R(2,2) = std_lat^2;
R(3,3) = std_heading^2;
%get non-diagonal elements
cov_xy = cov(dlon, dlat); %should this be in car frame or world frame?
R(1,2) = cov_xy(1,2);
R(2,1) = cov_xy(1,2);
cov_xheading = cov(dlon, dheading);
R(1,3) = cov_xheading(1,2);
R(3,1) = cov_xheading(1,2);
cov_yheading = cov(dlat,dheading);
R(2,3) = cov_yheading(1,2);
R(3,2) = cov_yheading(1,2);

R