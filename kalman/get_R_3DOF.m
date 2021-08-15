%file used to generate R matrix from GPS Data

% %DIRECT FROM GPS DATA -----------------------------------------------------
% %import data file
% filename = '2021-03-10-16-43-50_Velodyne-VLP-16-Data_garminSignage-position.csv';
% % opts = detectImportOptions(filename);
% data = readmatrix(filename);
% lat = data(:,1);
% lon = data(:,2);
% h = data(:,14);
% t = data(:,4);
% %"lat","lon","gpstime","time","accel1x","accel1y","accel2x","accel2y",
% %"accel3x","accel3y","gyro1","gyro2","gyro3","heading","temp1","temp2",
% %"temp3","Points:0","Points:1","Points:2"
% %--------------------------------------------------------------------------

%FROM BESTPOS--------------------------------------------------------------
lon = x_gps;
lat = y_gps;
% t_bp = nxrx; %broken
t_bp = 1:max(size(x_gps));
%Need to get h from kalman demo script...
%--------------------------------------------------------------------------
interpts_GPS = 110;
method_GPS = 'cubic';

%plot lat -----------------------------------------------------------------
figure()
hold on

latq = linspace(1,max(size(lat)),interpts_GPS);
interp_lat = interp1(lat, latq, method_GPS);

latq2 = linspace(1,interpts_GPS, max(size(lat)));
interp_lat2 = interp1(interp_lat, latq2, method_GPS);
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
 
%plot lon------------------------------------------------------------------
figure()
hold on

lonq = linspace(1,max(size(lon)),interpts_GPS);
interp_lon = interp1(lon, lonq, method_GPS);

lonq2 = linspace(1,interpts_GPS, max(size(lon)));
interp_lon2 = interp1(interp_lon, lonq2, method_GPS); 

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

headingq = linspace(1,max(size(h)),interpts_GPS);
interp_heading = interp1(h, headingq, method_GPS);

headingq2 = linspace(1,interpts_GPS, max(size(h)));
interp_heading2 = interp1(interp_heading, headingq2, method_GPS); 

% plot(t_bp, interp_heading2); %not working...
plot(h);
plot(interp_heading2)
legend('actual','interpolated')
title('heading')
xlabel('timestep')
ylabel('heading (deg)')

hold off
%plot time-----------------------------------------------------------------
% figure()
% hold on
% %remove zeros
% % t(t < 1e6) = [];
% 
% plot(t);
% 
% hold off

% % put everything together to get final R matrix----------------------------
figure()
hold on
plot(lon, lat);
plot(interp_lon2, interp_lat2);
legend('actual','interpolated')
title('Path (GPS data)')
xlabel('lon (deg)')
ylabel('lat (deg)')
hold off

dlat = lat - interp_lat2.';
dlon = lon - interp_lon2.';
dheading = h - interp_heading2.';

%in m already
std_lat = std(dlat);
std_lon = std(dlon);
std_heading = std(dheading);


std_heading = deg2rad(std_heading)

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