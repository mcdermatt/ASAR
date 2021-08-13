%file used to generate R matrix from GPS Data

%import data file
filename = '2021-03-10-16-43-50_Velodyne-VLP-16-Data_garminSignage-position.csv';
% opts = detectImportOptions(filename);
data = readmatrix(filename);
lat = data(:,1);
lon = data(:,2);
t = data(:,4);
%"lat","lon","gpstime","time","accel1x","accel1y","accel2x","accel2y",
%"accel3x","accel3y","gyro1","gyro2","gyro3","heading","temp1","temp2",
%"temp3","Points:0","Points:1","Points:2"

%plot lat -----------------------------------------------------------------
figure()
hold on
%remove zeros
% lat(lat == 0) = [];
idx = find(lat == 0);
lat(idx) = [];
t(idx) = [];

%flip latitdue upside down (to match positive y for now...)
lat = -lat;
plot(t,lat)
title('lat');
hold off
 
%plot lon------------------------------------------------------------------
figure()
hold on
lon(lon == 0) = [];
lon = -lon;

plot(lon)
title('lon')
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