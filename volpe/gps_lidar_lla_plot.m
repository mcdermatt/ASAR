
%verify that GPS and lidar are of the same scale and sign
figure(1)
hold on
plot(lon_lidar, lat_lidar)
plot(ppplon, ppplat)


figure()
hold on
plot(hgt_lidar) %something is off here...
% plot(pos_lidar_enu(:,3))
plot(ppphgt)

%use this to check alignment of times between lidar and gps data
startGPS = 1109; %cut off every GPS point before here
figure()
hold on
plot([0 t_lidar.'], lon_lidar)
gps_time = ppptime - ppptime(1);
plot(gps_time(startGPS:end) - startGPS , ppplon(startGPS:end))


%verify that GPS data is being interpolated correctly
figure()
hold on
plot(t_lidar, lon_lidar(1:end-1))
plot(t_lidar, lon_gps_lidartime)