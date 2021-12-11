
%verify that GPS and lidar are of the same scale and sign
figure(1)
hold on
plot(lon_lidar, lat_lidar)
plot(ppplon, ppplat)


figure()
hold on

% plot(t_lidar, hgt_lidar(2:end)) %something is off here...
%try rescaling to cm...
test = (hgt_lidar(2:end) - hgt_lidar(2))*-0.1 + hgt_lidar(2);
plot(t_lidar, test);
plot(t_lidar, hgt_gps_lidartime)

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
plot(ppplon(startGPS-3:end))

figure()
plot(ppplon)