%use this script to create a figure to demonstrate the error between GPS
%ground truth (smoothed via cubic spline) and LIDAR estiates over time. 

%key variables:
%   interp_x2 -> interpolated lidar data with NON-CONSTANT TIME STEP, size = (2851, 1), t-5
%   x_gps -> GPS data in m, size = (1,524), t_h

close all
method = 'linear';

% % old way -> change Lidar to GPS time frame -------------------------------
% % interpolate _gps to get the same number of data points as interp_x2
% xspangps = linspace(1,max(size(x_gps)),max(size(interp_x2)));
% xgpsi = interp1(x_gps, xspangps, method);
% xgpsi(2602:end) = []; %cut short to keep them the same length
% 
% %interpolate interp_x2 to get constant time steps
% % xspanlidar = 1/(t - t(1));%* 2850 / 564;
% xspanlidar = 1:max(size(interp_x2));
% xlidar = interp1(t - t(1), interp_x2, xspanlidar, method);
% xlidar(1:45) = [];
% %bug is here?
% xlidar =interp1(linspace(1,max(size(xlidar)),max(size(xlidar))),xlidar, linspace(1,564,max(size(interp_x2))), method);
% % xlidar =interp1(1:max(size(xlidar(isnan(xlidar) == 0))), xlidar, linspace(1,564,max(size(interp_x2))), method);
% % xlidar(1:250) = [];
% xlidar = xlidar + 0.4;


%New way -> change GPS to Lidar time frame ------------------------------
cutoff = 200;
xlidar = interp_x2(cutoff:2720); %make copy so we don't mess up the OG var
xgpsi = x_gps(1:495);
xspangps = t - t(cutoff);
%upscale x_gps to same number of points as lidar
xgpsi = interp1(xgpsi, 1:max(size(xgpsi)), method);
%move into lidar timeframe
xgpsi = interp1(xgpsi, xspangps, method);
xgpsi(1:cutoff) = []; %get rid of NaNs

ylidar = interp_y2(cutoff:2720); %make copy so we don't mess up the OG var
ygpsi = y_gps(1:495);
yspangps = t - t(cutoff);
%upscale y_gps to same number of points as lidar
ygpsi = interp1(ygpsi, 1:max(size(ygpsi)), method);
%move into lidar timeframe
ygpsi = interp1(ygpsi, yspangps, method);
ygpsi(1:cutoff) = []; %get rid of NaNs

yawcutoff = 200;
yawlidar = interp_yaw2(yawcutoff:2720); %make copy so we don't mess up the OG var
yawgpsi = -h(7:495);
yawspangps = t - t(yawcutoff);
%upscale y_gps to same number of points as lidar
yawgpsi = interp1(yawgpsi, 1:max(size(yawgpsi)), method);
%move into lidar timeframe
yawgpsi = interp1(yawgpsi, yspangps, method);
yawgpsi(1:yawcutoff) = []; %get rid of NaNs



lidar_error_x = [];
lidar_error_y = [];
lidar_error_yaw = [];
dxlidar = [];
dxgps = [];
dylidar = [];
dygps = [];
dyawlidar = [];
dyawgps = [];
step_size = 1;
for ct = 1:step_size:(max(size(xlidar ))-step_size)
    
    if ct == 1
        last = 0;
    else
        last = lidar_error_x(end);
    end
    
    dlidarx = (xlidar(ct+step_size) - xlidar(ct)) - (xgpsi(ct + step_size) - xgpsi(ct));
    dxlidar = [dxlidar, xlidar(ct+step_size) - xlidar(ct)];
    dxgps = [dxgps, xgpsi(ct + step_size) - xgpsi(ct)];
    lidar_error_x = [lidar_error_x, dlidarx];

    dlidary = (ylidar(ct+step_size) - ylidar(ct)) - (ygpsi(ct + step_size) - ygpsi(ct));
    dylidar = [dylidar, ylidar(ct+step_size) - ylidar(ct)];
    dygps = [dygps, ygpsi(ct + step_size) - ygpsi(ct)];
    lidar_error_y = [lidar_error_y, dlidary];
    
    dlidaryaw = (yawlidar(ct+step_size) - yawlidar(ct)) - (yawgpsi(ct + step_size) - yawgpsi(ct));
    dyawlidar = [dyawlidar, yawlidar(ct+step_size) - yawlidar(ct)];
    dyawgps = [dyawgps, yawgpsi(ct + step_size) - yawgpsi(ct)];
    lidar_error_yaw = [lidar_error_yaw, dlidaryaw];

end

lidar_error_x = rmmissing(lidar_error_x);
lidar_error_y = rmmissing(lidar_error_y);
lidar_error_yaw = rmmissing(lidar_error_yaw);

fig0 = figure();
fig0.Color = 'white';
hold on
plot(t(1:length(dxlidar)) - t(1), dxlidar,'Marker', '.', 'MarkerSize', 5)
plot(t(1:length(dxgps)) - t(1),dxgps,'Marker', '.', 'MarkerSize', 5)
title("Change in x position per time step")
legend('Lidar', 'GPS (interpolated)')
ylabel("Displacement (m)")
xlabel("time (s)")
hold off

fig10 = figure();
fig10.Color = 'white';
hold on
plot(t(1:length(dylidar)) - t(1), dylidar,'Marker', '.', 'MarkerSize', 5)
plot(t(1:length(dygps)) - t(1),dygps,'Marker', '.', 'MarkerSize', 5)
title("Change in y position per time step")
legend('Lidar', 'GPS (interpolated)')
ylabel("Displacement (m)")
xlabel("time (s)")
hold off

fig11 = figure();
fig11.Color = 'white';
hold on
plot(t(1:length(dyawlidar)) - t(1), dyawlidar,'Marker', '.', 'MarkerSize', 5)
plot(t(1:length(dyawgps)) - t(1),dyawgps,'Marker', '.', 'MarkerSize', 5)
title("Change in yaw per time step")
legend('Lidar', 'GPS (interpolated)')
ylabel("Displacement (deg)")
xlabel("time (s)")
hold off

% fig1 = figure();
% hold on
% plot(1:step_size:(max(size(xlidar))-step_size),lidar_error_x)
% title("Lidar drift per timestep")
% ylabel("position error (m)")
% xlabel("time (s)")
% hold off
% 
% fig1 = figure()
% fig1.Color = 'white'
% hold on
% plot(x_gps_m , y_gps_m );
% plot(pos_x_lidar, pos_y_lidar)
% plot(interp_x2, interp_y2)
% legend('GPS','Lidar')
% ylabel('x (m)')
% xlabel('y (m)')
% hold off

fig2 = figure();
fig2.Color = 'white';
hold on
% plot(interp_x2)
plot(t(1:length(xlidar)) - t(1), xlidar,'Marker', '.', 'MarkerSize', 5)
% plot(t_h,x_gps)
plot(t(1:length(xgpsi)) - t(1),xgpsi,'Marker', '.', 'MarkerSize', 5)
title("Accumulated x position")
legend('Lidar', 'GPS (interpolated)')
ylabel("Position (m)")
xlabel("time (s)")
hold off

fig3 = figure();
fig3.Color = 'white';
hold on
plot(t(1:length(ylidar)) - t(1), ylidar,'Marker', '.', 'MarkerSize', 5)
plot(t(1:length(ygpsi)) - t(1),ygpsi,'Marker', '.', 'MarkerSize', 5)
title("Accumulated y position")
legend('Lidar', 'GPS (interpolated)')
ylabel("Position (m)")
xlabel("time (s)")
hold off

fig5 = figure();
fig5.Color = 'white';
hold on
plot(t(1:length(yawlidar)) - t(1), yawlidar,'Marker', '.', 'MarkerSize', 5)
plot(t(1:length(yawgpsi)) - t(1),yawgpsi,'Marker', '.', 'MarkerSize', 5)
title("Accumulated yaw") 
legend('Lidar', 'GPS (interpolated)')
ylabel("Rotation (deg)")
xlabel("time (s)")
hold off


% %main fig------------------------------------------------------------------
fig4 = figure();
fig4.Color = 'white';
subplot(3,1,1)
% plot(t - t(1),dx);
% plot(dx);
% scatter(1:step_size:(max(size(yawlidar))-step_size),lidar_error_x, 2)
scatter(t(1:length(lidar_error_x)) - t(1),lidar_error_x, 2)
xSigma = std(lidar_error_x,'omitnan');
text(10,0.25,['\sigma = ' num2str(xSigma,'%.3f')]);
title('Pose Change in One Time Step for LIDAR as compared to GPS/INS')
hold on; grid on;
ylabel('Latitude Error (meters)');
xlim([0 550]);

subplot(3,1,2)
% plot(t - t(1),dy);
% scatter(1:step_size:(max(size(yawlidar))-step_size),lidar_error_y, 2)
scatter(t(1:length(lidar_error_y)) - t(1),lidar_error_y, 2)
ySigma = std(lidar_error_y,'omitnan');
text(10,0.25,['\sigma = ' num2str(ySigma,'%.3f')]);
hold on; grid on;
ylabel('Longitude Error (meters)');
xlim([0 550]);

subplot(3,1,3)
% plot(t - t(1), dyaw)
% scatter(1:step_size:(max(size(yawlidar))-step_size),lidar_error_yaw, 2)
scatter(t(1:length(lidar_error_yaw)) - t(1),lidar_error_yaw, 2)
yawSigma = std(lidar_error_yaw);
text(10,0.6,['\sigma = ' num2str(zSigma,'%.3f')]);
hold on; grid on;
ylabel('Azimuth Error (degrees)');
xlabel('time (s)')
xlim([0 550]);

 dxlidarmat = [t(1:length(dxlidar)) - t(1), dxlidar'];
 dxgpsmat = [t(1:length(dxgps)) - t(1), dxgps'];
 dylidarmat = [t(1:length(dylidar)) - t(1), dylidar'];
 dygpsmat = [t(1:length(dygps)) - t(1), dygps'];
 dyawlidarmat = [t(1:length(dyawlidar)) - t(1), dyawlidar'];
 dyawgpsmat = [t(1:length(dyawgps)) - t(1), dyawgps'];
 
 xlidarmat = [t(1:length(xlidar)) - t(1), xlidar];
 xgpsmat = [t(1:length(xgpsi)) - t(1), xgpsi];
 ylidarmat = [t(1:length(ylidar)) - t(1), ylidar];
 ygpsmat = [t(1:length(ygpsi)) - t(1), ygpsi];
 yawlidarmat = [t(1:length(yawlidar)) - t(1), yawlidar];
 yawgpsmat = [t(1:length(yawgpsi)) - t(1), yawgpsi];
  
 
 save volpe.mat dxlidarmat dxgpsmat dylidarmat dygpsmat dyawlidarmat dyawgpsmat xlidarmat xgpsmat ylidarmat ygpsmat yawlidarmat yawgpsmat
 
