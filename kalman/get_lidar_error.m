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


yawlidar = interp_yaw2(cutoff:2720); %make copy so we don't mess up the OG var
yawgpsi = -h(1:495);
yawspangps = t - t(cutoff);
%upscale y_gps to same number of points as lidar
yawgpsi = interp1(yawgpsi, 1:max(size(yawgpsi)), method);
%move into lidar timeframe
yawgpsi = interp1(yawgpsi, yspangps, method);
yawgpsi(1:cutoff) = []; %get rid of NaNs



lidar_error_x = [];
lidar_error_y = [];
lidar_error_yaw = [];
step_size = 1;
for ct = 1:step_size:(max(size(xlidar ))-step_size)
    
    if ct == 1
        last = 0;
    else
        last = lidar_error_x(end);
    end
    
    dlidarx = (xlidar(ct+step_size) - xlidar(ct)) - (xgpsi(ct + step_size) - xgpsi(ct));
    lidar_error_x = [lidar_error_x, dlidarx];

    dlidary = (ylidar(ct+step_size) - ylidar(ct)) - (ygpsi(ct + step_size) - ygpsi(ct));
    lidar_error_y = [lidar_error_y, dlidary];
    
    dlidaryaw = (yawlidar(ct+step_size) - yawlidar(ct)) - (yawgpsi(ct + step_size) - yawgpsi(ct));
    lidar_error_yaw = [lidar_error_yaw, dlidaryaw];

end

% fig1 = figure();
% hold on
% plot(1:step_size:(max(size(xlidar))-step_size),lidar_error_x)
% title("Lidar drift per timestep")
% ylabel("position error (m)")
% xlabel("time (s)")
% hold off

fig1 = figure()
fig1.Color = 'white'
hold on
plot(x_gps_m , y_gps_m );
plot(pos_x_lidar, pos_y_lidar)
plot(interp_x2, interp_y2)
legend('GPS','Lidar')
ylabel('x (m)')
xlabel('y (m)')
hold off

fig2 = figure();
hold on
% plot(interp_x2)
plot(xlidar)
% plot(t_h,x_gps)
plot(xgpsi)
title("x trajectory")
legend('Lidar, origonal timescale', 'GPS, in lidar timescale')
ylabel("position (m)")
xlabel("time (s)")
hold off

fig3 = figure();
hold on
plot(ylidar)
plot(ygpsi)
title("y trajectory")
legend('Lidar, origonal timescale', 'GPS, in lidar timescale')
ylabel("position (m)")
xlabel("time (s)")
hold off

fig5 = figure();
hold on
plot(yawlidar)
plot(yawgpsi)
title("yaw")
legend('Lidar, origonal timescale', 'GPS, in lidar timescale')
ylabel("rotation (deg)")
xlabel("time (s)")
hold off


% %main fig------------------------------------------------------------------
fig4 = figure();
fig4.Color = 'white';
subplot(3,1,1)
% plot(t - t(1),dx);

% plot(dx);
plot(1:step_size:(max(size(xlidar))-step_size),lidar_error_x)
% xSigma = std(dx,'omitnan');
% text(200,0.25,['\sigma = ' num2str(xSigma,'%.3f')]);
title('Lidar Error')
hold on; grid on;
ylabel('Latitude Error (meters)');
subplot(3,1,2)
% plot(t - t(1),dy);

plot(1:step_size:(max(size(ylidar))-step_size),lidar_error_y)
% ySigma = std(dy,'omitnan');
% text(200,0.05,['\sigma = ' num2str(ySigma,'%.3f')]);
hold on; grid on;
ylabel('Longitude Error (meters)');
subplot(3,1,3)
% plot(t - t(1), dyaw)

plot(1:step_size:(max(size(yawlidar))-step_size),lidar_error_yaw)
% zSigma = std(dyaw,'omitnan');
% text(200,0.15,['\sigma = ' num2str(zSigma,'%.3f')]);
hold on; grid on;
ylabel('Azimuth Error (degrees)');
xlabel('Lidar timestep')