%How to use this demo:
%   run <readoem7memmap_update.m>
%       select /Data/2021010pp7garminlidar/(data file goes here) in GUI
%       this imports "heading", "bplat", "bplon" to workspace

%imoport data on lidar distances traveled each step

% runlen = 1500;
runlen = max(size(lat)); %needs debug

%Q and R from get_Q_3DOF, get_R_3DOF
%noise covariance matrix (estimated from Lidar data)
%   units in m, rad
Q = [[0.0987    0.0000   -0.000];
     [0.000    0.0358    0.000];
     [-0.000    0.000    0.0007]];

% measurement covariance matrix (estimated from GPS data)
%   units in m, rad
% R = [[0.1829    0.0000   -0.0000];
%      [0.0000    0.6226   -0.0000];
%      [-0.0000   -0.0000    0.0587]]; %was this 8/15
R = [[0.05    0.00   -0.0];
     [0.00    0.05   -0.0];
     [-0.0   -0.0    0.00031]];
 

% state transition model
F = eye(3);

% observation model
H = eye(3); %TOD0: change this so we can take in measurements in lat,lon,deg

I = eye(3);
P_plus = R; %init here...
x_plus = zeros(3,1);

sigma_x_history = zeros(1,runlen);
state_estimate = zeros(3,runlen);

count = 1;
while count <= runlen
   
    %prediciton step --------------------------------------
    % get relative transformation estimate from NDT
    Gu = [Store.relPose(count,1), Store.relPose(count,2), rad2deg(Store.relAngle(count,3))].';   
    x_minus = F*x_plus + Gu; 
    
    P_minus = F*P_plus*(F.') + Q;
    %------------------------------------------------------
    
    
    %correction step---------------------------------------
    L = P_minus*(H.')*pinv((H*P_minus)*(H.') + R);
    
%   take in absolute position estimates from GPS
%     y = [lat(count), lon(count), heading(count)].';
    y = [lat(count), lon(count), h(count)].';
    % convert from lat/ lon to m - not needed anymore here...
%     y(1) = deg2km(y(1))*1000;
%     y(2) = 40075*cos(deg2rad(y(1)))*y(2);
    x_plus = x_minus + L*(y - (H*x_minus)); %y is absolute estimate from GPS/ INS
    
    P_plus = (I - L*H)*P_minus*((I - L*H).') + L*R*(L.');
    %------------------------------------------------------
    
    sigma_x_history(count) = sqrt(P_plus(1,1));
    
    state_estimate(:,count) = x_plus;
    
    count = count + 1; 
end

% Timescale conversion ------------------------------------------
%adjust heading data to allow continuous rotation
h = heading - heading(1); %set to 0 initial heading
h(1186:1646) = h(1186:1646) - 360;
h(1:1150) = []; %chop off beginning to get data to start at the same time
h(525:end) = []; %chop off end to make vectors same length of time

yaw_lidar_deg = -rad2deg(yaw_lidar);
yaw_lidar_deg(1:156) = []; %remove flat data at beginning
t_yaw = t;
t_yaw(1:156) = [];

%interpolate h to get same number of points yaw_lidar (to run filter)
hq = linspace(2680,3203,max(size(yaw_lidar_deg)));
t_h = linspace(2680,3203,524); %stretch to fit Lidar timescale
h_interp = interp1(t_h,h,t_yaw); %interpolate along t_yaw because lidar is non-uniform sampling
%TODO: change it so both h_interp and yaw_lidar are both sampled uniformly

%repeat for x and y data
% gps_in_m = Wgslla2xyz(bplat, bplon, bphgt); %nope
gps_enu = Wgslla2enu(bplat, bplon, bphgt, bplat(1), bplon(1), bphgt(1));
x_gps = -gps_enu(1,:);
y_gps = -gps_enu(2,:);
% yaw_gps = gps_in_m(3,:); %not needed?

x_gps(1:1150) = [];
x_gps(525:end) = [];
y_gps(1:1150) = [];
y_gps(525:end) = [];
x_gps = x_gps - x_gps(1); %ititialize at 0
y_gps = y_gps - y_gps(1);

% figure()
% hold on
% plot(t_yaw,h_interp)
% plot(t_yaw,yaw_lidar_deg)
% title('GPS vs Lidar Heading')
% legend('GPS', 'Lidar')
% xlabel('timestep')
% ylabel('heading (deg)')
% hold off

% figure()
% hold on
% title("heading difference between BESTPOS and Lidar")
% xlabel('timestep')
% ylabel('difference (deg)')
% plot(h_interp - yaw_lidar_deg);
% hold off

% figure()
% hold on
% title('y gps')
% plot(y_gps)
% hold off
% 
% figure()
% hold on
% title('x gps')
% plot(x_gps)
% hold off

% figure()
% hold on
% title("GPS vs Lidar")
% xlabel('x (m)')
% ylabel('y (m)')
% plot(x_gps, y_gps)
% plot(interp_x2,interp_y2)
% legend('GPS (BESTPOS)' , 'Lidar')
% hold off
% 
% figure()
% hold on
% title('x gps vs x lidar')
% plot(t_h,x_gps)
% plot(t,interp_x2)
% legend('GPS (BESTPOS)', 'Lidar')
% hold off
% 
% figure()
% hold on
% title('y gps vs y lidar')
% plot(t_h,y_gps)
% plot(t,interp_y2)
% legend('GPS (BESTPOS)', 'Lidar')
% hold off


%-------------------------------------------------------------------------

% figure()
% hold on
% plot(sigma_x_history)
% title('sigma x vs time')
% xlabel('time (s)')
% ylabel('sigma x (m)')

% %

% % Trajectory ----------------------
figure()
hold on
title('kalman filtered trajectory')
plot(x_gps, y_gps)
plot(pos_x_lidar,pos_y_lidar)
plot(state_estimate(2,:),state_estimate(1,:))
legend("BESTPOS", "Lidar", "Kalman")
hold off

% %Heading --------------------------
figure()
hold on
plot(t_h,h)
% plot(t_h,interp_heading2)
plot(t_yaw,yaw_lidar_deg)
plot(t_h,state_estimate(3,:)) %need to offset time so kalman data is not shifted
title('Heading: Raw GPS vs Spline GPS vs Lidar vs Kalman')
xlabel('time (s)')
ylabel('heading (deg)')
% legend('raw GPS data', 'spline smoothed GPS data', 'Raw lidar', 'Kalman Filtered (GPS+LIDAR)')
legend('raw GPS data', 'Raw lidar', 'Kalman Filtered (GPS+LIDAR)')
hold off

% pos x --------------------------
figure()
hold on
plot(t_h,x_gps)
% plot(t_h,interp_heading2)
plot(t,interp_x2)
plot(t_h,state_estimate(2,:)) %need to offset time so kalman data is not shifted
title('X: Raw GPS vs Spline GPS vs Lidar vs Kalman')
xlabel('time (s)')
ylabel('x (m)')
% legend('raw GPS data', 'spline smoothed GPS data', 'Raw lidar', 'Kalman Filtered (GPS+LIDAR)')
legend('raw GPS data', 'Raw lidar', 'Kalman Filtered (GPS+LIDAR)')
hold off