% File used to generate Q matrix for August Volpe Demo

% To run: load /Data/Sign_pose_voxel1_dist0.1.mat
%   opens structure "store" in workspace

% position: [2851×7 double]:
    % Columns 1-3: Translation vector (x,y,z) (in meters)
    % Columns 4-7: Rotational vector (qw,qx,qy,qz) (in meters)
    % Position is calculated from “relPose” (derived from NDT-SLAM) if 
    %   “relPose” passes the distance move threshold (indicated in the filename as :dist#)

% eulAngles: [2851×3 double]:
    % Columns 1-3: 3D rotation in (x,y,z = YAQ, Pitch(?), Roll(?) (in radians)
    % “eulAngles” are derived from rotational vectors in “position” variable.

% relPose: [5670×7 double]: 
    % Columns 1-3: Translation vector (x,y,z) (in meters)
    % Columns 4-7: Rotational vector (qw,qx,qy,qz) (in meters)
    % This variable is an output of NDT-SLAM algorithm.

% relAngle: [5670×3 double]:
    % Columns 1-3: 3D rotation in (x,y,z = Roll, Pitch, Yaw) (in radians)
    % “relAngles” are derived from rotational vectors in “relPose” variable.

% Optimized_pos: [5670×7 double]:
    % “Optimized_pos” is empty since Loop closure is not applied as a part of SLAM.

% timeFrame: [2851×2 double] :
    % Columns: 1: Lidar time (in seconds)
    % Columns: 2: Lidar frame# associated with Lidar time that are accepted since they pass the distance moved threshold.

%plot time for debug
figure()
hold on
t = Store.timeFrame(:,1);
plot(t)
title('time for lidar')
xlabel('step')
ylabel('lidar time')

hold off


%Get RPY ------------------------------------------------------------------
%get orientation data from CSV

yaw_lidar = Store.eulAngles(:,1); %NOTE DATA IS ACTUALLY YPR?

%conver to positive rotations only
yaw_lidar(1685:2459) = yaw_lidar(1685:2459) + 2*pi;

%--------------------------------------------------------------------------
    
% 1) take data in one dimension ------------------------------------------
interpts = 100; %max(size(pos_x_lidar)); %number of points used in interpolation
method = 'cubic'; %method of interpolation used

pos_x_lidar = Store.position(:,1);
pos_y_lidar = Store.position(:,2);

figure()
hold on
plot(pos_x_lidar, pos_y_lidar)

% xq breaks down first scan into fewer points
xq = linspace(1,max(size(pos_x_lidar)),interpts); 
interp_x = interp1(pos_x_lidar, xq, method);
%xq2 generates more points on spline defined by interp_x
xq2 = linspace(1,interpts,max(size(pos_x_lidar)));
interp_x2 = interp1(interp_x, xq2, method);

yq = linspace(1,max(size(pos_x_lidar)),interpts);
interp_y = interp1(pos_y_lidar, yq, method);
%yq2 generates more points on spline defined by interp_x
yq2 = linspace(1,interpts,max(size(pos_x_lidar)));
interp_y2 = interp1(interp_y, yq2, method);

yawq = linspace(1,max(size(pos_x_lidar)),interpts);
interp_yaw = interp1(yaw_lidar, yawq, method);
%yawq2 generates more points on spline defined by interp_x
yawq2 = linspace(1,interpts,max(size(pos_x_lidar)));
interp_yaw2 = interp1(interp_yaw, yawq2, method);

% plot(interp_x, interp_y)
plot(interp_x2, interp_y2)
legend('actual','interpolated')
xlabel('x (m)')
ylabel('y (m)')

figure()
hold on
plot(t,pos_x_lidar)
plot(t,interp_x2)
% plot(linspace(1,max(size(pos_x_lidar)), interpts),interp_x)
legend('actual','interpolated')
xlabel('t (s)')
ylabel('dx (m)')
hold off

figure()
hold on
plot(t, pos_y_lidar)
plot(t, interp_y2)
legend('actual','interpolated')
xlabel('t (s)')
ylabel('dy (m)')
hold off

figure()
hold on
plot(t, yaw_lidar)
plot(t, interp_yaw2)
legend('actual','interpolated')
xlabel('t (s)')
ylabel('yaw (rad)')
hold off

% 2) subtract off cubic spline fit 
dx = pos_x_lidar - interp_x2.';
dy = pos_y_lidar - interp_y2.';
dyaw = yaw_lidar - interp_yaw2.';

bfx_world = movmean(dx,50);
bfy_world = movmean(dy,50);

figure()
hold on
% plot(dx)
% plot(dy)
plot(bfx_world)
plot(bfy_world)
title('error in world frame')
legend('error x (world frame)','error y (world frame)')
ylabel('error(m)')
xlabel('t (s)')
hold off

% 3) take std of residual error
std_x = std(dx)
std_y = std(dy)
std_yaw = std(dyaw)

%--------------------------------------------------------------------------
% transform errors in world frame X and Y into frame of vechicle at
% corresponding point in time.

dx_car_frame = dx.*cos(interp_yaw2.') - dy.*sin(interp_yaw2.');
dy_car_frame = dx.*sin(interp_yaw2.') + dy.*cos(interp_yaw2.');

std_x_car_frame = std(dx_car_frame)
std_y_car_frame = std(dy_car_frame)

bfx = movmean(dx_car_frame,50);
bfy = movmean(dy_car_frame,50);

figure()
hold on

plot(bfx)
plot(bfy)
legend('error x (car frame)','error y (car frame)')
xlabel('t (s)')
ylabel('error (m)')
hold off

% put everything together to get final Q matrix----------------------------
Q = zeros(3);
Q(1,1) = std_x^2;
Q(2,2) = std_y^2;
Q(3,3) = std_yaw^2;
%get non-diagonal elements
cov_xy = cov(dx, dy); %should this be in car frame or world frame?
Q(1,2) = cov_xy(1,2);
Q(2,1) = cov_xy(1,2);
cov_xyaw = cov(dx, dyaw);
Q(1,3) = cov_xyaw(1,2);
Q(3,1) = cov_xyaw(1,2);
cov_yyaw = cov(dy,dyaw);
Q(2,3) = cov_yyaw(1,2);
Q(3,2) = cov_yyaw(1,2);

Q

% close all
 