% File used to generate Q matrix for August Volpe Demo

% To run: load /Data/Sign_pose_voxel1_dist0.1.mat
%   opens structure "store" in workspace

% set(0,'defaultfigurecolor',[1 1 1])

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

t = Store.timeFrame(:,1);
    
%plot time for debug
% figure()
% hold on
% plot(t)
% title('time for lidar')
% xlabel('step')
% ylabel('lidar time')
% hold off


%Get RPY ------------------------------------------------------------------
%get orientation data from CSV

yaw_lidar = Store.eulAngles(:,1); %NOTE DATA IS ACTUALLY YPR

%conver to positive rotations only
yaw_lidar(1685:2459) = yaw_lidar(1685:2459) + 2*pi;
%convert to deg
yaw_lidar = rad2deg(yaw_lidar);

%--------------------------------------------------------------------------
    
% 1) take data in one dimension ------------------------------------------
pos_x_lidar = Store.position(:,1);
pos_y_lidar = Store.position(:,2);

% %Matt's Subsampling and interpolation method ------
% %Possible issues with cutting off peaks
% interpts = 100; %max(size(pos_x_lidar)); %number of points used in interpolation
% method = 'cubic'; %method of interpolation used
% % xq breaks down first scan into fewer points
% xq = linspace(1,max(size(pos_x_lidar)),interpts); 
% interp_x = interp1(pos_x_lidar, xq, method);
% %xq2 generates more points on spline defined by interp_x
% xq2 = linspace(1,interpts,max(size(pos_x_lidar)));
% interp_x2 = interp1(interp_x, xq2, method);
% %repeat for y
% yq = linspace(1,max(size(pos_x_lidar)),interpts);
% interp_y = interp1(pos_y_lidar, yq, method);
% %yq2 generates more points on spline defined by interp_x
% yq2 = linspace(1,interpts,max(size(pos_x_lidar)));
% interp_y2 = interp1(interp_y, yq2, method);
% 
% yawq = linspace(1,max(size(pos_x_lidar)),interpts);
% interp_yaw = interp1(yaw_lidar, yawq, method);
% %yawq2 generates more points on spline defined by interp_x
% yawq2 = linspace(1,interpts,max(size(pos_x_lidar)));
% interp_yaw2 = interp1(interp_yaw, yawq2, method);
% %----------------------------------------------------


%Jason's Cubic fit method -----------------------------
% Build window
windowHalfWidth = 10;  % User defined parameter
windowFullWidth = 2*windowHalfWidth+1;
% Re-name data read by read0em7memmap_update.m
timeRef = t(1); % units sec 
timeVec = t;    % units sec
timeIntoExperiment = timeVec-timeRef;

% Initialize fit
interp_x2 = zeros(size(pos_x_lidar))*NaN;  % Initialize fit to values of "Not a Number"
dx = interp_x2;

interp_y2 = zeros(size(pos_y_lidar))*NaN;  % Initialize fit to values of "Not a Number"
dy = interp_y2;

interp_yaw2 = zeros(size(yaw_lidar))*NaN;  % Initialize fit to values of "Not a Number"
dyaw = interp_yaw2;

% Cycle through each time
lengthData = length(pos_x_lidar);   
for indx = windowHalfWidth+1:lengthData-windowHalfWidth % scan over window center points
    windowedData = pos_x_lidar(indx-windowHalfWidth:indx+windowHalfWidth);
    centerData = pos_x_lidar(indx);
    tCenter = timeIntoExperiment(indx);
    t_window = timeIntoExperiment(indx-windowHalfWidth:indx+windowHalfWidth)-tCenter;
    oneVec = ones(windowFullWidth,1);
    A = [oneVec t_window t_window.^2 t_window.^3]; % If sample rate is consistent, then this matrix is constant; 
                             %  but computing this matrix at each time step
                             %  allows us to deal with jumps in the sample
                             %  time
    coef = A\windowedData; % Compute cubic coefficients
    interp_x2(indx) = A(windowHalfWidth+1,:)*coef;
%     dx(indx) = centerData-interp_x(indx); % subtract fit from the data at the center point

    windowedData = pos_y_lidar(indx-windowHalfWidth:indx+windowHalfWidth);
    centerData = pos_y_lidar(indx);
    coef = A\windowedData; % Compute cubic coefficients
    interp_y2(indx) = A(windowHalfWidth+1,:)*coef;

    windowedData = yaw_lidar(indx-windowHalfWidth:indx+windowHalfWidth);
    centerData = yaw_lidar(indx);
    coef = A\windowedData; % Compute cubic coefficients
    interp_yaw2(indx) = A(windowHalfWidth+1,:)*coef;

end
%get rid of NaN values at beginning and end of interp
interp_x2(1:windowHalfWidth) = pos_x_lidar(1:windowHalfWidth);
interp_x2(end-windowHalfWidth:end) = pos_x_lidar(end-windowHalfWidth:end);
dx = pos_x_lidar - interp_x2;

interp_y2(1:windowHalfWidth) = pos_y_lidar(1:windowHalfWidth);
interp_y2(end-windowHalfWidth:end) = pos_y_lidar(end-windowHalfWidth:end);
dy = pos_y_lidar - interp_y2;

interp_yaw2(1:windowHalfWidth) = yaw_lidar(1:windowHalfWidth);
interp_yaw2(end-windowHalfWidth:end) = yaw_lidar(end-windowHalfWidth:end);
dyaw = yaw_lidar - interp_yaw2;

%------------------------------------------------------
% plot(t)

fig1 = figure()
fig1.Color = 'white'
hold on
plot(pos_x_lidar, pos_y_lidar)
plot(interp_x2, interp_y2)
legend('actual','interpolated')
xlabel('x (m)')
ylabel('y (m)')
hold off
% 

fig2 = figure()
fig2.Color = 'white'
tiledlayout(2,3)



% 2) subtract off cubic spline fit 
dx = pos_x_lidar - interp_x2;
dy = pos_y_lidar - interp_y2;
dyaw = yaw_lidar - interp_yaw2;

% 3) take std of residual error
std_x = std(dx)
std_y = std(dy)
std_yaw = std(dyaw)


nexttile
hold on
% scatter(t-t(1),pos_x_lidar, 3, 'Filled')
% scatter(t-t(1),interp_x2, 3, 'Filled')
title("x")
plot(t-t(1),pos_x_lidar)
% plot(t-t(1),interp_x2)
% plot(linspace(1,max(size(pos_x_lidar)), interpts),interp_x)
% legend('Raw Lidar','Cubic Fit')
xlabel('t (s)')
ylabel('x (m)')
hold off

nexttile
hold on
title("y")
plot(t-t(1), pos_y_lidar)
% plot(t-t(1), interp_y2)
% scatter(t-t(1), pos_y_lidar, 3, 'Filled')
% scatter(t-t(1), interp_y2, 3, 'Filled')
% legend('Raw Lidar','Cubic Fit')
xlabel('t (s)')
ylabel('y (m)')
hold off

nexttile
hold on
plot(t-t(1), yaw_lidar)
% plot(t-t(1), interp_yaw2)
title("yaw")
% scatter(t-t(1), yaw_lidar, 3, 'Filled')
% scatter(t-t(1), interp_yaw2, 3, 'Filled')
% legend('Raw Lidar','Cubic Fit')
xlabel('t (s)')
ylabel('yaw (deg)')
hold off

nexttile
hold on
title("σ_x = " + num2str(round(std_x,2)) + ",    mean = " + num2str(round(mean(dx),4)))
plot(t - t(1),dx)
xlabel("time (s)")
ylabel("spline fit error (m)")
hold off

nexttile
hold on
title("σ_y = " + num2str(round(std_y,2)) + ",    mean = " + num2str(round(mean(dy),4)))
plot(t - t(1),dy)
xlabel("time (s)")
ylabel("spline fit error (m)")
hold off

nexttile
hold on
title("σ_y_a_w = " + num2str(round(std_yaw,2)) + ",    mean = " + num2str(round(mean(dyaw),4)))
plot(t - t(1),dyaw)
xlabel("time (s)")
ylabel("spline fit error (deg)")
hold off


% bfx_world = movmean(dx,50);
% bfy_world = movmean(dy,50);

% figure()
% hold on
% plot(bfx_world)
% plot(bfy_world)
% title('error in world frame')
% legend('error x (world frame)','error y (world frame)')
% ylabel('error(m)')
% xlabel('t (s)')
% hold off


%--------------------------------------------------------------------------
% transform errors in world frame X and Y into frame of vechicle at
% corresponding point in time.

dx_car_frame = dx.*cos(interp_yaw2.') - dy.*sin(interp_yaw2.');
dy_car_frame = dx.*sin(interp_yaw2.') + dy.*cos(interp_yaw2.');

std_x_car_frame = std(dx_car_frame);
std_y_car_frame = std(dy_car_frame);

% bfx = movmean(dx_car_frame,50);
% bfy = movmean(dy_car_frame,50);

bfx = dx_car_frame;
bfy = dy_car_frame;

% figure()
% hold on
% plot(bfx)
% plot(bfy)
% legend('error x (car frame)','error y (car frame)')
% xlabel('t (s)')
% ylabel('error (m)')
% hold off

% put everything together to get final Q matrix----------------------------
Q = zeros(3);
Q(1,1) = std_x^2;
Q(2,2) = std_y^2;
Q(3,3) = std_yaw^2;
%get non-diagonal elements
% cov_xy = cov(dx, dy); %should this be in car frame or world frame?
% Q(1,2) = cov_xy(1,2);
% Q(2,1) = cov_xy(1,2);
% cov_xyaw = cov(dx, dyaw);
% Q(1,3) = cov_xyaw(1,2);
% Q(3,1) = cov_xyaw(1,2);
% cov_yyaw = cov(dy,dyaw);
% Q(2,3) = cov_yyaw(1,2);
% Q(3,2) = cov_yyaw(1,2);

Q


fig3 = figure();
fig3.Color = 'white';
subplot(3,1,1)
% plot(t - t(1),dx);
plot(dx);
zSigma = std(dx,'omitnan');
text(200,0.25,['\sigma = ' num2str(zSigma,'%.3f')]);
title('Lidar')
hold on; grid on;
ylabel('Latitude Error (meters)');
subplot(3,1,2)
% plot(t - t(1),dy);
plot(dy)
zSigma = std(dy,'omitnan');
text(200,0.05,['\sigma = ' num2str(zSigma,'%.3f')]);
hold on; grid on;
ylabel('Longitude Error (meters)');
subplot(3,1,3)
% plot(t - t(1), dyaw)
plot(dyaw)
zSigma = std(dyaw,'omitnan');
text(200,0.15,['\sigma = ' num2str(zSigma,'%.3f')]);
hold on; grid on;
ylabel('Azimuth Error (degrees)');
% xlabel('Time (s)');
xlabel('sample number')