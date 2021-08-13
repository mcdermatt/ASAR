% runlen = 10;
runlen = max(size(lat))-100;

%Q and R from get_Q_3DOF, get_R_3DOF
%noise covariance matrix (estimated from Lidar data)
%   units in m, rad
Q = [[0.0987    0.0020   -0.0002];
     [0.0020    0.0358    0.0003];
     [-0.0002    0.0003    0.0007]];

% measurement covariance matrix (estimated from GPS data)
%   units in m, rad
R = [[0.1829    0.0000   -0.0000];
     [0.0000    0.6226   -0.0000];
     [-0.0000   -0.0000    0.0587]];
 

% state transition model
F = eye(3);

% observation model
H = eye(3); %TOD0: change this so we can take in measurements in lat,lon,deg

I = eye(3);
P_plus = R;
x_plus = zeros(3,1);

sigma_x_history = zeros(1,runlen);
state_estimate = zeros(3,runlen);

count = 1;
while count <= runlen
   
    %prediciton step --------------------------------------
    % get relative transformation estimate from NDT
    Gu = [Store.relPose(count,1), Store.relPose(count,2), rad2deg(Store.relPose(count,4))].'; 
    x_minus = F*x_plus; %+ Gu; 
    
    P_minus = F*P_plus*(F.') + Q;
    %------------------------------------------------------
    
    
    %correction step---------------------------------------
    L = P_minus*(H.')*pinv((H*P_minus)*(H.') + R);
    
%   take in absolute position estimates from GPS
    y = [lat(count), lon(count), heading(count)].';
    x_plus = x_minus + L*(y - (H*x_minus)); %y is absolute estimate from GPS/ INS
    
    P_plus = (I - L*H)*P_minus*((I - L*H).') + L*R*(L.');
    %------------------------------------------------------
    
    sigma_x_history(count) = sqrt(P_plus(1,1));
    
    state_estimate(:,count) = x_plus;
    
    count = count + 1; 
end

% figure()
% hold on
% plot(sigma_x_history)
% title('sigma x vs time')
% xlabel('time (s)')
% ylabel('sigma x (m)')

% %

% %Heading --------------------------
figure()
hold on
plot(heading)
plot(interp_heading2)
plot(state_estimate(3,:))
title('raw gps heading vs spline gps heading vs kalman filtered heading (using GPS + Lidar)')
xlabel('time (s)')
ylabel('heading (deg)')
legend('raw GPS data', 'spline smoothed GPS data', 'Kalman Filtered data')

hold off