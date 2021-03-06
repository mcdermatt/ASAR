%main loop for Matt's implementation of Kalman filter of GPS/INS/Lidar data
%   going to try this first in 2D and see how things go


%TODO:
%        - confirm I should be using system dynamics for F in prediction
%            step matrix. Also this should be constant, right??

datalen = 100; %temp

% resArr(:,1:3) = lla
% lla == Lat. Lon. Alt. -> constains only INS info if INS only flag set in
% Liangchun's code

%time -> contains time absolute values associated with each INS measurement
%   

% TEMP INIT... (eventually these will be actual data matrices)
x_ins = zeros(4,1);
% g_u_ins = zeros(4,1);
g_u_ins = [1; 0; 1; 0];
P_ins = 1*ones(4,4);
Q_ins = 1*eye(4);

x_lidar = zeros(4,1);
P_lidar = 1*ones(4,4);
Q_lidar = 1*eye(4);

%absolute position measurements (wrong)
% lon_gps = dat.ppppos.ppplon;
% lat_gps = dat.ppppos.ppplat;
% hgt_gps = dat.ppppos.ppphgt;
% xyz_gps = wgslla2xyz(lon_gps, lat_gps, hgt_gps);
% x_gps = xyz_gps(1,:).';
% y_gps = xyz_gps(2,:).';
% z_gps = xyz_gps(3,:).';

gps_enu = load('gps_enu.mat').gps_enu;
pos_x_gps = gps_enu(1,:).';
pos_y_gps = gps_enu(2,:).';
pos_z_gps = gps_enu(3,:).';


%relative position estimates (correct) 
relPos_lidar = load('relPos_lidar.mat').relPos_lidar;
pos_x_lidar = relPos_lidar(:,1);
pos_y_lidar = relPos_lidar(:,2);

tlidar = load('lidar_time.mat').t;
tlidar = tlidar - tlidar(1);


new_GPS = false;
new_Lidar = false;
idx_lidar = 2;
next_lidar_timestep = tlidar(idx_lidar);
idx_gps = 2;
% next_gps_timestamp = 0; % not needed?

pos_history = zeros(2851, 4);

t = 0;
%loop through full length of dataset
for i = 1:datalen
    iter = 1;
    while new_GPS == false
        %debug
        if idx_lidar > 2849
            break
        end
        
        while new_Lidar == false
            %debug
            if idx_lidar > 2849
                break
            end
            
            %repeatedly make predictions based on INS only until Lidar is
            %available
            dt_ins   = 0.005; %constant step size in <imutime>
            

            %TODO: update vel_x and vel_y using moving average of last
            %       x and y position states
            
            F_ins =   [1 dt_ins 0 0;  % posx
                       0 1      0 0;  % velx
                       0 0      1 dt_ins; % posy
                       0 0      0 1];  %vely

            [x_ins, P_ins] = predictor(F_ins, x_ins, g_u_ins, P_ins, Q_ins);
            
            t = t + 0.005;
            pos_history(idx_lidar,:) = x_ins.';
            
            %if current time t is past the next lidar measurement
            %timestamp, use the lidar measurement now
            if t > next_lidar_timestep
                idx_lidar = idx_lidar + 1;
                next_lidar_timestep = tlidar(idx_lidar);
                new_Lidar = true;
%                 'using lidar now'
            end
        end
        % Get estimates from Lidar
        dt_lidar      = tlidar(idx_lidar-1) - tlidar(idx_lidar - 2); 
        F_lidar =   [1 dt_lidar 0 0;  % posx
                     0 1        0 0;  % velx
                     0 0        1 dt_lidar; % posy
                     0 0        0 1];  %vely    
        g_u_lidar = [pos_x_lidar(idx_lidar); 0; pos_y_lidar(idx_lidar); 0];
                 
        [x_lidar, P_lidar] = predictor(F_lidar, x_lidar, g_u_lidar, ...
                                        P_lidar, Q_lidar);
        %need to stretch x_lidar to account for the same duration
        %as the past n estimates from x_ins...
                                    
        % Fuse Lidar and INS here using weighted least-squares        
        W = [pinv(P_lidar) zeros(4,4); zeros(4,4) pinv(P_ins)];
        A = [eye(4); eye(4)] ;
        x_combined = pinv(A.'*W*A)*(A.')*W*[x_lidar; x_ins];
        P_combined = pinv(A.'*W*A)*(A.')*W*[P_lidar; P_ins];

        x_ins = x_combined;
        P_ins = P_combined;
        
        new_Lidar = false;
        
        if mod(t, 1) < 0.005 %time is just over the second mark
            idx_gps = idx_gps + 1
            new_GPS = true;          
        end
        
    end
    
    % Run correction step
    R = eye(4); %TODO: what does this do??
    H = eye(4); %temp
    y = [pos_x_gps(idx_gps); 0; pos_y_gps(idx_gps);0]; %temp 
    P_gps = 0.1*eye(4);
%     [x_plus, P_plus] = corrector(x_ins , P_ins, y, H, R); 
    [x_ins, P_ins] = corrector(x_ins , P_gps, y, H, R); %using x,p ins for variable to hold most accurate state info
    
    new_GPS = false;
    
end

figure()
plot(pos_history(1:2551,1), pos_history(1:2551,3))
% plot(pos_history(1:2551,3))

function [x_minus, P_minus] = predictor(F, x_plus, g_u, P_plus, Q)

    % Prediction step of EKF
    % Used for dead reckoning sensors (Lidar, INS)
    
    % Equation 1: actual prediction ------------------------------------
    % x_minus_new = F*x_plus_old + G(u)  
    
    %   x_minus_new == next uncorrected prediction
    %   F           == prediction step matrix 
    %                   use system dynamics??
    %   x_plus_old  == previous corrected prediction
    %   g_u         == G*u: control vector

    
    x_minus = F*x_plus + g_u;
    
%     x_minus = 0; %debug
    %-------------------------------------------------------------------
    
    % Equation 2: external influence (i.e. relative sensor readings)----
    % P_minus_new = F*P_plus_old*F.T + Q
    
    %   P_plus_old = prediction covariance matrix output from last iteration
    %   F          = same prediction step matrix as prev eqation
    %   Q          = sensor noise covariance (external influence)
    
     
    P_minus = F*P_plus*(F.') + Q;
%     P_minus = 0; %debug
    %--------------------------------------------------------------------
    
end

function [x_plus, P_plus] = corrector(x_minus , P_minus, y, H, R)

    %Update step of EKF
    %Used for absolute position sensors (GPS)
    %   x_plus  == corrected state estimate
    %   x_minus == corercted state error covariance
    %   H -> transforms state estimates back to measurement units
    %   R == covariance of sensor noise
    %   y == sensor measurement
    
    %calculate kalman gain k (weighting matrix)
    k = P_minus*(H.') * (H*P_minus*(H.') + R)^(-1);
    
    %correct state estiamte
    x_plus = x_minus + k*( y - x_minus );
%     k*( [1;1;1;1] - x_minus )

    %correct state error covariance P
    P_plus = (eye(4) - k*H)*P_minus*((eye(4) - k*H).') + k*R*(k.');

    
end

function xyz = wgslla2xyz(wlat, wlon, walt)

	A_EARTH = 6378137;
	flattening = 1/298.257223563;
	NAV_E2 = (2-flattening)*flattening; % also e^2
	d2r = pi/180;

	slat = sin(wlat*d2r);
	clat = cos(wlat*d2r);
	r_n = A_EARTH./sqrt(1 - NAV_E2.*slat.*slat);
	xyz =  [[(r_n + walt).*clat.*cos(wlon.*d2r)].' 
	        [(r_n + walt).*clat.*sin(wlon.*d2r)].' 
	        [(r_n.*(1 - NAV_E2) + walt).*slat ].'];

	if ((wlat < -90.0) | (wlat > +90.0) |...
				(wlon < -180.0) | (wlon > +360.0))
		error('WGS lat or WGS lon out of range');
    end
end