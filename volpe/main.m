%main loop for Matt's implementation of Kalman filter of GPS/INS/Lidar data
%   going to try this first in 2D and see how things go


%TODO:
%        - confirm I should be using system dynamics for F in prediction
%            step matrix. Also this should be constant, right??

datalen = 1; %temp

% INIT...
x_ins = ones(4,1);
g_u_ins = ones(4,1);
P_ins = ones(4,4);
Q_ins = 0.1*eye(4);

x_lidar = ones(4,1);
g_u_lidar = ones(4,1);
P_lidar = ones(4,4);
Q_lidar = 0.01*eye(4);

new_GPS = false;
new_Lidar = false;

%loop through full length of dataset
for i = 1:datalen
    iter = 1;
    while new_GPS == false

        while new_Lidar == false
            %repeatedly make predictions based on INS only until Lidar is
            %available
            dt   = 0.2;
            F_ins =   [1 dt 0 0;  % posx
                       0 1  0 0;  % velx
                       0 0  1 dt; % posy
                       0 0  0 1];  %vely
            
            [x_ins, P_ins] = predictor(F_ins, x_ins, g_u_ins, P_ins, Q_ins);
            
            %temp
            if rand(1) > 0.8
                new_Lidar = true;
            end
        
        % Get estimates from Lidar
        dt_lidar      = 0.5;
        F_lidar =   [1 dt_lidar 0 0;  % posx
                     0 1        0 0;  % velx
                     0 0        1 dt_lidar; % posy
                     0 0        0 1];  %vely    
        
        [x_lidar, P_lidar] = predictor(F_lidar, x_lidar, g_u_lidar, ...
                                        P_lidar, Q_lidar);
                 
        % Fuse Lidar and INS here using weighted least-squares        
        W = [pinv(P_lidar) zeros(4,4); zeros(4,4) pinv(P_ins)];
        A = [eye(4); eye(4)] ;
        x_combined = pinv(A.'*W*A)*(A.')*W*[x_lidar; x_ins];
        P_combined = pinv(A.'*W*A)*(A.')*W*[P_lidar; P_ins];

%         x_ins = x_combined;
%         P_ins = P_combined;
        
        end
        %temp for debug -------
        iter = iter + 1;
        if iter > 3
            new_GPS = true;
        end
        %----------------------
    end
    
    % Run correction step
    
end


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

function [x_plus, P_plus] = corrector(x_minus , P_minus, H, R)

    %Update step of EKF
    %Used for absolute position sensors (GPS)
    %   x_plus  ==
    %   x_minus ==
    %   y       == 

    %x_plus = x_minus + k( h^(-1) - x_minus )

    % y = Hx
    %[X_lidar X_ins].T = ([I I].T)(x)
    

    x_plus = None;
    P_plus = None;
    
end