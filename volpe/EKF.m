function [lla_ins, ned_lidar, lla_combined, vel, rpy, dv, qbn, xHatM_ins, x_lidar, xHatM_combined, gpsUpdated, lidarUpdated, gpsCnt, lidarCnt, gpsResUpd, PM_ins, PM_lidar, PM_combined, F200, Qk200] = ...
    EKF(lla0_ins, ned0_lidar, lla0_combined, lla_ins_last, vel0, dv0, qbn0, msrk, msrk1, gpsmsr, lidarmsr, xHatM_ins, x_lidar, x_lidar_cum, PM_ins, PP_ins_last, PM_combined, gpsCnt, lidarCnt, PM_lidar_last, fuse_lidar, fuse_gps, F200, Qk200)

%extract input parameters -------------
% lla0_ins = EKF_input(1);
% ned0_lidar = EKF_input(2);
% lla0_combined = EKF_input(3);
% lla_ins_last = EKF_input(4);
% vel0 = EKF_input(5);
% rpy0 = EKF_input(6);
% dv0 = EKF_input(7);
% qbn0 = EKF_input(8);
% qbn_lidar = EKF_input(9);
% msrk = EKF_input(10);
% msrk1 = EKF_input(11);
%---------------------------------------


WGS84_A = 6378137.0;           % earth semi-major axis (WGS84) (m) 
WGS84_B = 6356752.3142;        % earth semi-minor axis (WGS84) (m) 
e = sqrt(WGS84_A * WGS84_A - WGS84_B * WGS84_B) / WGS84_A; %0.0818

gpstime = gpsmsr(1);
gpsmsr(2:3) = deg2rad(gpsmsr(2:3));

%Provided by Liangchun:
[F, Qk, vel, dv, lla_in, lla_ins, qbn, cbn] = phi_angle_model(msrk1, msrk, vel0, dv0, lla0_ins, qbn0);

% normalization
% e_q = sumsqr(qbn);
% e_q = 1 - 0.5 * (e_q - 1);
% qbn = e_q.*qbn;
qbn = quatnormalize(qbn);
rpy = flip(quat2eul(qbn));

% Extended Kalman Filter --------
xHatM_ins = F*xHatM_ins;          % Forward Euler integration (A priori)

PM_ins = F*PM_ins*F'+Qk;                    % cov. Estimate
% PM_ins(1,1)

F200 = F*F200; %keep product of sequential F matrices for use in Lidar fusion0
%for debug- add up Qk's between fusion frames to verify PM_ins is evolving
Qk200 = Qk200 + Qk; 

%transform xHatM_ins into ENU here before combining with lidar...
[xHatM_ins_E, xHatM_ins_N, xHatM_ins_U] = geodetic2enu(lla_ins(1), lla_ins(2), lla_ins(3), lla0_ins(1), lla0_ins(2), lla0_ins(3), wgs84Ellipsoid, 'radians');
xHatM_ins(1:3) = [xHatM_ins_N, xHatM_ins_E, xHatM_ins_U];

%set noise covariance matrix for lidar
Qk_lidar = zeros(3,3); %test w/ zeros
% from Final_Slides.pptx - experimentally determined values from the summer
% Qk_lidar(1,1) = 0.000784; %(m) 
% Qk_lidar(2,2) = 0.000484; %(m)
% Qk_lidar(3,3) = 0.005625; %(m)

Qk_lidar(1,1) = 1.4e-17; %sigmalon**2 (deg)
Qk_lidar(2,2) = 1.4e-17; %sigmalat**2 (deg)
Qk_lidar(3,3) = 0.0001;%sigmaz**2 (meters)

lidartime = lidarmsr(1);
if lidartime <= msrk1(1) && lidartime > msrk(1)
    %only using lidar translation estimates:
       %[dxyz dvxyz quats ...] -> 21x1 vec    

%     x_lidar = lidarmsr(2:4).'; %was this-> wrong??
    x_lidar = [lidarmsr(3), lidarmsr(2), lidarmsr(4)].';
    
    
    PM_lidar = PM_lidar_last + Qk_lidar;
    
    lidarUpdated = 1;
    lidarCnt = lidarCnt + 1;
    
    ned_lidar = ned0_lidar + (x_lidar(1:3).');
    
else
    lidarUpdated = 0;
    ned_lidar = ned0_lidar;
    PM_lidar = PM_lidar_last; %TODO - figure out a cleaner way to do this
                   %(this results in a step function for INS uncertainty)                                
end

%e == 0.0818
rn = WGS84_A / sqrt(1 - e * e * sin(lla_in(1)) * sin(lla_in(1)));
rm = WGS84_A * (1 - e * e) / sqrt(power(1 - e * e * sin(lla_in(1)) * sin(lla_in(1)), 3));
coeff = diag([(rm+lla_in(3)); ((rn+lla_in(3))*cos(lla_in(1))); -1]);

% update with GPS measurements
if gpstime <= msrk1(1) && gpstime > msrk(1)  % If GPS time between two inertial measurement times, do an update %

    if fuse_lidar == 1
        %new ------------------------------------------------------

        % W = ([P_k         , F_big*P_{k-1};
        %       P_{k-1}*F_big^T , P_{k-1} + Qk_lidar ])^-1
        % P_k = P_bar + P_prime
        % P_bar = ??
        
        W = pinv([PM_ins, F200*PP_ins_last;
                  PP_ins_last*(F200.'), PP_ins_last + [10*Qk_lidar, zeros(3,18); zeros(18,21)]]);   
        H = [eye(21);  eye(3), zeros(3,18); zeros(18,21)];

        %TODO- update PM_combined at every time step
        PM_combined = pinv(H.' * W * H);

        %lla_lidar = (lla_ins at last fusion timestamp) + (x_lidar in LLA)
        %update cumulative counts for this segment to include current xHatM estimates 
        x_lidar_cum = x_lidar + x_lidar_cum.';        
        lla_lidar = lla_ins_last + [x_lidar_cum(1)/coeff(1,1), x_lidar_cum(2)/coeff(2,2), -x_lidar_cum(3)];
        
%        %This SHOULD work (but currently does not due to issues with W???):
%         lla_ins = pinv((H.')*W*H)*(H.')*W*[lla_ins.'; zeros(18,1); lla_lidar.'; zeros(18,1)];
% %         %resize to 3x1
%         lla_ins = lla_ins(1:3).';
% %         lla_ins

       
        %DEBUG: WLS is fluctuating A LOT and occationally giving negative values
%         weights = pinv((H.')*W*H)*(H.')*W;
%         'ins'
%         weights(1,1)  %ins
%         'lidar'
%         weights(1,22) %lidar
    
%         'PM_lidar'
%         PM_lidar(1,1)
%         'PM_ins'
%         PM_ins(1,1)
%         'PP_ins_last'
%         PP_ins_last(1,1)
%         'PM_combined'
%         PM_combined(1,1) %this is exploding...
        %----------------------------------------------------------
                
        %Reset F and Q
        F200 = F; 
        Qk200 = zeros(21,21);
        
        xHatM_combined = xHatM_ins;
        
    else
        lla_combined = lla_ins;
        xHatM_combined = xHatM_ins;
    end
%     xHatM = xHatM_ins; % not sure if correct...
    
    %R == [std_gps_lat, std_gps_lon, std_gps_hgt]
    R = 0.01*diag(gpsmsr(5:7).^2);  % R matrix -- note coefficient of 0.01!!!
    dr = [0;0;0];  % LEVER ARM in m (XYZ body frame, YX(-Z) aligns with NED)
    lla_gps_corr = coeff*(gpsmsr(2:4))' - cbn*dr; %from Liangchun

    gpsResUpd = [gpstime, lla_gps_corr'];
    if fuse_gps == 1
        %TODO DEBUG: lla_in (rad), gpsmsr (rad)
        y = coeff*lla_in' - lla_gps_corr; %this is in enu   
        
        %Kalman Correction Step --------------------------
        %FROM GPS DATA
        H = zeros(3,21);
        %H transforms (lla IN RADIANS!!) -> (m)
        H(1:3, 1:3) = coeff;

        L = PM_ins*H'*pinv(H*PM*H'+R, 1e-20); %added tol to help with singular inversion

        yHat = H*xHatM; %was this
%         yHat = xHatM(1:3); % using this since xHatM is already in (m)
        xHatP = xHatM + L*(y-yHat);         % a posteriori estimate (m) 
        
%         c = L*(y-yHat);
%         c(7:end)        

%         coeff*lla_in'
%         lla_gps_corr

        xHatM_ins = xHatP;
                
        PM_ins = (eye(size(F))-L*H)*PM*(eye(size(F))-L*H)'+L*R*L';
        %-------------------------------------------------

    %     L(1,1) % 1.5e-7 == 1/coeff(1,1)
    %     coeff(1,1) = 6.36e6, coeff(2,2) = 4.97e6

        vel = vel - (xHatM_ins(4:6))'; % this used to cause stepping behavior
    
    end
        
    
    %TODO: use WLS to combine contributions of Lidar and INS to update qbn    
%     xi = rad2deg(xHatM_ins(7:9));
%     E = [0 -xi(3) xi(2); xi(3) 0 -xi(1); -xi(2) xi(1) 0];
%     cbn = (eye(3)+E)*cbn;

  
    
    %comment out below to ignore correction step- not ideal but helps
    %convergence?? -LX
%     qbn = dcm2quat(cbn');

    % normalization
%     e_q = sumsqr(qbn);
%     e_q = 1 - 0.5 * (e_q - 1);
%     qbn = e_q.*qbn;
    qbn = quatnormalize(qbn);

    % flip the sign if two quaternions are opposite in sign
    if (qbn(1)*qbn0(1)<0)
        qbn = -qbn;
    end

    rpy = flip(quat2eul(qbn));

    dv = vel - vel0;

    gpsUpdated = 1; % flag for INS fusion, regardless if GPS is actually used
    gpsCnt = gpsCnt + 1;
    
    %--------------------
    
    lla_combined = lla_ins(1:3); %TODO: get rid of lla_combined (not useful)
    
else
    lla_combined = lla0_combined;
%     xHatP = xHatM; %TODO-- figure out if I should still output this...
    xHatM_combined = xHatM_ins; %placeholder
    
    gpsResUpd = [];
    gpsUpdated = 0;
    
%     PM_combined = PM_ins; %placeholder
    % evolve PM_combined states similar to PM_ins
%     PM_combined = F*PM_combined*F' +  Qk;

    %test
%     W = pinv([PM_ins, F200*PP_ins_last;
%               PP_ins_last*(F200.'), PP_ins_last + [10*Qk_lidar, zeros(3,18); zeros(18,21)]]);  
%     H = [eye(21);  eye(3), zeros(3,18); zeros(18,21)];
%     PM_combined = pinv(H.' * W * H);

end


end
