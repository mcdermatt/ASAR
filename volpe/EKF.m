function [out] = EKF(vals)

%function for updating state and covaiance estimates to prepare for sensor
%fusion in an EKF
%NOTE: EKF prediction step is NOT implemented inside this function

%extract input parameters -------------
lla0_ins = vals.lla0_ins;
ned0_lidar = vals.ned0_lidar;
lla0_combined = vals.lla0_combined;
lla_ins_last = vals.lla_ins_last;
vel0 = vals.vel0;
rpy0 = vals.rpy0;
dv0 = vals.dv0;
qbn0 = vals.qbn0;
% qbn_lidar = vals.qbn_lidar;
msrk = vals.msrk;
msrk1 = vals.msrk1;
gpsmsr = vals.gpsmsr;
lidarmsr = vals.lidarmsr;
x_ins = vals.x_ins;
xHatM_ins = vals.xHatM_ins;
x_lidar = vals.x_lidar;
x_lidar_cum = vals.x_lidar_cum;
PM_ins = vals.PM_ins;
PP_ins_last = vals.PP_ins_last;
PM_combined = vals.PM_combined;
gpsCnt = vals.gpsCnt;
lidarCnt = vals.lidarCnt;
PM_lidar_last = vals.PM_lidar_last;
fuse_lidar = vals.fuse_lidar;
fuse_gps = vals.fuse_gps;
Qk200 = vals.Qk200;
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

% evolve INS state estimates via Forward Euler integration (A priori)
xHatM_ins = F*xHatM_ins;         

PM_ins = F*PM_ins*F'+Qk;                    % cov. Estimate

% F200 = F*F200; %keep product of sequential F matrices for use in Lidar fusion0
%for debug- add up Qk's between fusion frames to verify PM_ins is evolving
Qk200 = Qk200 + Qk; 

%transform xHatM_ins into ENU here before combining with lidar...
[x_ins_E, x_ins_N, x_ins_U] = geodetic2enu(lla_ins(1), lla_ins(2), lla_ins(3), lla0_ins(1), lla0_ins(2), lla0_ins(3), wgs84Ellipsoid, 'radians');
x_ins(1:3) = [x_ins_N, x_ins_E, x_ins_U];

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
    x_lidar = [lidarmsr(3), lidarmsr(2), lidarmsr(4)].';
    ned_lidar = ned0_lidar + (x_lidar(1:3).');
    PM_lidar = PM_lidar_last + Qk_lidar;   
    lidarUpdated = 1;
    lidarCnt = lidarCnt + 1;
else
    lidarUpdated = 0;
    ned_lidar = ned0_lidar;
    PM_lidar = PM_lidar_last;                                 
end

%e == 0.0818
rn = WGS84_A / sqrt(1 - e * e * sin(lla_in(1)) * sin(lla_in(1)));
rm = WGS84_A * (1 - e * e) / sqrt(power(1 - e * e * sin(lla_in(1)) * sin(lla_in(1)), 3));
coeff = diag([(rm+lla_in(3)); ((rn+lla_in(3))*cos(lla_in(1))); -1]);

% update with GPS measurements
if gpstime <= msrk1(1) && gpstime > msrk(1)  % If GPS time between two inertial measurement times, do an update %

    %R == [std_gps_lat, std_gps_lon, std_gps_hgt]
    R = 0.01*diag(gpsmsr(5:7).^2);  % R matrix -- note coefficient of 0.01!!!
    dr = [0;0;0];  % LEVER ARM in m (XYZ body frame, YX(-Z) aligns with NED)
    lla_gps_corr = coeff*(gpsmsr(2:4))' - cbn*dr; %from Liangchun

    gpsResUpd = [gpstime, lla_gps_corr'];
    if fuse_gps == 1
        
        y = coeff*lla_in' - lla_gps_corr; %this is in enu   
        
        H = zeros(3,21);
        H(1:3, 1:3) = coeff; %H transforms (lla IN RADIANS!!) -> (m)

        L = PM_ins*H'*pinv(H*PM_ins*H'+R, 1e-20); %added tol to help with singular inversion

        yHat = H*xHatM_ins;
%         yHat = xHatM(1:3); % using this if xHatM is already in (m)

        %Use GPS to update correction -------------------------------------
        xHatM_ins = xHatM_ins + L*(y-yHat);  %was this
%         xHatM_ins = xHatM_ins + L*(y-yHat); 
        %------------------------------------------------------------------
        
        %2/28: don't take PM from GPS combination if we're just using GPS
        %        to correct bias of INS unit
%         PM_ins = (eye(size(F))-L*H)*PM_ins*(eye(size(F))-L*H)'+L*R*L';

    %     L(1,1) % 1.5e-7 == 1/coeff(1,1)
    %     coeff(1,1) = 6.36e6, coeff(2,2) = 4.97e6
      
   
    %Direct state correction (depreciated) --------------------
    
% %         lla_ins = gpsmsr(2:4);

%     %     xi = rad2deg(xHatM_ins(7:9));
%     %     E = [0 -xi(3) xi(2); xi(3) 0 -xi(1); -xi(2) xi(1) 0];
%     %     cbn = (eye(3)+E)*cbn;
% 
%         %comment out below to ignore correction step on quaternions
%         % not ideal but helps convergence?? -LX
%     %     qbn = dcm2quat(cbn');
% 
%         % normalization
%     %     e_q = sumsqr(qbn);
%     %     e_q = 1 - 0.5 * (e_q - 1);
%     %     qbn = e_q.*qbn;
%     
%         qbn = quatnormalize(qbn);
%         
%         % flip the sign if two quaternions are opposite in sign
%         if (qbn(1)*qbn0(1)<0)
%             qbn = -qbn;
%         end
% 
%         vel = vel - [xHatM_ins(4,4), xHatM_ins(5,5), xHatM_ins(6,6)];
% %         vel = vel - (xHatM_ins(4:6))'; % was this        
%         
%         rpy = flip(quat2eul(qbn));
%         dv = vel - vel0;
    %-----------------------------------------------------------
        
    end
    gpsUpdated = 1; % flag used in INS fusion, regardless if GPS is actually used
    gpsCnt = gpsCnt + 1;
            
else
    
    gpsResUpd = [];
    gpsUpdated = 0;

end


%store updated estimates --------------------------------------
out.lla_ins = lla_ins;
out.ned_lidar = ned_lidar;
out.vel = vel;
out.rpy = rpy;
out.dv = dv;
out.qbn = qbn;
out.xHatM_ins = xHatM_ins;
out.x_ins = x_ins;
out.x_lidar = x_lidar;
out.gpsUpdated = gpsUpdated;
out.lidarUpdated = lidarUpdated;
out.gpsCnt = gpsCnt;
out.lidarCnt = lidarCnt;
out.gpsResUpd = gpsResUpd;
out.PM_ins = PM_ins;
out.PM_lidar = PM_lidar;
out.F = F;
out.Qk200 = Qk200;    
%----------------------------------------------------------------

end
