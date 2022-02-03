function [F, Qk, vel, dv, lla_in, lla_ins, qbn, cbn] = phi_angle_model(msrk1, msrk, vel0, dv0, lla0_ins, qbn0)

%Inputs:
%   msrk1    = new INS measurement vector (first element is timestamp)
%   msrk     = last INS measuremnt vector (recorded ~0.005s ago)
%   vel0     = initial velocity at time = msrk1(1)
%   dv0      = initial change in velocity at time = msrk1(1)
%   lla0_ins = initial estimated Lat. Lon. Alt
%   qbn0     = initial estimated heading in quaternions 

%Outputs:
%   F        =    

WGS84_A = 6378137.0;           % earth semi-major axis (WGS84) (m) 
WGS84_B = 6356752.3142;        % earth semi-minor axis (WGS84) (m) 
e = sqrt(WGS84_A * WGS84_A - WGS84_B * WGS84_B) / WGS84_A; %0.0818

% rotational angular velocity of earth
omega_e = 7.2921151467e-5;   

% wgs84 = wgs84Ellipsoid;
dt = msrk1(1) - msrk(1);

% update the velocity
gy0 = msrk(2:4);
ac0 = msrk(5:7);
gy = msrk1(2:4);
ac = msrk1(5:7);
dvfb_k = 1.0/12.0*(cross(gy0,ac) + cross(ac0,gy)) + 0.5*cross(gy,ac);
dvfb_k = ac + dvfb_k;

% compute cbn0
cbn0 = quat2dcm(qbn0);

qne = zeros(1,4);
qne(1) = cos(-pi / 4.0 - lla0_ins(1) / 2.0) * cos(lla0_ins(2) / 2.0);
qne(2) = -sin(-pi / 4.0 - lla0_ins(1) / 2.0) * sin(lla0_ins(2) / 2.0);
qne(3) = sin(-pi / 4.0 - lla0_ins(1) / 2.0) * cos(lla0_ins(2) / 2.0);
qne(4) = cos(-pi / 4.0 - lla0_ins(1) / 2.0) * sin(lla0_ins(2) / 2.0);

qee_h = zeros(1,4); qnn_l = zeros(1,4);

% compute rn, rm
rn = WGS84_A / sqrt(1 - e * e * sin(lla0_ins(1)) * sin(lla0_ins(1)));
rm = WGS84_A * (1 - e * e) / sqrt(power(1 - e * e * sin(lla0_ins(1)) * sin(lla0_ins(1)), 3));

% compute wie
wie(1) = omega_e * cos(lla0_ins(1));
wie(2) = 0;
wie(3) = -omega_e * sin(lla0_ins(1));

% compute wen
wen(1) = vel0(2) / (rn + lla0_ins(3));
wen(2) = -vel0(1) / (rm + lla0_ins(3));
wen(3) = -vel0(2) * tan(lla0_ins(1)) / (rn + lla0_ins(3));
win = wie + wen;

% compute g_l
gl = zeros(1,3);

grav = [9.7803267715, 0.0052790414, 0.0000232718, -0.000003087691089, 0.000000004397731, 0.000000000000721];
sinB = sin(lla0_ins(1));
sinB2 = sinB * sinB;
sinB4 = sinB2 * sinB2;
gl(3) = grav(1) * (1.0 + grav(2) * sinB2 + grav(3) * sinB4) + (grav(4) + grav(5) * sinB2) * lla0_ins(3) + grav(6) * lla0_ins(3) * lla0_ins(3);

zeta_m = dt.*win/2.0;
half_zeta = 0.5*zeta_m;
mold = norm(half_zeta);
sc = sin(mold) / mold;
qnn_l(1) = cos(mold);
qnn_l(2) = sc * half_zeta(1);
qnn_l(3) = sc * half_zeta(2);
qnn_l(4) = sc * half_zeta(3);

wiee = [0 0 omega_e];

xi_m = dt.*wiee/2.0;       % 0.5: half size for extrapolating lla at time(k+1/2) 
half_xi = -0.5*xi_m;
mold = norm(half_xi);
sc = sin(mold) / mold;
qee_h(1) = cos(mold);
qee_h(2) = sc * half_xi(1);
qee_h(3) = sc * half_xi(2);
qee_h(4) = sc * half_xi(3);

qne_l = quatmultiply(qne, qnn_l);
qne_m = quatmultiply(qee_h, qne_l);

lla_m = zeros(1,3);

if qne_m(1) ~= 0
    lla_m(2) = 2 * atan(qne_m(4) / qne_m(1));
    lla_m(1) = 2 * (-pi / 4.0 - atan(qne_m(3) / qne_m(1)));
elseif qne_m(1) == 0 & qne_m(3) == 0
    lla_m(2) = pi;
    lla_m(1) = 2 * (-pi / 4.0 - atan(-qne_m(2) / qne_m(4)));
elseif qne_m(1) == 0 & qne_m(4) == 0
    lla_m(2) = 2 * atan(-qne_m(2) / qne_m(3));
    lla_m(1) = pi / 2.0;
end
    
lla_m(3) = lla0_ins(3) - (vel0(3) * dt) / 2.0;

% extrapolate the speed
vel_m = vel0 + 0.5 * dv0;

% compute the wie_m, wen_m - ANGULAR VELOCITIES
% compute rn, rm
rn = WGS84_A / sqrt(1 - e * e * sin(lla_m(1)) * sin(lla_m(1)));
rm = WGS84_A * (1 - e * e) / sqrt(power(1 - e * e * sin(lla_m(1)) * sin(lla_m(1)), 3));

% compute wie_m
wie_m(1) = omega_e * cos(lla_m(1));
wie_m(2) = 0;
wie_m(3) = -omega_e * sin(lla_m(1));

% compute wen_m
wen_m(1) = vel_m(2) / (rn + lla_m(3));
wen_m(2) = -vel_m(1) / (rm + lla_m(3));
wen_m(3) = -vel_m(2) * tan(lla_m(1)) / (rn + lla_m(3));
win_m = wie_m + wen_m;

% compute g_l
gl_m = zeros(1,3);
sinB = sin(lla_m(1));
sinB2 = sinB * sinB;
sinB4 = sinB2 * sinB2;
gl_m(3) = grav(1) * (1.0 + grav(2) * sinB2 + grav(3) * sinB4) + (grav(4) + grav(5) * sinB2) * lla0_ins(3) + grav(6) * lla0_ins(3) * lla0_ins(3);

% compute true zeta
zeta = win_m .* dt;
cnn = eye(3,3);

cnn(1,2) = -0.5 * zeta(3);
cnn(1,3) =  0.5 * zeta(2);
cnn(2,1) =  0.5 * zeta(3);
cnn(2,3) = -0.5 * zeta(1);
cnn(3,1) = -0.5 * zeta(2);
cnn(3,2) =  0.5 * zeta(1);
    
% calculate dvfn_k, dvgn_k
dvfn_k = (cbn0'*cnn)*dvfb_k; %was this in Liangchun's code
% dvfn_k = (cbn0'*cnn)*dvfb_k .* dt; % I think Liangchun forgot to multiply by dt here???
dvfn_k = dvfn_k';
dvgn_k = (gl_m-cross((2*wie_m+wen_m), vel_m)).* dt;

% update velocity
dv = dvfn_k + dvgn_k;
vel = vel0 + dv;

qnn_h = zeros(1,4);
qee_l = zeros(1,4);
    
% recompute the wie_m, wen_m
rn = WGS84_A / sqrt(1 - e * e * sin(lla_m(1)) * sin(lla_m(1)));
rm = WGS84_A * (1 - e * e) / sqrt(power(1 - e * e * sin(lla_m(1)) * sin(lla_m(1)), 3));

% compute wie_m
wie_m(1) = omega_e * cos(lla_m(1));
wie_m(2) = 0;
wie_m(3) = -omega_e * sin(lla_m(1));

% compute wen_m
wen_m(1) = vel_m(2) / (rn + lla_m(3));
wen_m(2) = -vel_m(1) / (rm + lla_m(3));
wen_m(3) = -vel_m(2) * tan(lla_m(1)) / (rn + lla_m(3));
win_m = wie_m + wen_m;

% compute g_l
gl_m = zeros(1,3);
sinB = sin(lla_m(1));
sinB2 = sinB * sinB;
sinB4 = sinB2 * sinB2;
gl_m(3) = grav(1) * (1.0 + grav(2) * sinB2 + grav(3) * sinB4) + (grav(4) + grav(5) * sinB2) * lla0_ins(3) + grav(6) * lla0_ins(3) * lla0_ins(3);

% recompute zeta, xi
zeta_m = win_m.*dt;
half_zeta = 0.5*zeta_m;
mold = norm(half_zeta);
sc = sin(mold) / mold;
qnn_h(1) = cos(mold);
qnn_h(2) = sc * half_zeta(1);
qnn_h(3) = sc * half_zeta(2);
qnn_h(4) = sc * half_zeta(3);

xi_m = wiee.* dt;
half_xi = -0.5*xi_m;
mold = norm(half_xi);
sc = sin(mold) / mold;
qee_l(1) = cos(mold);
qee_l(2) = sc * half_xi(1);
qee_l(3) = sc * half_xi(2);
qee_l(4) = sc * half_xi(3);

% recompute the qnn_h
qne_h = quatmultiply(qne, qnn_h);
qne = quatmultiply(qee_l, qne_h);

lla_ins = zeros(1,3);

if qne(1) ~= 0
    lla_ins(2) = 2 * atan(qne(4) / qne(1));
    lla_ins(1) = 2 * (-pi / 4.0 - atan(qne(3) / qne(1)));
elseif qne(1) == 0 & qne(3) == 0
    lla_ins(2) = pi;
    lla_ins(1) = 2 * (-pi / 4.0 - atan(-qne(2) / qne(4)));
elseif qne(1) == 0 & qne(4) == 0
    lla_ins(2) = 2 * atan(-qne(2) / qne(3));
    lla_ins(1) = pi / 2.0;
end

lla_ins(3) = lla0_ins(3) - vel_m(3) * dt;

% update the attitude
qdthe_half = zeros(1,4);
qne0 = zeros(1,4);

% compute qne0
qne0(1) = cos(-pi / 4.0 - lla0_ins(1) / 2.0) * cos(lla0_ins(2) / 2.0);
qne0(2) = -sin(-pi / 4.0 - lla0_ins(1) / 2.0) * sin(lla0_ins(2) / 2.0);
qne0(3) = sin(-pi / 4.0 - lla0_ins(1) / 2.0) * cos(lla0_ins(2) / 2.0);
qne0(4) = cos(-pi / 4.0 - lla0_ins(1) / 2.0) * sin(lla0_ins(2) / 2.0);

qneo_inv = quatinv(qne0);

qdthe = quatmultiply(qneo_inv, qne)';

vec = zeros(1,3);

if qdthe(1) ~= 0
    phi_m = atan(sqrt(qdthe(2)*qdthe(2)+qdthe(3)*qdthe(3)+qdthe(4)*qdthe(4))/qdthe(1));
    f = 0.5 * sin(phi_m) / phi_m;
    vec(1) = qdthe(2) / f;
    vec(2) = qdthe(3) / f;
    vec(3) = qdthe(4) / f;
else
    vec(1:3) = pi * qdthe(2:4);
end

vec_half = 0.5 * vec;

v = 0.5 * vec_half;
mold = norm(v, 3);
sc = sin(mold) / mold;
qdthe_half(1) = cos(mold);
qdthe_half(2) = sc * v(1);
qdthe_half(3) = sc * v(2);
qdthe_half(4) = sc * v(3);
    
qne_m = quatmultiply(qne0, qdthe_half);        % communication law
lla_m = zeros(1,3);

if qne_m(1) ~= 0
    lla_m(2) = 2 * atan(qne_m(4) / qne_m(1));
    lla_m(1) = 2 * (-pi / 4.0 - atan(qne_m(3) / qne_m(1)));
elseif qne_m(1) == 0 & qne_m(3) == 0
    lla_m(2) = pi;
    lla_m(1) = 2 * (-pi / 4.0 - atan(-qne_m(2) / qne_m(4)));
elseif qne_m(1) == 0 & qne_m(4) == 0
    lla_m(2) = 2 * atan(-qne_m(2) / qne_m(3));
    lla_m(1) = pi / 2.0;
end

lla_m(3) = (lla0_ins(3) + lla_ins(3)) / 2.0;

% compute rn, rm
rn = WGS84_A / sqrt(1 - e * e * sin(lla_m(1)) * sin(lla_m(1)));
rm = WGS84_A * (1 - e * e) / sqrt(power(1 - e * e * sin(lla_m(1)) * sin(lla_m(1)), 3));

% compute wie_m
wie_m(1) = omega_e * cos(lla_m(1));
wie_m(2) = 0;
wie_m(3) = -omega_e * sin(lla_m(1));

% compute wen_m
wen_m(1) = vel_m(2) / (rn + lla_m(3));
wen_m(2) = -vel_m(1) / (rm + lla_m(3));
wen_m(3) = -vel_m(2) * tan(lla_m(1)) / (rn + lla_m(3));
win_m = wie_m + wen_m;

% compute g_l
gl_m = zeros(1,3);
sinB = sin(lla_m(1));
sinB2 = sinB * sinB;
sinB4 = sinB2 * sinB2;
gl_m(3) = grav(1) * (1.0 + grav(2) * sinB2 + grav(3) * sinB4) + (grav(4) + grav(5) * sinB2) * lla0_ins(3) + grav(6) * lla0_ins(3) * lla0_ins(3);

% recompute zeta, xi
phi = gy+1/12*cross(gy0, gy);
half_phi = 0.5*phi;
mold = norm(half_phi);
sc = sin(mold) / mold;
qbb(1) = cos(mold);
qbb(2) = sc * half_phi(1);
qbb(3) = sc * half_phi(2);
qbb(4) = sc * half_phi(3);

zeta = win_m.*dt;
half_zeta = -0.5*zeta;
mold = norm(half_zeta);
sc = sin(mold) / mold;
qnn(1) = cos(mold);
qnn(2) = sc * half_zeta(1);
qnn(3) = sc * half_zeta(2);
qnn(4) = sc * half_zeta(3);

tmp_q = quatmultiply(qbn0, qbb);
qbn = quatmultiply(qnn, tmp_q);

% normalization
% e_q = sumsqr(qbn);
% e_q = 1 - 0.5 * (e_q - 1);
% qbn = e_q.*qbn;
qbn = quatnormalize(qbn);

rpy = flip(quat2eul(qbn));

% ------------------ Phi-angle model GPS/INS LC algorithm -------------- %
% prediction
% compute rn, rm
rn = WGS84_A / sqrt(1 - e * e * sin(lla_ins(1)) * sin(lla_ins(1)));
rm = WGS84_A * (1 - e * e) / sqrt(power(1 - e * e * sin(lla_ins(1)) * sin(lla_ins(1)), 3));

c1 = 0; c2 = 1;
    
lla_in = c1*lla0_ins + c2*lla_ins;
vel_in = c1*vel0 + c2*vel;

Arr = zeros(3,3);
Arr(1,3) = -vel_in(1)/(rm + lla_in(3))^2;
Arr(2,1) = vel_in(2)*sin(lla_in(1))/((rn + lla_in(3))*(cos(lla_in(1)))^2);
Arr(2,3) = -vel_in(2)/((rn + lla_in(3))^2*cos(lla_in(1)));

Arv = zeros(3,3);
Arv(1,1) = 1/(rm + lla_in(3));
Arv(2,2) = 1/((rn + lla_in(3))*cos(lla_in(1)));
Arv(3,3) = -1;

Avr(1,1) = -2*vel_in(2)*omega_e*cos(lla_in(1))-(vel_in(2))^2/((rn + lla_in(3))*(cos(lla_in(1)))^2);
Avr(1,3) = -vel_in(1)*vel_in(3)/(rm + lla_in(3))^2+(vel_in(2))^2*tan(lla_in(1))/(rn + lla_in(3))^2;
Avr(2,1) = 2*omega_e*(vel_in(1)*cos(lla_in(1))-vel_in(3)*sin(lla_in(1)))+(vel_in(2)*vel_in(1))/((rn + lla_in(3))*(cos(lla_in(1)))^2);
Avr(2,3) = (-vel_in(2)*vel_in(3)-vel_in(1)*vel_in(2)*tan(lla_in(1)))/(rn + lla_in(3))^2;
Avr(3,1) = 2*vel_in(2)*omega_e*sin(lla_in(1));
Rmn = sqrt(rm*rn);
Avr(3,3) = (vel_in(2))^2/(rn + lla_in(3))^2+(vel_in(1))^2/(rm + lla_in(3))^2-2*gl(3)/(Rmn + lla_in(3));

Aer = zeros(3,3);
Aer(1,1) = -omega_e*sin(lla_in(1));
Aer(1,3) = -vel_in(2)/(rn + lla_in(3))^2;
Aer(2,3) = vel_in(1)/(rm+lla_in(3))^2;
Aer(3,1) = -omega_e*cos(lla_in(1))-vel_in(2)/((rn + lla_in(3))*(cos(lla_in(1)))^2);
Aer(3,3) = vel_in(2)*tan(lla_in(1))/(rn + lla_in(3))^2;

Avv = zeros(3,3);
Avv(1,1) = vel_in(3)/(rm + lla_in(3));
Avv(1,2) = -2*omega_e*sin(lla_in(1))-2*vel_in(2)*tan(lla_in(1))/(rn + lla_in(3));
Avv(1,3) = vel_in(1)/(rm + lla_in(3));
Avv(2,1) = 2*omega_e*sin(lla_in(1))+vel_in(2)*tan(lla_in(1))/(rn + lla_in(3));
Avv(2,2) = (vel_in(3)+vel_in(1)*tan(lla_in(1)))/(rn + lla_in(3));
Avv(2,3) = 2*omega_e*cos(lla_in(1))+vel_in(2)/(rn + lla_in(3));
Avv(3,1) = -2*vel_in(1)/(rm+lla_in(3));
Avv(3,2) = -2*omega_e*cos(lla_in(1))-2*vel_in(2)/(rn + lla_in(3));

Aev = zeros(3,3);
Aev(1,2) = 1/(rn + lla_in(3));
Aev(2,1) = -1/(rm + lla_in(3));
Aev(3,2) = -tan(lla_in(1))/(rn + lla_in(3));

% compute cbn
cbn = quat2dcm(qbn);
cbn = cbn';

fn = cbn*(ac/dt);

%I think this has to do with "Correlation time"?
Tba = 4*3600;
Tbg = 4*3600;
Tsa = 4*3600;
Tsg = 4*3600;

fn_cross = [0 -fn(3) fn(2); fn(3) 0 -fn(1); -fn(2) fn(1) 0];
win_cross = [0 -win(3) win(2); win(3) 0 -win(1); -win(2) win(1) 0];
f_diag = diag(ac/dt);
w_diag = diag(gy/dt);

A = [Arr, Arv, zeros(3,3), zeros(3,3), zeros(3,3), zeros(3,3), zeros(3,3); ...
    Avr, Avv, fn_cross, cbn, zeros(3,3), cbn*f_diag, zeros(3,3); ...
    Aer, Aev, -win_cross, zeros(3,3), -cbn, zeros(3,3), -cbn*w_diag; ...
    zeros(3,3), zeros(3,3), zeros(3,3), diag([-1.0/Tba,-1.0/Tba, -1.0/Tba]), zeros(3,3), zeros(3,3), zeros(3,3); ...
    zeros(3,3),zeros(3,3),zeros(3,3),zeros(3,3), diag([-1.0/Tbg,-1.0/Tbg, -1.0/Tbg]), zeros(3,3), zeros(3,3); ...
    zeros(3,3), zeros(3,3), zeros(3,3), zeros(3,3), zeros(3,3), diag([-1.0/Tsa,-1.0/Tsa, -1.0/Tsa]), zeros(3,3); ...
    zeros(3,3), zeros(3,3), zeros(3,3), zeros(3,3), zeros(3,3), zeros(3,3), diag([-1.0/Tsg,-1.0/Tsg, -1.0/Tsg])];

A = double(A);

F = eye(21)+A*dt;

B = zeros(21,18);
B(4:6,1:3) = cbn;
B(7:9,4:6) = -cbn;
B(10:12,7:9) = eye(3);
B(13:15,10:12) = eye(3);
B(16:18,13:15) = eye(3);
B(19:21,16:18) = eye(3);

% set up Q
qg = (deg2rad(0.2))^2;   %initial bias error         
% qg = (deg2rad(0.5))^2; 
qa = 0.05^2;             %non-orthogonality  - not provided for new PwrPak7
% qa = 0.06^2;

%Gyro white noise = 0.0022 deg/sqrt(h) per Liangchun's pdf
% qbg = 2*(deg2rad(0.0022)/60)^2/Tbg; %with old unit
qbg = 2*(deg2rad(0.06)/60)^2/Tbg; %ours is 0.06
% Liangchun's accel White noise = 0.00075 (m/s/sqrt(hr))
% qba = 2*(0.00075/60)^2/Tba; %with old unit
qba = 2*(0.025/60)^2/Tba; %again, much higher on the new unit

qsg = 2*(10*10^-6)^2/Tsg;
qsa = 2*(10*10^-6)^2/Tsa;
Q = zeros(18,18);
Q(1,1) = qa; Q(2,2) = qa; Q(3,3) = qa; 
Q(4,4) = qg; Q(5,5) = qg; Q(6,6) = qg;
Q(7,7) = qba; Q(8,8) = qba; Q(9,9) = qba; 
Q(10,10) = qbg; Q(11,11) = qbg; Q(12,12) = qbg;
Q(13,13) = qsa; Q(14,14) = qsa; Q(15,15) = qsa; 
Q(16,16) = qsg; Q(17,17) = qsg; Q(18,18) = qsg;
Qk = 0.5*(F*B*Q*B'+B*Q*B'*F')*dt;
% Qk = (F*B*Q*B'*F')*dt;

end