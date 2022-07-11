% Run readoem7memmap_update.m and rife_repackageData.m OR
% load signageData 

% ELLAPSED TIME
t = rawimusx.rawimugnsssow-rawimusx.rawimugnsssow(1);

%ACCEL data
% Condition data
ax = double(rawimusx.rawimuxaccel)*(0.400/65536)*(9.80665/1000)/200;
ay = double(rawimusx.rawimuyaccel)*(0.400/65536)*(9.80665/1000)/200;
az = double(rawimusx.rawimuzaccel)*(0.400/65536)*(9.80665/1000)/200;

%%% Detrend
%ax = ax - median(rx);
%ay = ay - median(ry);
%az = az - median(rz);

%GYRO data
% Condition data
gx = double(rawimusx.rawimuxgyro)*0.0151515/65536/200;
gy = double(rawimusx.rawimuygyro)*0.0151515/65536/200;
gz = double(rawimusx.rawimuzgyro)*0.0151515/65536/200;


% Plot
figure(1);
subplot(3,1,1);
plot(t,ax)
ylabel('Accel_x m/s')
subplot(3,1,2);
plot(t,ay)
ylabel('Accel_y m/s')
subplot(3,1,3);
plot(t,az)
ylabel('Accel_z m/s')
xlabel('Ellapsed Time (sec)');

%
figure(2);
subplot(3,1,1);
plot(t,gx,'b.-')
ylabel('Gyro_x deg')
%axis([1100 1200 -0.05 0.05])
subplot(3,1,2);
plot(t,gy,'b.-')
ylabel('Gyro_y deg')
%axis([1100 1200 -0.05 0.05])
subplot(3,1,3);
plot(t,gz,'b.-')
ylabel('Gyro_z deg')
xlabel('Ellapsed Time (sec)');
%axis([1100 1200 -0.2 0.2])

%%
% ERROR PLOTS
figure(3)
subplot(3,1,1);
plot(inspvax.sow1465-inspvax.sow1465(1),inspvax.insrollstd,'b.-')
ylabel('Roll \sigma_x deg')
%axis([1100 1200 -0.05 0.05])
subplot(3,1,2);
plot(inspvax.sow1465-inspvax.sow1465(1),inspvax.inspitchstd,'b.-')
ylabel('Pitch \sigma_y deg')
%axis([1100 1200 -0.05 0.05])
subplot(3,1,3);
plot(inspvax.sow1465-inspvax.sow1465(1),inspvax.insazimstd,'b.-')
ylabel('Yaw \sigma_z deg')
xlabel('Ellapsed Time (sec)');
%axis([1100 1200 -0.2 0.2])
