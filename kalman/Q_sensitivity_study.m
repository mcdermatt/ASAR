
t = Store.timeFrame(:,1);

%Get RPY ------------------------------------------------------------------
%get orientation data from CSV
yaw_lidar = Store.eulAngles(:,1); %NOTE DATA IS ACTUALLY YPR?
%conver to positive rotations only
yaw_lidar(1685:2459) = yaw_lidar(1685:2459) + 2*pi;
%--------------------------------------------------------------------------

interpts = linspace(1000,100,10);
method = 'cubic'; %method of interpolation used
err = zeros(3,max(size(interpts)));

for i = 1:max(size(interpts))
    pos_x_lidar = Store.position(:,1);
    pos_y_lidar = Store.position(:,2);

    % xq breaks down first scan into fewer points
    xq = linspace(1,max(size(pos_x_lidar)),interpts(i)); 
    interp_x = interp1(pos_x_lidar, xq, method);
    %xq2 generates more points on spline defined by interp_x
    xq2 = linspace(1,interpts(i),max(size(pos_x_lidar)));
    interp_x2 = interp1(interp_x, xq2, method);

    yq = linspace(1,max(size(pos_x_lidar)),interpts(i));
    interp_y = interp1(pos_y_lidar, yq, method);
    %yq2 generates more points on spline defined by interp_x
    yq2 = linspace(1,interpts(i),max(size(pos_x_lidar)));
    interp_y2 = interp1(interp_y, yq2, method);

    yawq = linspace(1,max(size(pos_x_lidar)),interpts(i));
    interp_yaw = interp1(yaw_lidar, yawq, method);
    %yawq2 generates more points on spline defined by interp_x
    yawq2 = linspace(1,interpts(i),max(size(pos_x_lidar)));
    interp_yaw2 = interp1(interp_yaw, yawq2, method);

    % 2) subtract off cubic spline fit 
    dx = pos_x_lidar - interp_x2.';
    dy = pos_y_lidar - interp_y2.';
    dyaw = yaw_lidar - interp_yaw2.';

    bfx_world = movmean(dx,50);
    bfy_world = movmean(dy,50);

    % 3) take std of residual error
    std_x = std(dx)
    std_y = std(dy)
    std_yaw = std(dyaw)
    
    err(1,i) = std_x;
    err(2,i) = std_y;
    err(3,i) = std_yaw;
    
end

figure()
hold on
title("Sensitivity to Spline Fit")
plot(interpts,err(1,:))
plot(interpts,err(2,:))
legend('x','y')
xlabel("Number of interpolated points")
ylabel("Standard deviation (m)")
hold off
