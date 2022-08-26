%script for loading synthetic point clouds and registering with canned NDT
% solver

clear all
close all

ans_cum = [];

for j = 1:3
    for i = 1:40
        40*(j-1) + i
    
    %     scan1_fn = "MC_trajectories/scene1_scan1.txt";
    %     scan2_fn = "MC_trajectories/scene1_scan2.txt";
        scan1_fn = "MC_trajectories/scene1_scan" + string(mod(40*(j-1) + i, 40) + 1) + ".txt";
        scan2_fn = "MC_trajectories/scene1_scan" + string(mod(40*(j-1) + i, 40)+2) + ".txt";    

        scan1 = readmatrix(scan1_fn);
        scan2 = readmatrix(scan2_fn);
        
        %TODO: need to rotate scan2 relative to scan1 since rotations can't be done
        % easily in point cloud generation script
        rotm = eul2rotm([0.05, 0, 0]);
        scan2 = scan2*rotm;
        
        %remove ground plane--------------------------
        gph = -0.5; %ground plane height
        goodidx1 = find(scan1(:,3)>-1.8);
        scan1 = scan1(goodidx1, :);
        goodidx2 = find(scan2(:,3)>-1.8);
        scan2 = scan2(goodidx2, :);
        % groundPtsIdx1 = segmentGroundFromLidarData(moving); %builtin func
        %---------------------------------------------
        
        %add noise to each PC
        noise_scale = 0.02;
        scan1 = scan1 + noise_scale*randn(size(scan1));
        scan2 = scan2 + noise_scale*randn(size(scan2));
        
        moving = pointCloud(scan2);
        c1=uint8(zeros(moving.Count,3));
        c1(:,1)=255;
        c1(:,2)=50;
        c1(:,3)=50;
        moving.Color = c1;
        
        fixed = pointCloud(scan1);
        c2=uint8(zeros(fixed.Count,3));
        c2(:,1)=50;
        c2(:,2)=50;
        c2(:,3)=255;
        fixed.Color = c2;
        
    %     figure()
    %     hold on
    %     pcshow(fixed)
    %     pcshow(moving)
        
        gridstep = 1;
    
        % cheat initial transformation estimate
        tinit = rigid3d(eul2rotm([-0.05,0,0]), [0.5,0,0]);
    
        %NDT---------------------------------------------
        [tform, movingReg, rmse] = pcregisterndt(moving, fixed, gridstep, OutlierRatio=0.5, MaxIterations=50); %try messing with OutlierRatio
        
        %------------------------------------------------
        
        %ICP---------------------------------------------
    %     [tform,movingReg, rmse] = pcregistericp(moving,fixed);    
%         [tform,movingReg, rmse] = pcregistericp(moving,fixed, 'metric', 'pointToPlane');
    %     [tform,movingReg, rmse] = pcregistericp(moving,fixed, 'metric', 'pointToPlane', "InitialTransform",tinit); %cheating for debug
    %     [tform,movingReg, rmse] = pcregistericp(moving,fixed, 'metric', 'pointToPoint');
        %------------------------------------------------
        
        %LOAM ---------------------------------------------
        
        %--------------------------------------------------
        
        ans = [tform.Translation, rotm2eul(tform.Rotation)]
        ans_cum = [ans_cum; ans];
    end
end
RMSE = sqrt(mean((ans_cum - [0.5, 0, 0, -0.05, 0, 0]).^2))

%save to file
% fn = "MC_results/traj1_cart_ICP_point2plane_NO_GP.txt";
fn = "MC_results/traj1_cart_NDT_NO_GP.txt";
writematrix(ans_cum, fn, 'Delimiter', 'tab')
