%script for loading synthetic point clouds and registering with canned NDT
% solver

clear all
close all

ans_cum = [];

for j = 1:3
    for i = 1:38%40
%         40*(j-1) + i
        38*(j-1) + i
    
    %     scan1_fn = "MC_trajectories/scene1_scan1.txt";
    %     scan2_fn = "MC_trajectories/scene1_scan2.txt";
        scan1_fn = "MC_trajectories/scene1_scan" + string(mod(40*(j-1) + i, 40) + 1) + ".txt";
        scan2_fn = "MC_trajectories/scene1_scan" + string(mod(40*(j-1) + i, 40)+2) + ".txt";    
%         scan1_fn = "unshadowed_points/scene_1_scan_" + string(mod(40*(j-1) + i, 40)) + "_no_shadows.txt";
%         scan2_fn = "unshadowed_points/scene_1_scan_" + string(mod(40*(j-1) + i, 40)+1) + "_no_shadows.txt";    


        scan1 = readmatrix(scan1_fn);
        scan2 = readmatrix(scan2_fn);
        
        %TODO: need to rotate scan2 relative to scan1 since rotations can't be done
        % easily in point cloud generation script
        rotm = eul2rotm([0.05, 0, 0]);
        scan2 = scan2*rotm;
        
        %remove ground plane--------------------------
        % gph = -1.5; %ground plane height for KITTI
        gph = -1.8; %ground plane height for simulated scene 1
        goodidx1 = find(scan1(:,3)>gph);
        scan1 = scan1(goodidx1, :);
        goodidx2 = find(scan2(:,3)>gph);
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
%         tinit = rigid3d(eul2rotm([-0.05,0,0]), [0.5,0,0]);
        tinit = rigid3d(eul2rotm([0,0,0]), [0.5 + 0.01*randn(),0,0]);


        %NDT---------------------------------------------
%         [tform, movingReg, rmse] = pcregisterndt(moving, fixed, gridstep, OutlierRatio=0.3)%, MaxIterations=50); %try messing with OutlierRatio
    
        %------------------------------------------------
        
        %ICP---------------------------------------------
%         [tform,movingReg, rmse] = pcregistericp(moving,fixed, 'metric', 'pointToPoint',MaxIterations=50);
%         [tform,movingReg, rmse] = pcregistericp(moving,fixed, 'metric', 'pointToPlane');
%         [tform,movingReg, rmse] = pcregistericp(moving,fixed, 'metric', 'pointToPoint', "InitialTransform",tinit); %cheating for debug
%         [tform,movingReg, rmse] = pcregistericp(moving,fixed, 'metric', 'pointToPlane', "InitialTransform",tinit, InlierRatio=0.1); %cheating for debug
%         [tform,movingReg, rmse] = pcregistericp(moving,fixed, 'metric', 'pointToPlane', InlierRatio=0.8);
        %------------------------------------------------
        
% %         %LOAM ---------------------------------------------
%         gridStep = 0.5;
% 
%         %convert from "Unorganized" to "Organized" point cloud for LOAM
%         horizontalResolution = 1024;
%         params = lidarParameters('HDL64E', horizontalResolution); %TODO - set for synthetic data
%         moving = pcorganize(moving, params);
%         fixed = pcorganize(fixed, params);
% 
% %         movingPoints = detectLOAMFeatures(moving);
% %         fixedPoints = detectLOAMFeatures(fixed);
% 
%         tform = pcregisterloam(moving,fixed,gridStep);
%         
% %         %--------------------------------------------------


        %LOAM ---------------------------------------------
        gridStep = 1.0;
        
        %convert from "Unorganized" to "Organized" point cloud for LOAM
        horizontalResolution = 1800; %1024;
        verticalResolution = 67.0;
        verticalFov = [2,-24.8];
        % params = lidarParameters('HDL64E', horizontalResolution); %TODO - set for synthetic data
        params = lidarParameters(verticalResolution, verticalFov, horizontalResolution);
        moving = pcorganize(moving, params);
        fixed = pcorganize(fixed, params);
        % [tform, rmse] = pcregisterloam(moving,fixed,gridStep, "MatchingMethod","one-to-many"); %using organized PCs
        
        %using LOAM points~~~~~~~~~~~~~~~~~~~~~
        movingLOAM = detectLOAMFeatures(moving);
        fixedLOAM = detectLOAMFeatures(fixed);
        %downsample to improve registration time (no effect on accuracy)
        fixedLOAM = downsampleLessPlanar(fixedLOAM,gridStep);
        movingLOAM = downsampleLessPlanar(movingLOAM,gridStep);
%         [tform, rmse] = pcregisterloam(movingLOAM,fixedLOAM,"MatchingMethod","one-to-one", InitialTransform=tinit); 
        [tform, rmse] = pcregisterloam(movingLOAM,fixedLOAM,"MatchingMethod","one-to-one", SearchRadius=1.0, InitialTransform=tinit); 
%         [tform, rmse] = pcregisterloam(movingLOAM,fixedLOAM,"MatchingMethod","one-to-one", SearchRadius=1.0); 
        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
           
        %--------------------------------------------------


        
        ans = [tform.Translation, rotm2eul(tform.Rotation)]
        ans_cum = [ans_cum; ans];
    end
end
RMSE = sqrt(mean((ans_cum - [0.5, 0, 0, -0.05, 0, 0]).^2))


%save to file
% fn = "MC_results/traj1_cart_ICP_point2plane_NO_GP.txt";
% fn = "MC_results/traj1_cart_ICP_point2point_NO_GP.txt";
% fn = "MC_results/traj1_spherical_ICP_point2plane_NO_GP.txt";
% fn = "MC_results/traj1_cart_NDT_NO_GP_v2.txt";
% fn = "MC_results/traj1_cart_LOAM_NO_GP.txt";
fn = "MC_results/traj1_cart_LOAM.txt";
writematrix(ans_cum, fn, 'Delimiter', 'tab')
