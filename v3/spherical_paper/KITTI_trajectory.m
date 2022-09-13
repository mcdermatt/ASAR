%script for loading synthetic point clouds and registering with canned NDT
% solver

clear all
close all

ans_cum = [];
ans = [0,0,0,0,0,0];

OXTS_fn = "KITTI_results/OXTS_baseline.txt";
gt = readmatrix(OXTS_fn);

for i = 1:150%4499 %150
    i

    %raw data
    scan1_fn = "C:/kitti/2011_09_26/2011_09_26_drive_0005_raw/velodyne_points/data/" + sprintf( '%010d', i-1 ) + ".txt";
    scan2_fn = "C:/kitti/2011_09_26/2011_09_26_drive_0005_raw/velodyne_points/data/" + sprintf( '%010d', i ) + ".txt";

    %rectified data
%     scan1_fn = "C:/kitti/2011_09_26/2011_09_26_drive_0005_sync/velodyne_points/data_text/scan" + string(i -1) + ".txt";
%     scan2_fn = "C:/kitti/2011_09_26/2011_09_26_drive_0005_sync/velodyne_points/data_text/scan" + string(i) + ".txt";

%     %4500 frame MEGA KITTI
%     scan1_fn = "E:/KITTI/drive_00_text/scan"+ string(i -1) + ".txt";
%     scan2_fn = "E:/KITTI/drive_00_text/scan"+ string(i) + ".txt";

    scan1 = readmatrix(scan1_fn);
    scan2 = readmatrix(scan2_fn);

    %ignore reflectance data
    scan1 = scan1(:,1:3);
    scan2 = scan2(:,1:3);
    
%     %remove ground plane--------------------------
%     gph = -1.5; %ground plane height for KITTI  
% %     gph = -1.8; %ground plane height for simulated scene 1
%     goodidx1 = find(scan1(:,3)>gph);
%     scan1 = scan1(goodidx1, :);
%     goodidx2 = find(scan2(:,3)>gph);
%     scan2 = scan2(goodidx2, :);
%     % groundPtsIdx1 = segmentGroundFromLidarData(moving); %builtin func
%     %---------------------------------------------
    
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
    tinit = rigid3d(eul2rotm([0,0,0]), [ans(1),0,0]);
%     tinit = rigid3d(eul2rotm([0,0,0]), [0.5 + 0.01*randn(),0,0]);
%     tinit = rigid3d(eul2rotm([0,0,0]), [gt(i,1) + 0.05*randn(), 0, 0]); %seed solution with correct soln + some noise

%     %NDT---------------------------------------------
    [tform, movingReg, rmse] = pcregisterndt(moving, fixed, gridstep, OutlierRatio=0.3, Tolerance=[0.001, 0.05], MaxIterations=50); %try messing with OutlierRatio
% 
%     %------------------------------------------------
    
    %ICP---------------------------------------------
%         [tform,movingReg, rmse] = pcregistericp(moving,fixed, 'metric', 'pointToPoint',MaxIterations=50);
%         [tform,movingReg, rmse] = pcregistericp(moving,fixed, 'metric', 'pointToPlane');
%         [tform,movingReg, rmse] = pcregistericp(moving,fixed, 'metric', 'pointToPoint', "InitialTransform",tinit); %cheating for debug
%         [tform,movingReg, rmse] = pcregistericp(moving,fixed, 'metric', 'pointToPlane', "InitialTransform",tinit, InlierRatio=0.1); %cheating for debug
%         [tform,movingReg, rmse] = pcregistericp(moving,fixed, 'metric', 'pointToPlane', InlierRatio=0.8);
%     [tform,movingReg, rmse] = pcregistericp(moving,fixed, 'metric', 'pointToPlane', "InitialTransform",tinit);

    %GP-ICP~~~~~~~~~~~~~~~~~~~~~~~~~~
%     % fit ground planes for each scan
%     maxDistance = 0.2;
%     referenceVector = [0,0,1];
%     maxAngularDistance = 5;
%     [model1,inlierIndices1,outlierIndices1] = pcfitplane(fixed, maxDistance, referenceVector, maxAngularDistance);
%     plane1 = select(fixed, inlierIndices1); %subset of points in 1st scan on ground
%     [model2,inlierIndices2,outlierIndices2] = pcfitplane(moving, maxDistance, referenceVector, maxAngularDistance);
%     plane2 = select(moving, inlierIndices2); %subset of points in 2nd scan on ground
%     figure()
%     hold on
%     pcshow(plane1)
%     pcshow(plane2)
%     %register ground planes
%     [tform_gp,movingReg_gp, rmse_gp] = pcregistericp(plane1, plane2, 'metric', 'pointToPlane', InitialTransform=tinit);
%     tform_gp
%     
%     %apply transformation in (z, pitch, roll) from ground plane registration to entire scans
%     moving = pctransform(select(moving, outlierIndices2) , tform_gp);
%     fixed = select(fixed, outlierIndices1);
%     %only consider points not on ground plane
%     [tform,movingReg, rmse] = pcregistericp(moving,fixed, 'metric', 'pointToPlane', InitialTransform=tinit);
%     tform.Translation = tform.Translation + tform_gp.Translation;
%     tform.Rotation = tform_gp.Rotation * tform.Rotation;
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


% %     LOAM ---------------------------------------------
% %     gridStep = 1.0;
%     gridStep = 3.0;
% 
%     
%     %convert from "Unorganized" to "Organized" point cloud for LOAM
%     horizontalResolution = 2000; %4000;
%     params = lidarParameters('HDL64E', horizontalResolution); %TODO - debug value for horizontal resolution
%     moving = pcorganize(moving, params);
%     fixed = pcorganize(fixed, params);
%     
%     %using organized PCs
%     [tform, rmse] = pcregisterloam(moving,fixed,gridStep, "MatchingMethod","one-to-many", tolerance = [0.001, 0.05]); 
%     
% %     %using LOAM points~~~~~~~~~~~~~~~~~~~~~
% %     movingLOAM = detectLOAMFeatures(moving);
% %     fixedLOAM = detectLOAMFeatures(fixed);
% %     fixedLOAM = downsampleLessPlanar(fixedLOAM,gridStep);
% %     movingLOAM = downsampleLessPlanar(movingLOAM,gridStep);
% %     [tform, rmse] = pcregisterloam(movingLOAM,fixedLOAM,"MatchingMethod","one-to-one"); 
% % %     [tform, rmse] = pcregisterloam(movingLOAM,fixedLOAM,"MatchingMethod","one-to-one", InitialTransform=tinit); 
%     %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%     
% %     --------------------------------------------------

    
    ans = [tform.Translation, rotm2eul(tform.Rotation)]
    ans_cum = [ans_cum; ans];
end

%save to file
% fn = "KITTI_results/KITTI_LOAM_v2.txt";
% fn = "KITTI_results/KITTI_LOAM_noGP.txt";
% fn = "KITTI_results/KITTI_ICP.txt";
% fn = "KITTI_results/KITTI_LOAM_v3.txt";
fn = "KITTI_results/NDT_test.txt";
% fn = "KITTI_full_results/KITTI_estimates_LOAM.txt";
% fn = "KITTI_full_results/KITTI_estimates_ICP_v2.txt";
writematrix(ans_cum, fn, 'Delimiter', 'tab')
