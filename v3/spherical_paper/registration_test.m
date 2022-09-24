%script for loading synthetic point clouds and registering with canned NDT
% solver

clear all
close all

ans_cum = [];

%test using KITTI data
% scan1_fn = "C:/kitti/2011_09_26/2011_09_26_drive_0005_raw/velodyne_points/data/0000000008.txt";
% scan2_fn = "C:/kitti/2011_09_26/2011_09_26_drive_0005_raw/velodyne_points/data/0000000009.txt";

% scan1_fn = "MC_trajectories/scene1_scan2.txt";
% scan2_fn = "MC_trajectories/scene1_scan3.txt";
scan1_fn = "MC_trajectories/scene1_scan2_v2.txt"; % for debugging sensor parameters (FOV, etc.) TODO: delete v2 files when done...
scan2_fn = "MC_trajectories/scene1_scan3_v2.txt";
% scan1_fn = "unshadowed_points/scene_1_scan_4A_no_shadows.txt";
% scan2_fn = "unshadowed_points/scene_1_scan_5B_no_shadows.txt";    

scan1 = readmatrix(scan1_fn);
scan2 = readmatrix(scan2_fn);


scan1 = scan1(:,1:3);
scan2 = scan2(:,1:3);

%TODO: need to rotate scan2 relative to scan1 since rotations can't be done
% easily in point cloud generation script
rotm = eul2rotm([0.05, 0, 0]);
scan2 = scan2*rotm;
% scan2 = scan2+ [0.5, 0, 0];

% %remove ground plane (simple z test)--------
% gph = -1.5; %ground plane height for KITTI
gph = -1.8; %ground plane height for simulated scene 1
goodidx1 = find(scan1(:,3)>gph);
scan1 = scan1(goodidx1, :);
goodidx2 = find(scan2(:,3)>gph);
scan2 = scan2(goodidx2, :);
% groundPtsIdx1 = segmentGroundFromLidarData(moving); %builtin func
% %---------------------------------------------

% %add noise to each PC
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


% cheat initial transformation estimate
offset = 0.1*randn();
tinit = rigid3d(eul2rotm([0,0,0]), [0.5 + offset,0,0]);

%NDT---------------------------------------------
gridstep = 1;
[tform, movingReg, rmse] = pcregisterndt(moving, fixed, gridstep, OutlierRatio=0.5, MaxIterations=50); %try messing with OutlierRatio
%------------------------------------------------

%ICP---------------------------------------------
% vanilla ICP registration
% [tform,movingReg, rmse] = pcregistericp(moving,fixed);    
% [tform,movingReg, rmse] = pcregistericp(moving,fixed, 'metric', 'pointToPlane');
% [tform,movingReg, rmse] = pcregistericp(moving,fixed, 'metric', 'pointToPlane', "InitialTransform",tinit, InlierRatio=0.95); %cheating for debug
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

%G-ICP ~~~~~~~~~~~~~~~~~~~~~~~~~~
% [qt, mse_profile] = gicp(moving, fixed)

% %GP-ICP~~~~~~~~~~~~~~~~~~~~~~~~~~
% % fit ground planes for each scan
% maxDistance = 0.2;
% referenceVector = [0,0,1];
% maxAngularDistance = 5;
% [model1,inlierIndices1,outlierIndices1] = pcfitplane(fixed, maxDistance, referenceVector, maxAngularDistance);
% plane1 = select(fixed, inlierIndices1); %subset of points in 1st scan on ground
% [model2,inlierIndices2,outlierIndices2] = pcfitplane(moving, maxDistance, referenceVector, maxAngularDistance);
% plane2 = select(moving, inlierIndices2); %subset of points in 2nd scan on ground
% figure()
% hold on
% pcshow(plane1)
% pcshow(plane2)
% %register ground planes
% [tform_gp,movingReg_gp, rmse_gp] = pcregistericp(plane1, plane2, 'metric', 'pointToPlane', InitialTransform=tinit);
% % tform_gp
% 
% %apply transformation in (z, pitch, roll) from ground plane registration to entire scans
% moving = pctransform(select(moving, outlierIndices2) , tform_gp);
% fixed = select(fixed, outlierIndices1);
% %only consider points not on ground plane
% [tform,movingReg, rmse] = pcregistericp(moving,fixed, 'metric', 'pointToPlane', InitialTransform=tinit);
% tform.Translation = tform.Translation + tform_gp.Translation;
% tform.Rotation = tform_gp.Rotation * tform.Rotation;
% %------------------------------------------------

% % %LOAM ---------------------------------------------
% % gridStep = 0.5;
% gridStep = 1.0;
% 
% %convert from "Unorganized" to "Organized" point cloud for LOAM
% horizontalResolution = 1800; %1024;
% verticalResolution = 67.0;
% verticalFov = [2,-24.8];
% % params = lidarParameters('HDL64E', 4000); %for KITTI
% params = lidarParameters(verticalResolution, verticalFov, horizontalResolution); %for synthetic data
% moving = pcorganize(moving, params);
% fixed = pcorganize(fixed, params);
% % [tform, rmse] = pcregisterloam(moving,fixed,gridStep, "MatchingMethod","one-to-many", SearchRadius=3.0, Tolerance=[0.001, 0.05]); %using organized PCs
% 
% % %using LOAM points~~~~~~~~~~~~~~~~~~~~~
% movingLOAM = detectLOAMFeatures(moving);
% fixedLOAM = detectLOAMFeatures(fixed);
% fixedLOAM = downsampleLessPlanar(fixedLOAM,gridStep);
% movingLOAM = downsampleLessPlanar(movingLOAM,gridStep);
% % [tform, rmse] = pcregisterloam(movingLOAM,fixedLOAM,"MatchingMethod","one-to-many", InitialTransform=tinit, Tolerance=[0.001, 0.05], Verbose=false); 
% [tform, rmse] = pcregisterloam(movingLOAM,fixedLOAM,"MatchingMethod","one-to-many", "SearchRadius",3, ...
%     InitialTransform=tinit, Tolerance=[0.001, 0.05], verbose = false, MaxIterations=50); 
% % %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% 
% % %--------------------------------------------------

figure()
hold on

pcshow(fixed)
% pcshow(moving)
tform.Translation
ptCloudOut = pctransform(moving, tform);
pcshow(ptCloudOut)

% figure()
% scatter3(moving.Location(1,:,1), moving.Location(1,:,2), moving.Location(1,:,3))


% pcshow(fixedLOAM.Location) %all points 
%TODO: plot just the points Labeled as "sharp corner"
% pcshow(movingLOAM.Location)
% pcshowpair(movingLOAM, fixedLOAM)

ans = [tform.Translation, rotm2eul(tform.Rotation)]
