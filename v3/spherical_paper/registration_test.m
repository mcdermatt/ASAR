%script for loading synthetic point clouds and registering with canned NDT
% solver

clear all
close all

ans_cum = [];

%test using KITTI data
% scan1_fn = "C:/kitti/2011_09_26/2011_09_26_drive_0005_raw/velodyne_points/data/0000000008.txt";
% scan2_fn = "C:/kitti/2011_09_26/2011_09_26_drive_0005_raw/velodyne_points/data/0000000009.txt";

scan1_fn = "MC_trajectories/scene1_scan21.txt";
scan2_fn = "MC_trajectories/scene1_scan22.txt";

scan1 = readmatrix(scan1_fn);
scan2 = readmatrix(scan2_fn);


scan1 = scan1(:,1:3);
scan2 = scan2(:,1:3);

%TODO: need to rotate scan2 relative to scan1 since rotations can't be done
% easily in point cloud generation script
% rotm = eul2rotm([0.05, 0, 0]);
% scan2 = scan2*rotm;

% %remove ground plane--------------------------
% gph = -1.5; %ground plane height for KITTI
% % gph = -1.8; %ground plane height for simulated scene 1
% goodidx1 = find(scan1(:,3)>gph);
% scan1 = scan1(goodidx1, :);
% goodidx2 = find(scan2(:,3)>gph);
% scan2 = scan2(goodidx2, :);
% % groundPtsIdx1 = segmentGroundFromLidarData(moving); %builtin func
% %---------------------------------------------

% %add noise to each PC
% noise_scale = 0.02;
% scan1 = scan1 + noise_scale*randn(size(scan1));
% scan2 = scan2 + noise_scale*randn(size(scan2));

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

gridstep = 1;

% cheat initial transformation estimate
tinit = rigid3d(eul2rotm([-0.05,0,0]), [0.5,0,0]);

%NDT---------------------------------------------
% [tform, movingReg, rmse] = pcregisterndt(moving, fixed, gridstep, OutlierRatio=0.5, MaxIterations=50); %try messing with OutlierRatio

%------------------------------------------------

%ICP---------------------------------------------
% [tform,movingReg, rmse] = pcregistericp(moving,fixed);    
%     [tform,movingReg, rmse] = pcregistericp(moving,fixed, 'metric', 'pointToPlane');
% [tform,movingReg, rmse] = pcregistericp(moving,fixed, 'metric', 'pointToPlane', "InitialTransform",tinit); %cheating for debug
% [tform,movingReg, rmse] = pcregistericp(moving,fixed, 'metric', 'pointToPoint');
%------------------------------------------------

%LOAM ---------------------------------------------
gridStep = 2.0;

%convert from "Unorganized" to "Organized" point cloud for LOAM
horizontalResolution = 1800; %1024;
verticalResolution = 67.0;
verticalFov = [2,-24.8];
% params = lidarParameters('HDL64E', 4000); %for KITTI
params = lidarParameters(verticalResolution, verticalFov, horizontalResolution); %for synthetic data
moving = pcorganize(moving, params);
fixed = pcorganize(fixed, params);
% [tform, rmse] = pcregisterloam(moving,fixed,gridStep, "MatchingMethod","one-to-many"); %using organized PCs

%using LOAM points~~~~~~~~~~~~~~~~~~~~~
movingLOAM = detectLOAMFeatures(moving);
fixedLOAM = detectLOAMFeatures(fixed);
%downsample to improve registration time (no effect on accuracy)
fixedLOAM = downsampleLessPlanar(fixedLOAM,gridStep);
movingLOAM = downsampleLessPlanar(movingLOAM,gridStep);
% [tform, rmse] = pcregisterloam(movingLOAM,fixedLOAM,"MatchingMethod","one-to-many", InitialTransform=tinit); 
[tform, rmse] = pcregisterloam(movingLOAM,fixedLOAM,"MatchingMethod","one-to-one", "SearchRadius",1); 
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




%--------------------------------------------------

figure()
hold on
% pcshow(fixed)
% pcshow(moving)
pcshow(fixedLOAM.Location) %all points 
%TODO: plot just the points Labeled as "sharp corner"
pcshow(movingLOAM.Location)
% pcshowpair(movingLOAM, fixedLOAM)

ans = [tform.Translation, rotm2eul(tform.Rotation)]
