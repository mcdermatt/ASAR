%script for loading synthetic point clouds and registering with canned NDT
% solver

clear all
close all

ans_cum = [];

%raw data
% scan1_fn = "C:/kitti/2011_09_26/2011_09_26_drive_0005_raw/velodyne_points/data/0000000140.txt";
% scan2_fn = "C:/kitti/2011_09_26/2011_09_26_drive_0005_raw/velodyne_points/data/0000000141.txt";
%synced data
scan1_fn = "C:/kitti/2011_09_26/2011_09_26_drive_0005_sync/velodyne_points/data_text/scan115.txt";
scan2_fn = "C:/kitti/2011_09_26/2011_09_26_drive_0005_sync/velodyne_points/data_text/scan116.txt";
scan1 = readmatrix(scan1_fn);
scan2 = readmatrix(scan2_fn);

OXTS_fn = "KITTI_results/OXTS_baseline.txt";
gt = readmatrix(OXTS_fn);


%ignore reflectance data
scan1 = scan1(:,1:3);
scan2 = scan2(:,1:3);

% %remove ground plane--------------------------
% % gph = -0.5; %ground plane height
% goodidx1 = find(scan1(:,3)>-1.5);
% scan1 = scan1(goodidx1, :);
% goodidx2 = find(scan2(:,3)>-1.5);
% scan2 = scan2(goodidx2, :);
% % groundPtsIdx1 = segmentGroundFromLidarData(moving); %builtin func
% %---------------------------------------------

%add noise to each PC
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


% cheat initial transformation estimate
tinit = rigid3d(eul2rotm([0,0,0]), [gt(115,1) + 0.05*randn(),0,0]);

%NDT---------------------------------------------
% gridstep = 1.0;
% [tform, movingReg, rmse] = pcregisterndt(moving, fixed, gridstep, OutlierRatio=0.5, MaxIterations=50); %try messing with OutlierRatio

%------------------------------------------------

%ICP---------------------------------------------
% [tform,movingReg, rmse] = pcregistericp(moving,fixed);    
% [tform,movingReg, rmse] = pcregistericp(moving,fixed, 'metric', 'pointToPlane');
% [tform,movingReg, rmse] = pcregistericp(moving,fixed, 'metric', 'pointToPlane', "InitialTransform",tinit); %cheating for debug
% [tform,movingReg, rmse] = pcregistericp(moving,fixed, 'metric', 'pointToPoint');
%------------------------------------------------

% %LOAM ---------------------------------------------
gridStep = 1.0;
% gridStep = 0.25;

%convert from "Unorganized" to "Organized" point cloud for LOAM
horizontalResolution = 2000; %4000;
params = lidarParameters('HDL64E', horizontalResolution); %TODO - debug value for horizontal resolution
moving = pcorganize(moving, params);
fixed = pcorganize(fixed, params);

%using organized PCs
% [tform, rmse] = pcregisterloam(moving,fixed,gridStep, "MatchingMethod","one-to-many"); 

% %using LOAM points~~~~~~~~~~~~~~~~~~~~~
movingLOAM = detectLOAMFeatures(moving);
fixedLOAM = detectLOAMFeatures(fixed);
fixedLOAM = downsampleLessPlanar(fixedLOAM,gridStep);
movingLOAM = downsampleLessPlanar(movingLOAM,gridStep);
[tform, rmse] = pcregisterloam(movingLOAM,fixedLOAM,"MatchingMethod","one-to-many", InitialTransform=tinit); %works best so far
% [tform, rmse] = pcregisterloam(movingLOAM,fixedLOAM,"MatchingMethod","one-to-one", InitialTransform=tinit); %test

% %--------------------------------------------------

figure()
hold on
%full PCs---------------
% pcshow(fixed)
% % pcshow(moving)
% ptCloudOut = pctransform(moving, tform);
% pcshow(ptCloudOut)
%-----------------------

pcshow(fixedLOAM.Location)
pcshow(movingLOAM.Location)

ans = [tform.Translation, rotm2eul(tform.Rotation)]
