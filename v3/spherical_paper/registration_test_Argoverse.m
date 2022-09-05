%script for loading synthetic point clouds and registering with canned NDT
% solver

clear all
close all

ans_cum = [];


gt_lidar = readmatrix("Argoverse_results/urban_canyon/gt_lidar.txt");
gt = diff(gt_lidar(:,1:2));
gt = sqrt(sum(gt.^2,2)); %#absolute movement per frame in horizontal plane

% 087fec73-1a0c-399a-9292-cc2cf99dc97f
scan1_fn = "D:/argoverse_benchmarks/urban_canyon/scan85.txt"; %works on 10, fails on 0
scan2_fn = "D:/argoverse_benchmarks/urban_canyon/scan86.txt";

scan1 = readmatrix(scan1_fn);
scan2 = readmatrix(scan2_fn);

% OXTS_fn = "KITTI_results/OXTS_baseline.txt";
% gt = readmatrix(OXTS_fn);


%ignore reflectance data
scan1 = scan1(:,1:3);
scan2 = scan2(:,1:3);

%remove ground plane--------------------------
gph = 0.3; %ground plane height
goodidx1 = find(scan1(:,3)>gph);
scan1 = scan1(goodidx1, :);
goodidx2 = find(scan2(:,3)>gph);
scan2 = scan2(goodidx2, :);
% groundPtsIdx1 = segmentGroundFromLidarData(moving); %builtin func
%---------------------------------------------

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
offset = 0.2*randn();
tinit = rigid3d(eul2rotm([0,0,0]), [gt(85) + offset,0,0]);

% %NDT---------------------------------------------
% gridstep = 2.0;
% [tform, movingReg, rmse] = pcregisterndt(moving, fixed, gridstep, OutlierRatio=0.5, MaxIterations=50); %try messing with OutlierRatio
% % % [tform, movingReg, rmse] = pcregisterndt(moving, fixed, gridstep);
% %------------------------------------------------

% %ICP---------------------------------------------
% % [tform,movingReg, rmse] = pcregistericp(moving,fixed);    
% % [tform,movingReg, rmse] = pcregistericp(moving,fixed, 'metric', 'pointToPlane');
% [tform,movingReg, rmse] = pcregistericp(moving,fixed, 'metric', 'pointToPlane', "InitialTransform",tinit); %cheating for debug
% % [tform,movingReg, rmse] = pcregistericp(moving,fixed, 'metric', 'pointToPoint');
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
% tform_gp
% 
% %apply transformation in (z, pitch, roll) from ground plane registration to entire scans
% moving = pctransform(select(moving, outlierIndices2) , tform_gp);
% fixed = select(fixed, outlierIndices1);
% %only consider points not on ground plane
% [tform,movingReg, rmse] = pcregistericp(moving,fixed, 'metric', 'pointToPlane', InitialTransform=tinit);
% %------------------------------------------------

%LOAM ---------------------------------------------
gridStep = 3.0;
% gridStep = 4.0;

% convert from "Unorganized" to "Organized" point cloud for LOAM
verticalResolution = 64;
horizontalResolution = 1800; %3600; %1800;
verticalFoV = [25, -25];

verticalBeamAngles = flip(sort([linspace(-25,15,32), linspace(-15, 25, 32)]));


% params = lidarParameters('VLP32C', horizontalResolution); 
% params = lidarParameters(verticalResolution,verticalFoV,horizontalResolution); %also incorrect??
params = lidarParameters(verticalBeamAngles, horizontalResolution);
moving = pcorganize(moving, params);
fixed = pcorganize(fixed, params);

%using organized PCs
% [tform, rmse] = pcregisterloam(moving,fixed,gridStep, "MatchingMethod","one-to-many", SearchRadius=10); 

% %using LOAM points~~~~~~~~~~~~~~~~~~~~~
movingLOAM = detectLOAMFeatures(moving);
fixedLOAM = detectLOAMFeatures(fixed);
% fixedLOAM = downsampleLessPlanar(fixedLOAM,gridStep);
% movingLOAM = downsampleLessPlanar(movingLOAM,gridStep);
% [tform, rmse] = pcregisterloam(movingLOAM,fixedLOAM,"MatchingMethod","one-to-many", InitialTransform=tinit); %works best so far
[tform, rmse] = pcregisterloam(movingLOAM,fixedLOAM,"MatchingMethod","one-to-one", InitialTransform=tinit, ...
    verbose = false, tolerance = [0.001, 0.05], MaxIterations=50); %test

% %--------------------------------------------------

figure()
hold on
% % % full PCs---------------
% pcshow(fixed)
% % pcshow(moving)
% ptCloudOut = pctransform(moving, tform);
% pcshow(ptCloudOut)
% % % -----------------------

pcshow(fixedLOAM.Location)
% pcshow(movingLOAM.Location)

gt(86)
ans = [tform.Translation, rotm2eul(tform.Rotation)]

