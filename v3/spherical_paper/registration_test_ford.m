%script for loading synthetic point clouds and registering with canned NDT
% solver

clear all
close all

ans_cum = [];

% scan1_fn = "E:/Ford/IJRR-Dataset-1/SCANS/Scan0075.mat";
% scan2_fn = "E:/Ford/IJRR-Dataset-1/SCANS/Scan0076.mat";
frame = 1250;
scan1_fn = "E:/Ford/IJRR-Dataset-1/SCANS/Scan" + sprintf( '%04d', frame + 75) + ".mat";
scan2_fn = "E:/Ford/IJRR-Dataset-1/SCANS/Scan" + sprintf( '%04d', frame + 76 ) + ".mat";
s1 = load(scan1_fn);
s2 = load(scan2_fn);

gt_fn = "ford_results/truth_body_frame.txt";
gt = readmatrix(gt_fn);


%ignore reflectance data
scan1 = s1.SCAN.XYZ.';
scan2 = s2.SCAN.XYZ.';

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
offset = 0.1*randn();
tinit = rigid3d(eul2rotm([0,0,0]), [0 , 0.1*gt(frame,2) + offset, 0]);

%NDT---------------------------------------------
% gridstep = 1.0;
% % [tform, movingReg, rmse] = pcregisterndt(moving, fixed, gridstep, OutlierRatio=0.5, MaxIterations=50); %try messing with OutlierRatio
% [tform, movingReg, rmse] = pcregisterndt(moving, fixed, gridstep, InitialTransform=tinit);
%------------------------------------------------

%ICP---------------------------------------------
% [tform,movingReg, rmse] = pcregistericp(moving,fixed);    
% [tform,movingReg, rmse] = pcregistericp(moving,fixed, 'metric', 'pointToPlane');
% [tform,movingReg, rmse] = pcregistericp(moving,fixed, 'metric', 'pointToPlane', "InitialTransform",tinit); %cheating for debug
% [tform,movingReg, rmse] = pcregistericp(moving,fixed, 'metric', 'pointToPoint');

% % %GP-ICP~~~~~~~~~~~~~~~~~~~~~~~~~~
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
% tform.Translation = tform.Translation + tform_gp.Translation;
% tform.Rotation = tform_gp.Rotation * tform.Rotation;
% tform.Translation = tform.Translation + tform_gp.Translation;
% tform.Rotation = tform_gp.Rotation * tform.Rotation;
% % %------------------------------------------------

% % %LOAM ---------------------------------------------
% gridStep = 1.0;
% % gridStep = 0.25;
% 
% %convert from "Unorganized" to "Organized" point cloud for LOAM
% horizontalResolution = 2000; %4000;
% params = lidarParameters('HDL64E', horizontalResolution); %TODO - debug value for horizontal resolution
% moving = pcorganize(moving, params);
% fixed = pcorganize(fixed, params);
% 
% %using organized PCs
% % [tform, rmse] = pcregisterloam(moving,fixed,gridStep, "MatchingMethod","one-to-many"); 
% 
% % %using LOAM points~~~~~~~~~~~~~~~~~~~~~
% movingLOAM = detectLOAMFeatures(moving);
% fixedLOAM = detectLOAMFeatures(fixed);
% % fixedLOAM = downsampleLessPlanar(fixedLOAM,gridStep);
% % movingLOAM = downsampleLessPlanar(movingLOAM,gridStep);
% [tform, rmse] = pcregisterloam(movingLOAM,fixedLOAM,"MatchingMethod","one-to-many", InitialTransform=tinit, ...
%     SearchRadius=1, MaxIterations = 50, Tolerance=[0.001, 0.001]); %works best so far
% % [tform, rmse] = pcregisterloam(movingLOAM,fixedLOAM,"MatchingMethod","one-to-one", InitialTransform=tinit); %test
% % %--------------------------------------------------

% figure()
% hold on
% %full PCs---------------
% pcshow(fixed)
% % pcshow(moving)
% ptCloudOut = pctransform(moving, tform);
% pcshow(ptCloudOut)
% %-----------------------

% pcshow(fixedLOAM.Location)
% pcshow(movingLOAM.Location)

0.1*gt(frame+1,:)
ans = [tform.Translation, rotm2eul(tform.Rotation)]
