%script for simulating lidar scans of input base stl scene (generated using
%autodesk inventor)

%USES INDIVIDUAL OBJECT FILES

%TODO - augment translation by moving PC's generally closer to the correct
% solution - this mimics how the iterative implementation will help solve
% We want to make the basin of attraction as regular as possible

clear all 
close all

nSamples =  7000; %25;
epochs = 1;

% sam1_cum = [];
% sam2_cum = [];
% truth_cum = [];
true_pos1 = [];

sam1_cum = zeros(epochs*nSamples, 3);
sam2_cum = zeros(epochs*nSamples, 3);
truth_cum = zeros(epochs, 3);

for e = 1:epochs
%     e

    %import stl
%     FileName = "C:/Users/Derm/vaRLnt/v3/perspective_shift/figures/wall.stl"; %simple
    FileName = "C:/Users/Derm/vaRLnt/v3/perspective_shift/figures/wall_v2.stl"; %more detailed
    scale = [7, 20, 10];
    rot_corr = [0, 0, 90];
    mindist = 4;
    pos = [-40.0, -40.0, -5.];
    vel = [10*randn() 10*randn() 0];
    true_pos1 = [true_pos1; [pos(1), pos(2), pos(3) - 5.2]];



%     FileName = "C:/Users/Derm/vaRLnt/v3/perspective_shift/figures/Assembly1.stl"; 
%     scale = [50, 50, 3];
%     rot_corr = [0, 0, 90];
%     mindist = 4;
%     pos = [0, 0, -1.0];
%     vel = [100, 0, 0];
%     true_pos1 = [true_pos1; [pos(1), pos(2), pos(3) - scale(3)]];

    %get vertices, faces, and normals from stl
    OpenFile = stlread(FileName);
    vertices = OpenFile.Points;
    faces = OpenFile.ConnectivityList;

%     %test----------
%     fn= 'C:\Users\Derm\Desktop\big\ModelNet10\toilet\train\toilet_0055.off';
%     scale = [1., 1., 1.];
%     rot_corr = [0, 0, 0];
%     mindist = 4;
%     [vertices, faces] = read_off(fn);
%     vertices = vertices.';
%     faces = faces.';
%     %-------------
    
    rot1 = 0; % rad2deg(2*pi*rand());
    rot2 = 0; %rad2deg(2*pi*rand());
    rot3 = 0; %rad2deg(2*pi*rand());
%     vel = [100, 0, 0];
%     rot = 0;    %temp- just for demo dataset
   

%     pos = [2, 1, 0];
%     vel = [0, -10, 0];

    mesh = extendedObjectMesh(vertices,faces);     %generate extended object mesh
    mesh = rotate(mesh, rot_corr);     %rotate mesh to correct orientation (provided by each mesh)

    %init lidar unit
    SensorIndex = 1;
    sensor = monostaticLidarSensor(SensorIndex);
    sensor.MountingLocation = [0, 0, 0]; %AHHHHAHHHHAHHHH!!!! Why is this not default zero!!??!??!??!
    
    % set parameters of virtual lidar unit to match velodyne VLP-16
    sensor.UpdateRate = 10;
    sensor.ElevationLimits = [-35,3]; %[-30, 10];  %[-22, 2]; 
    sensor.RangeAccuracy = 0.01; % was 0.01, now adding noise at end;
    sensor.AzimuthResolution = 0.2; %0.2;
    sensor.ElevationResolution = 0.4; %0.4;
    sensor.MaxRange = 100; %100;
    
    
    % Create a tracking scenario. Add an ego platform and a target platform.
    scenario = trackingScenario;
    ego = platform(scenario, 'Position', [0, 0, 0.2]);
    target = platform(scenario,'Trajectory',kinematicTrajectory('Position', pos,'Velocity', vel, 'Orientation', quat2rotm(eul2quat([rot1, rot2, rot3])) ));

    target.Mesh = mesh;
    target.Dimensions.Length = scale(1); 
    target.Dimensions.Width = scale(2);
    target.Dimensions.Height = scale(3);
%     target.pose();

    show(target.Mesh)
    
    % Obtain the mesh of the target viewed from the ego platform after advancing the scenario one step forward.
    advance(scenario);
    tgtmeshes = targetMeshes(ego);
    % Use the created sensor to generate point clouds from the obtained target mesh.
    time = scenario.SimulationTime;
    [ptCloud1, config, clusters] = sensor(tgtmeshes, time);
    
    %repeat for 2nd scan
    advance(scenario);
    tgtmeshes = targetMeshes(ego);
    time = scenario.SimulationTime;
    [ptCloud2, config, clusters] = sensor(tgtmeshes, time);
    
%     figur-ptCloud2(:,1),ptCloud2(:,2),ptCloud2(:,3),'.')
    
    %remove all NaNss
    ptCloud1 = rmmissing(ptCloud1);
    ptCloud2 = rmmissing(ptCloud2);
    
%     [azimuth1, elevation1, r1] = cart2sph(ptCloud1(:,1), ptCloud1(:,2), ptCloud1(:,3));
%     [azimuth2, elevation2, r2] = cart2sph(ptCloud2(:,1), ptCloud2(:,2), ptCloud2(:,3));
%     
%     %loop through distinct objects captured in frame
%     bins = -pi:(2*pi/16):pi;
    
    sam1 = ptCloud1(uint16(ceil(size(ptCloud1, 1)*rand(nSamples,1))), :);
    sam2 = ptCloud2(uint16(ceil(size(ptCloud2, 1)*rand(nSamples,1))), :);
    
    sam1_cum( (e-1)*nSamples + 1 : (e)*nSamples, : ) = sam1;
    sam2_cum( (e-1)*nSamples  + 1 : (e)*nSamples, : ) = sam2;
    truth_cum((e-1) + 1 :(e),:) = ones(1,3).*vel;

end

figure()
hold on
% scatter3(sam1_cum(:,1), sam1_cum(:,2), sam1_cum(:,3))
% scatter3(sam2_cum(:,1), sam2_cum(:,2), sam2_cum(:,3))

% scatter3(ptCloud1(:,1), ptCloud1(:,2), ptCloud1(:,3))
% scatter3(ptCloud2(:,1), ptCloud2(:,2), ptCloud2(:,3))
scatter3(sam1(:,1), sam1(:,2), sam1(:,3), '.')
% scatter3(sam2(:,1), sam2(:,2), sam2(:,3))
scatter3(sam2(:,1)-0.1*vel(1), sam2(:,2)-0.1*vel(2), sam2(:,3)-0.1*vel(3), '.')
set(gca,'XLim',[-10 10],'YLim',[-10 10],'ZLim',[-10 10])

% % add gaussian noise to all range estimates
% sam1_cum = sam1_cum + 0.02*randn(size(sam1_cum));
% sam2_cum = sam2_cum + 0.02*randn(size(sam2_cum));
 
% % %for smaller datasets (keep in git repo)
% writematrix(sam1_cum, "training_data/scan1.txt", 'Delimiter', 'tab')
% writematrix(sam2_cum, "training_data/scan2.txt", 'Delimiter', 'tab')
% writematrix(truth_cum, "training_data/ground_truth.txt", 'Delimiter', 'tab')
% % writematrix(true_pos1, "training_data/true_pos1.txt", 'Delimiter', 'tab')

%for larger datasets (don't save with git)
% writematrix(sam1_cum, "C:/Users/Derm/Desktop/big/pshift/scan1_1k_50_samples.txt", 'Delimiter', 'tab')
% writematrix(sam2_cum, "C:/Users/Derm/Desktop/big/pshift/scan2_1k_50_samples.txt", 'Delimiter', 'tab')
% writematrix(truth_cum, "C:/Users/Derm/Desktop/big/pshift/ground_truth_1k_50_samples.txt", 'Delimiter', 'tab')

%fig 2 for 3D paper
writematrix(sam1_cum, "C:/Users/Derm/vaRLnt/v3/perspective_shift/figures/s1.txt", 'Delimiter', 'tab')
writematrix(sam2_cum, "C:/Users/Derm/vaRLnt/v3/perspective_shift/figures/s2.txt", 'Delimiter', 'tab')
writematrix([pos(1), pos(2), pos(3) - .2], "C:/Users/Derm/vaRLnt/v3/perspective_shift/figures/gt.txt", 'Delimiter', 'tab')

%human and wall
% writematrix(sam1_cum, "C:/Users/Derm/vaRLnt/v3/perspective_shift/figures/fig1_s1.txt", 'Delimiter', 'tab')
% writematrix(sam2_cum, "C:/Users/Derm/vaRLnt/v3/perspective_shift/figures/fig1_s2.txt", 'Delimiter', 'tab')
% writematrix([vel(1), vel(2), vel(3)], "C:/Users/Derm/vaRLnt/v3/perspective_shift/figures/fig1_gt.txt", 'Delimiter', 'tab')


% %for debug
% figure()
% hold on
% scatter3(sam1(:,1), sam1(:,2), sam1(:,3))
% scatter3(sam2(:,1), sam2(:,2), sam2(:,3))
% scatter3(moved_sam2(:,1), moved_sam2(:,2), moved_sam2(:,3))

% %write scaled translated and rotated figure to new stl file for viz
TR = triangulation( target.Mesh.Faces, target.Mesh.Vertices);
stlwrite(TR, 'wall_scaled.stl')
% stlwrite(TR, 'Assembly1_scaled.stl')
