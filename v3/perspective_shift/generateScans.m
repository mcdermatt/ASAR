%script for simulating lidar scans of input base stl scene (generated using
%autodesk inventor)

%TODO: adjust start of bins depending on how objects in the scene are
%      intially rotated

%TODO: generate ground truth and save all to file

%TODO: add main loop, how much training data do we need??

clear all 
close all

sam1_cum = [];
sam2_cum = [];

%import stl
FileName = 'training_data/simple_ring1.stl';
OpenFile = stlread(FileName);

%get vertices, faces, and normals from stl
vertices = OpenFile.Points;
faces = OpenFile.ConnectivityList;

%generate extended object mesh
mesh = extendedObjectMesh(vertices,faces);
%rotate mesh to correct orientation
mesh = rotate(mesh, [0, 0, 90]); %[yaw, ?, ?]

%init lidar unit
SensorIndex = 1;
sensor = monostaticLidarSensor(SensorIndex);

% set parameters of virtual lidar unit to match velodyne VLP-16
sensor.UpdateRate = 10;
sensor.ElevationLimits = [-22, 2]; %[-24.8, 2];
sensor.RangeAccuracy = 0.0001; %0.03; %0.01;
sensor.AzimuthResolution = 0.2; %0.08;
sensor.ElevationResolution = 0.4;
% sensor.MaxRange = 50;


% Create a tracking scenario. Add an ego platform and a target platform.
scenario = trackingScenario;
ego = platform(scenario, 'Position', [0, 0, 1.72]);
% ego.Position = [0, 0, 1.72];
target = platform(scenario,'Trajectory',kinematicTrajectory('Position',[0 0 0],'Velocity',[25 0 0])); %no rotation
% target = platform(scenario,'Trajectory',kinematicTrajectory('Position',[10 0 0],'Velocity',[5 0 0], 'AngularVelocity', [0, 0, 0.1])); %with rotatation

target.Mesh = mesh;
target.Dimensions.Length = 100; 
target.Dimensions.Width = 100;
% target.Dimensions.Height = 5; %difficult scan
target.Dimensions.Height = 18;

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

figure()
axis equal
plot3(ptCloud1(:,1),ptCloud1(:,2),ptCloud1(:,3),'.')
hold on
plot3(ptCloud2(:,1),ptCloud2(:,2),ptCloud2(:,3),'.')

%remove all NaNss
ptCloud1 = rmmissing(ptCloud1);
ptCloud2 = rmmissing(ptCloud2);

[azimuth1, elevation1, r1] = cart2sph(ptCloud1(:,1), ptCloud1(:,2), ptCloud1(:,3));
[azimuth2, elevation2, r2] = cart2sph(ptCloud2(:,1), ptCloud2(:,2), ptCloud2(:,3));

%loop through distinct objects captured in frame
bins = -pi:(2*pi/16):pi;
nSamples = 25;
for i = 1:16
    %use azimuth angle to bin points into unique objects
    idx1 = find(azimuth1 > bins(i) & azimuth1 < bins(i+1));
    idx2 = find(azimuth2 > bins(i) & azimuth2 < bins(i+1));
    
    object1_full = ptCloud1(idx1, :);
    object2_full = ptCloud2(idx2, :);

    sam1 = object1_full(uint16(ceil(size(object1_full, 1)*rand(nSamples,1))), :);
    sam2 = object2_full(uint16(ceil(size(object2_full, 1)*rand(nSamples,1))), :);

    sam1_cum = [sam1_cum; sam1];
    sam2_cum = [sam2_cum; sam2];


end

figure()
hold on
scatter3(sam1_cum(:,1), sam1_cum(:,2), sam1_cum(:,3))
scatter3(sam2_cum(:,1), sam2_cum(:,2), sam2_cum(:,3))
% scatter3(object1_full(:,1), object1_full(:,2), object1_full(:,3))
% scatter3(sam1(:,1), sam1(:,2), sam1(:,3))

% Y = discretize(azimuth, -pi:2*pi/16:pi)

% writematrix(ptCloud1, "training_data/scan1.txt", 'Delimiter', 'tab')
% writematrix(ptCloud2, "training_data/scan2.txt", 'Delimiter', 'tab')