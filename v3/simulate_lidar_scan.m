%script for simulating lidar scans of input base stl scene (generated using
%autodesk inventor)
clear all 
close all

%import stl
% FileName = 'virtual_scenes/scene1.stl'; %easy scan with house 
% FileName = 'virtual_scenes/scene2.stl'; %more difficult scan
% FileName = 'virtual_scenes/scene3.stl'; % rectangles w/ occlusion
% FileName = 'virtual_scenes/scene4.stl'; % cylinders w/ occlusion
% FileName = 'virtual_scenes/scene5.stl'; % rectangles w/0 occlusion
% FileName = 'virtual_scenes/scene6.stl'; % cylinders w/0 occlusion
% FileName = 'virtual_scenes/scene7.stl'; % skinny cylinders w/0 occlusion
% FileName = 'virtual_scenes/large_room1.stl'; % no back wall
% FileName = 'virtual_scenes/large_room2.stl'; % with back wall
FileName = 'virtual_scenes/verify_geometry.stl'; %use with /perspective_shift/geometry ipynb
OpenFile = stlread(FileName);

%get vertices, faces, and normals from stl
vertices = OpenFile.Points;
faces = OpenFile.ConnectivityList;

%generate extended object mesh
mesh = extendedObjectMesh(vertices,faces);
%rotate mesh to correct orientation
% mesh = rotate(mesh, [0, 0, 90]); %else
mesh = rotate(mesh, [270, 0, 90]); %for room


%init lidar unit
SensorIndex = 1;
sensor = monostaticLidarSensor(SensorIndex);
sensor.MountingLocation = [0, 0, 0]; %AHHHHAHHHHAHHHH!!!! Why is this not default zero!!??!??!??!

% set parameters of virtual lidar unit to match velodyne VLP-16
sensor.UpdateRate = 10;
sensor.ElevationLimits = [-22, 2]; %[-24.8, 2]; %22
sensor.RangeAccuracy = 0.0001; %0.03; %0.01;
sensor.AzimuthResolution = 0.2; %0.08;
sensor.ElevationResolution = 0.4;
% sensor.MaxRange = 50;


% Create a tracking scenario. Add an ego platform and a target platform.
scenario = trackingScenario;
ego = platform(scenario, 'Position', [0, 0, 1.72]);
% ego.Position = [0, 0, 1.72];
target = platform(scenario,'Trajectory',kinematicTrajectory('Position',[0 0 0],'Velocity',[0 5 0])); %no rotation
% target = platform(scenario,'Trajectory',kinematicTrajectory('Position',[-0.3 0 0],'Velocity',[-7 0 0], 'AngularVelocity', [0, 0, 0.1])); %with rotatation
target.Mesh = mesh;

%default~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
target.Dimensions.Length = 100; 
target.Dimensions.Width = 100;
target.Dimensions.Height = 6 %18;
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% %need to scale differently for simple room ~~~~~~~
% target.Dimensions.Length = 50; 
% target.Dimensions.Width = 40;
% target.Dimensions.Height = 4;
% %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% show(target.Mesh)

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
% 
% writematrix(ptCloud1, "scene2_scan1.txt", 'Delimiter', 'tab')
% writematrix(ptCloud2, "scene2_scan2.txt", 'Delimiter', 'tab')
% writematrix(ptCloud1, "scene3_scan1.txt", 'Delimiter', 'tab')
% writematrix(ptCloud2, "scene3_scan2.txt", 'Delimiter', 'tab')
% writematrix(ptCloud1, "scene4_scan1.txt", 'Delimiter', 'tab')
% writematrix(ptCloud2, "scene4_scan2.txt", 'Delimiter', 'tab')
% writematrix(ptCloud1, "simple_room_scan1.txt", 'Delimiter', 'tab')
% writematrix(ptCloud2, "simple_room_scan2.txt", 'Delimiter', 'tab')
writematrix(ptCloud1, "verify_geometry_scan1.txt", 'Delimiter', 'tab')
writematrix(ptCloud2, "verify_geometry_scan2.txt", 'Delimiter', 'tab')

