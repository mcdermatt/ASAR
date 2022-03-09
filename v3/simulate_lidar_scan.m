%script for simulating lidar scans of input base stl scene (generated using
%autodesk inventor)
clear all 
close all

%import stl
FileName = 'virtual_scenes/scene1.stl'; %easy scan with house 
% FileName = 'virtual_scenes/scene2.stl'; %more difficult scan
OpenFile = stlread(FileName);

%get vertices, faces, and normals from stl
vertices = OpenFile.Points;
faces = OpenFile.ConnectivityList;

%generate extended object mesh
mesh = extendedObjectMesh(vertices,faces);
%rotate mesh to correct orientation
mesh = rotate(mesh, [0, 0, 90]);

%init lidar unit
SensorIndex = 1;
sensor = monostaticLidarSensor(SensorIndex);

% set parameters of virtual lidar unit to match velodyne VLP-16
sensor.UpdateRate = 10;
sensor.ElevationLimits = [-15, 15];

% Create a tracking scenario. Add an ego platform and a target platform.
scenario = trackingScenario;
ego = platform(scenario, 'Position', [0, 0, 1.72]);
% ego.Position = [0, 0, 1.72];
target = platform(scenario,'Trajectory',kinematicTrajectory('Position',[10 -3 0],'Velocity',[5 0 0]));

target.Mesh = mesh;
target.Dimensions.Length = 100; 
target.Dimensions.Width = 100;
% target.Dimensions.Height = 12;
target.Dimensions.Height = 32;

show(target.Mesh)

% Obtain the mesh of the target viewed from the ego platform after advancing the scenario one step forward.
advance(scenario);
tgtmeshes = targetMeshes(ego);

% Use the created sensor to generate point clouds from the obtained target mesh.
time = scenario.SimulationTime;
[ptCloud, config, clusters] = sensor(tgtmeshes, time);

figure()
% hold on
axis equal
plot3(ptCloud(:,1),ptCloud(:,2),ptCloud(:,3),'o')

%remove all NaNs
ptCloud = rmmissing(ptCloud);
% ptCloud = cast(ptCloud, single);

writematrix(ptCloud, "scene1_scan1.txt", 'Delimiter', 'tab')
% csvwrite("scene1_scan1.csv", ptCloud)