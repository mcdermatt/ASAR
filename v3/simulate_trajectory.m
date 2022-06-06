%script for simulating lidar scans of input base stl scene (generated using
%autodesk inventor)
clear all 
close all

runLen = 41;
scene = 1;

%import stl
if scene == 1
    FileName = 'virtual_scenes/scene2.stl';         %scene 1
else
    FileName = 'virtual_scenes/mountain_simple.stl'; %scene 2 
end

OpenFile = stlread(FileName);

%get vertices, faces, and normals from stl
vertices = OpenFile.Points;
faces = OpenFile.ConnectivityList;

%generate extended object mesh
mesh = extendedObjectMesh(vertices,faces);
%rotate mesh to correct orientation
mesh = rotate(mesh, [0, 0, 90]); %else
% mesh = rotate(mesh, [270, 0, 90]); %for room


%init lidar unit
SensorIndex = 1;
sensor = monostaticLidarSensor(SensorIndex);
sensor.MountingLocation = [0, 0, 0]; %AHHHHAHHHHAHHHH!!!! Why is this not default zero!!??!??!??!

% set parameters of virtual lidar unit to match velodyne VLP-16
sensor.UpdateRate = 10;
sensor.ElevationLimits =  [-24.8, 2];  % was [-22, 2]; %22
sensor.RangeAccuracy = 0.0001; %0.03; %0.01;
sensor.AzimuthResolution = 0.2; %0.08;
sensor.ElevationResolution = 0.4;
% sensor.MaxRange = 50;


% Create a tracking scenario. Add an ego platform and a target platform.
scenario = trackingScenario;

if scene == 1
%     ego = platform(scenario, 'Trajectory', kinematicTrajectory('Position',[-10,0,3],'Velocity',[5 0 0], 'AngularVelocity', [0, 0, 0.1], 'Acceleration', [0, 0, 0])); 
    ego = platform(scenario,'Position',[0, 0, 0]);
    target = platform(scenario, 'Trajectory', kinematicTrajectory('Position',[10,0,-2],'Velocity',[-5 0 0], 'AngularVelocity', [0, 0, 0]));
    % target = platform(scenario,'Position',[0, 0, 0]); %was this

end

if scene == 2
%     ego = platform(scenario, 'Trajectory', kinematicTrajectory('Position',[-10,0,3],'Velocity',[5 0 0], 'AngularVelocity', [0, 0, 0])); %was this
    ego = platform(scenario,'Position',[0, 0, 0]);
    target = platform(scenario, 'Trajectory', kinematicTrajectory('Position',[10,0,-3],'Velocity',[-5 0 0], 'AngularVelocity', [0, 0, 0]));
    % target = platform(scenario,'Position',[0, 0, 0]); %was this
end


target.Mesh = mesh;
% show(target.Mesh)

%default~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
target.Dimensions.Length = 100; 
target.Dimensions.Width = 100;
target.Dimensions.Height = 20; %6; %18;
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%height = 20 for with trees, 

figure()
axis equal
hold on

for idx = 1:runLen
    advance(scenario);
    % Obtain the mesh of the target viewed from the ego platform after advancing the scenario one step forward.
    tgtmeshes = targetMeshes(ego);
    
    % Use the created sensor to generate point clouds from the obtained target mesh.
    time = scenario.SimulationTime;
    [ptCloud, config, clusters] = sensor(tgtmeshes, time);
    
    %remove all NaNs
    ptCloud = rmmissing(ptCloud);
    
    %transform all points back into frame of vehicle (global enu -> body frame xyz)
    %NOTE -> better to do this inside python script (rotation is all
    %relative since we have a 360deg view, important part is keeping
    %translation of vehicle aligned with PC xyz axis...)
%     rot = eul2rotm([-(idx)*0.1, 0, 0]);
%     ptCloud = ptCloud*rot;

    %save to file
    fn = "MC_trajectories/scene" + scene + "_scan" + idx + ".txt";
    writematrix(ptCloud, fn, 'Delimiter', 'tab')
    
    %add to plot
    plot3(ptCloud(:,1),ptCloud(:,2),ptCloud(:,3),'.')
end

