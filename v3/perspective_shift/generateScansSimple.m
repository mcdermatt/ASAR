%script for simulating lidar scans of input base stl scene (generated using
%autodesk inventor)

%USES INDIVIDUAL OBJECT FILES

clear all 
close all

nSamples = 25; %50
epochs = 1000;

% sam1_cum = [];
% sam2_cum = [];
% truth_cum = [];

sam1_cum = zeros(epochs*nSamples, 3);
sam2_cum = zeros(epochs*nSamples, 3);
truth_cum = zeros(epochs, 3);

for e = 1:epochs

    e

    %import stl
    roll = rand();
    if roll < 1/2
        FileName = 'training_data/simple_object1.stl';
    end
    if roll < 2/2 & roll > 1/2
        FileName = 'training_data/simple_object2.stl';
    end
%     if roll < 3/5 & roll > 2/5
%         FileName = 'training_data/simple_ring3.stl';
%     end
%     if roll > 3/5 & roll < 4/5
%         FileName = 'training_data/simple_ring4.stl';
%     end
    OpenFile = stlread(FileName);
    
    %get vertices, faces, and normals from stl
    vertices = OpenFile.Points;
    faces = OpenFile.ConnectivityList;
    
    vel = [5*randn() 5*randn() 0.01*randn()];
    pos = [1*randn() 1*randn 0];
%     rot = 2*(ceil(16*rand(1)))/16*pi;
%     scale = 100 + 20*randn();
    rot = 2*pi*randn();
    scale = 20 + 2*randn();


    %generate extended object mesh
    mesh = extendedObjectMesh(vertices,faces);
    %rotate mesh to correct orientation
    mesh = rotate(mesh, [rad2deg(rot), 0, 90]); %[yaw, ?, ?]
    
    %init lidar unit
    SensorIndex = 1;
    sensor = monostaticLidarSensor(SensorIndex);
    
    % set parameters of virtual lidar unit to match velodyne VLP-16
    sensor.UpdateRate = 10;
    sensor.ElevationLimits = [-22, 2]; %[-24.8, 2];
    sensor.RangeAccuracy = 0.01; %0.03; %0.01;
    sensor.AzimuthResolution = 0.2; %0.08;
    sensor.ElevationResolution = 0.4;
    sensor.MaxRange = 9;
    
    
    % Create a tracking scenario. Add an ego platform and a target platform.
    scenario = trackingScenario;
    ego = platform(scenario, 'Position', [0, 0, 1.72]);
    % ego.Position = [0, 0, 1.72];
    target = platform(scenario,'Trajectory',kinematicTrajectory('Position', pos,'Velocity', vel)); %no rotation
    % target = platform(scenario,'Trajectory',kinematicTrajectory('Position',[10 0 0],'Velocity',[5 0 0], 'AngularVelocity', [0, 0, 0.1])); %with rotatation
    
    target.Mesh = mesh;
    target.Dimensions.Length = scale; 
    target.Dimensions.Width = scale;
    target.Dimensions.Height = 18;
    
%     show(target.Mesh)
    
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
    
    [azimuth1, elevation1, r1] = cart2sph(ptCloud1(:,1), ptCloud1(:,2), ptCloud1(:,3));
    [azimuth2, elevation2, r2] = cart2sph(ptCloud2(:,1), ptCloud2(:,2), ptCloud2(:,3));
    
    %loop through distinct objects captured in frame
    bins = -pi:(2*pi/16):pi;
    
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


% scatter3(object1_full(:,1), object1_full(:,2), object1_full(:,3))
scatter3(sam1(:,1), sam1(:,2), sam1(:,3))
scatter3(sam2(:,1), sam2(:,2), sam2(:,3))

%for smaller datasets (keep in git repo)
writematrix(sam1_cum, "training_data/scan1.txt", 'Delimiter', 'tab')
writematrix(sam2_cum, "training_data/scan2.txt", 'Delimiter', 'tab')
writematrix(truth_cum, "training_data/ground_truth.txt", 'Delimiter', 'tab')