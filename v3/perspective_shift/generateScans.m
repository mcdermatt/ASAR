%script for simulating lidar scans of input base stl scene (generated using
%autodesk inventor)

%USES RING FILES WITH MULTIPLE OBJECTS

clear all 
close all

nSamples = 25; %50
nObjects = 16; %defined in Inventor file
epochs = 1000;

% sam1_cum = [];
% sam2_cum = [];
% truth_cum = [];

sam1_cum = zeros(epochs*nObjects*nSamples, 3);
sam2_cum = zeros(epochs*nObjects*nSamples, 3);
truth_cum = zeros(nObjects*epochs, 3);

for e = 1:epochs

    e

    %import stl
    roll = rand();
    if roll < 1/5
        FileName = 'training_data/simple_ring1.stl';
    end
    if roll < 2/4 & roll > 1/4
        FileName = 'training_data/simple_ring2.stl';
    end
    if roll < 3/4 & roll > 2/4
        FileName = 'training_data/simple_ring3.stl';
    end
    if roll > 3/4 % & roll < 4/5
        FileName = 'training_data/simple_ring4.stl';
    end

%     %special case: objects close to vehicle
%     if roll > 4/5
%         FileName = 'training_data/simple_ring5.stl';
%     end
    OpenFile = stlread(FileName);
    
    %get vertices, faces, and normals from stl
    vertices = OpenFile.Points;
    faces = OpenFile.ConnectivityList;
    
    vel = [5*randn() 5*randn() 0.01*randn()];
    pos = [0 0 0];
    rot = 2*(ceil(16*rand(1)))/16*pi;
    scale = 100 + 20*randn();
    
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
    % sensor.MaxRange = 50;
    
    
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
    
    for i = 1:nObjects
        %use azimuth angle to bin points into unique objects
        idx1 = find(azimuth1 > bins(i) & azimuth1 < bins(i+1));
        idx2 = find(azimuth2 > bins(i) & azimuth2 < bins(i+1));
        
        object1_full = ptCloud1(idx1, :);
        object2_full = ptCloud2(idx2, :);
    
        sam1 = object1_full(uint16(ceil(size(object1_full, 1)*rand(nSamples,1))), :);
        sam2 = object2_full(uint16(ceil(size(object2_full, 1)*rand(nSamples,1))), :);
    
%         truth_cum = [truth_cum; vel];
%         sam1_cum = [sam1_cum; sam1];
%         sam2_cum = [sam2_cum; sam2];

        sam1_cum( (e-1)*nObjects*nSamples + (i-1)*nSamples + 1 : (e-1)*nObjects*nSamples + i*nSamples, : ) = sam1;
        sam2_cum( (e-1)*nObjects*nSamples + (i-1)*nSamples + 1 : (e-1)*nObjects*nSamples + i*nSamples, : ) = sam2;
%         sam2_cum( (e-1)*nObjects + (i-1)*nSamples + 1 :(e-1)*nObjects + i*nSamples, : ) = sam2;

    end
    truth_cum((e-1)*nObjects + 1 :(e)*nObjects,:) = ones(nObjects,3).*vel;

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

%for larger datasets (don't save with git)
% writematrix(sam1_cum, "C:/Users/Derm/Desktop/big/pshift/scan1_50k.txt", 'Delimiter', 'tab')
% writematrix(sam2_cum, "C:/Users/Derm/Desktop/big/pshift/scan2_50k.txt", 'Delimiter', 'tab')
% writematrix(truth_cum, "C:/Users/Derm/Desktop/big/pshift/ground_truth_50k.txt", 'Delimiter', 'tab')