%script for simulating lidar scans of input base stl scene (generated using
%autodesk inventor)

%USES INDIVIDUAL OBJECT FILES

%TODO - augment translation by moving PC's generally closer to the correct
% solution - this mimics how the iterative implementation will help solve
% We want to make the basin of attraction as regular as possible

clear all 
close all

nSamples = 25; %25;
epochs = 100;

% sam1_cum = [];
% sam2_cum = [];
% truth_cum = [];
true_pos1 = [];

sam1_cum = zeros(epochs*nSamples, 3);
sam2_cum = zeros(epochs*nSamples, 3);
truth_cum = zeros(epochs, 3);

for e = 1:epochs
    e

    %import stl
    roll = floor(6*rand());
%     roll = 5; %all cylinders 
%     roll = 3;
%     roll = 6; %taxi
%     roll = 7; %vw bus
    roll = 8; %dummy
    if roll == 0
        FileName = 'training_data/simple_object1.stl';
    %     scale = [2, 2, 10];
          scale = [0.3+2*rand(), 0.3+2*rand(), 1+5*rand()];
        rot_corr = [90, 0, 90];
        mindist = 3.5;
    end
    if roll == 1
        FileName = 'training_data/simple_object2.stl';
    % %     scale = [.2, 0.5, 3];
          scale = [0.1+2*rand(), 0.1+2*rand(), 1+5*rand()];
        rot_corr = [90, 0, 90];
        mindist = 4;
    end
    if roll == 2
        FileName = 'training_data/car1.stl';
        scale = [4.23, 1.79, 1.82];
        rot_corr = [90, 0, 90]; %corrective rotation to orient model wheels down
        mindist = 4;
    end
    if roll == 3
        FileName = 'training_data/car2.stl';
        scale = [4.7, 2.09, 1.45];
        rot_corr = [90, 0, 0];
        mindist = 5;
    end
    if roll == 4
        FileName = 'training_data/bus1.stl';
        scale = [12.2, 2.23, 3.5];
        rot_corr = [90, 0, 90];
        mindist = 8;
    end
    if roll == 5
        FileName = 'training_data/simple_object3.stl'; %cylinder
        dia = 0.1 + 1.5*rand();
        scale = [dia, dia, 10];
%         scale = [0.5, 0.5, 10]
        rot_corr = [90, 0, 90];
        mindist = 1;
    end
   
    if roll == 6
        FileName = 'training_data/Taxi.stl';
        scale = [2, 5, 2];
        rot_corr = [0, 0, 0];
        mindist = 3;
    end

    if roll == 7
        FileName = 'training_data/VW_Bus.stl'; 
        scale = [2, 6, 2.5];
        rot_corr = [0, 0, 0];
        mindist = 3;
    end

    if roll == 8
        FileName = 'training_data/dummy.stl'; 
        scale = [0.6, 0.3, 2.0];
        rot_corr = [0, 0, 90];
        mindist = 4;
    end

    OpenFile = stlread(FileName);
    
    %get vertices, faces, and normals from stl
    vertices = OpenFile.Points;
    faces = OpenFile.ConnectivityList;
    
    vel = [10*randn() 10*randn() 1*randn()];
    rot = rad2deg(2*pi*rand());

    %sample random  initial position, but NOT INSIDE OBJECT
    too_close = true;
    while too_close
        pos = [3*randn(), 3*randn, 0];    
        if sqrt(pos(1)^2 + pos(2)^2) > mindist
            if sqrt((pos(1) + vel(1)*0.1 )^2 + (pos(2) + vel(2)*0.1 )^2) > mindist
                too_close = false;
            end
        end
    end

    pos = [4.0, 4.0, 0.01];

    true_pos1 = [true_pos1; pos];

%     pos = [2, 1, 0];
%     vel = [0, -10, 0];

    mesh = extendedObjectMesh(vertices,faces);     %generate extended object mesh
    mesh = rotate(mesh, rot_corr);     %rotate mesh to correct orientation (provided by each mesh)

    %init lidar unit
    SensorIndex = 1;
    sensor = monostaticLidarSensor(SensorIndex);
    
    % set parameters of virtual lidar unit to match velodyne VLP-16
    sensor.UpdateRate = 10;
    sensor.ElevationLimits = [-30, 10];  %[-22, 2]; 
    sensor.RangeAccuracy = 0.0001; % was 0.01, now adding noise at end;
    sensor.AzimuthResolution = 0.4; %0.2;
    sensor.ElevationResolution = 1.2; %0.4;
    sensor.MaxRange = 50;
    
    
    % Create a tracking scenario. Add an ego platform and a target platform.
    scenario = trackingScenario;
    ego = platform(scenario, 'Position', [0, 0, 1.72]);
    % ego.Position = [0, 0, 1.72];
    target = platform(scenario,'Trajectory',kinematicTrajectory('Position', pos,'Velocity', vel, 'Orientation', quat2rotm(eul2quat([rot, 0, 0])) )); %no rotation
    % target = platform(scenario,'Trajectory',kinematicTrajectory('Position',[10 0 0],'Velocity',[5 0 0], 'AngularVelocity', [0, 0, 0.1])); %with rotatation
%     rotation = eul2quat([rot, 0, 90]);
%     target.pose.Orientation = rotation;

    target.Mesh = mesh;
    target.Dimensions.Length = scale(1); 
    target.Dimensions.Width = scale(2);
    target.Dimensions.Height = scale(3);
%     target.pose();

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

% scatter3(ptCloud1(:,1), ptCloud1(:,2), ptCloud1(:,3))
% scatter3(ptCloud2(:,1), ptCloud2(:,2), ptCloud2(:,3))
scatter3(sam1(:,1), sam1(:,2), sam1(:,3))
scatter3(sam2(:,1), sam2(:,2), sam2(:,3))

% %augment data by translating scan 2 (remember to adjust solution vector
% %accordingly) -------------------------------------------------------------
% sam1_cum = [sam1_cum; sam1_cum];
% % temp = 5*randn(size(truth_cum)); % single random translation vector
% % temp = -rand(size(truth_cum)).*truth_cum; %translate some fraction of truth translation vec back to registration
% % temp = (-1.0+ 0.2*rand(size(truth_cum))).*truth_cum*0.1; %translate most of the way to correct soln
% temp = (-1.0+ 0.1*randn(size(truth_cum))).*truth_cum*0.1; %translate to correct solution +/- some small error
% 
% truth_cum = [truth_cum; truth_cum + 10*temp];
% moved_sam2 = sam2_cum + repelem(temp, nSamples ,1);
% sam2_cum = [sam2_cum; sam2_cum + repelem(temp, nSamples ,1)]; %need to tile temp 25 times
% %-------------------------------------------------------------------------
% 
% %augment data 2^4 times by rotating about vertical axis
% for i = 1:4
%     r = eul2rotm([randn()*360, 0, 0]);
%     old1 = reshape(transpose(sam1_cum), [1, 3, size(sam1_cum, 1)]);
%     rot1 = pagemtimes(old1, r);
%     rot1 = transpose(reshape(rot1, [3, size(sam1_cum, 1)]));
%     
%     old2 = reshape(transpose(sam2_cum), [1, 3, size(sam2_cum, 1)]);
%     rot2 = pagemtimes(old2, r);
%     rot2 = transpose(reshape(rot2, [3, size(sam2_cum, 1)]));
%     
%     % scatter3(rot1(:,1), rot1(:,2), rot1(:,3))
%     % scatter3(rot2(:,1), rot2(:,2), rot2(:,3))
%     truth_old = reshape(transpose(truth_cum), 1, 3, size(truth_cum, 1));
%     rot_truth = pagemtimes(truth_old, r);
%     rot_truth = transpose(reshape(rot_truth, [3, size(truth_cum, 1)]));
%     
%     %append rotated stuff to OG
%     sam1_cum = [sam1_cum; rot1];
%     sam2_cum = [sam2_cum; rot2];
%     truth_cum = [truth_cum; rot_truth];
% end

%last step, add gaussian noise to all range estimates
sam1_cum = sam1_cum + 0.01*randn(size(sam1_cum));
sam2_cum = sam2_cum + 0.01*randn(size(sam2_cum));

% %for smaller datasets (keep in git repo)
writematrix(sam1_cum, "training_data/scan1.txt", 'Delimiter', 'tab')
writematrix(sam2_cum, "training_data/scan2.txt", 'Delimiter', 'tab')
writematrix(truth_cum, "training_data/ground_truth.txt", 'Delimiter', 'tab')
writematrix(true_pos1, "training_data/true_pos1.txt", 'Delimiter', 'tab')


%for larger datasets (don't save with git)
% writematrix(sam1_cum, "C:/Users/Derm/Desktop/big/pshift/scan1_10k.txt", 'Delimiter', 'tab')
% writematrix(sam2_cum, "C:/Users/Derm/Desktop/big/pshift/scan2_10k.txt", 'Delimiter', 'tab')
% writematrix(truth_cum, "C:/Users/Derm/Desktop/big/pshift/ground_truth_10k.txt", 'Delimiter', 'tab')

%test scene for viz
% writematrix(sam1_cum, "training_data/car_demo_scan1.txt", 'Delimiter', 'tab')
% writematrix(sam2_cum, "training_data/car_demo_scan2.txt", 'Delimiter', 'tab')
% writematrix(truth_cum, "training_data/car_demo_ground_truth.txt", 'Delimiter', 'tab')

% %for debug
% figure()
% hold on
% scatter3(sam1(:,1), sam1(:,2), sam1(:,3))
% scatter3(sam2(:,1), sam2(:,2), sam2(:,3))
% scatter3(moved_sam2(:,1), moved_sam2(:,2), moved_sam2(:,3))
