%generate lidar training data from ModelNet40 dataset
%TODO: fix broken OFF files

clear all 
close all

nSamples =  50; %25;
epochs = 10000;

true_pos1 = [];
sam1_cum = zeros(epochs*nSamples, 3);
sam2_cum = zeros(epochs*nSamples, 3);
truth_cum = zeros(epochs, 3);

for e = 1:epochs
    e
    try
        %import .off file---------
        %get list of all sub-directories within ModelNet40 dir
        listing = dir('D:/ModelNet40/');
        roll1 = ceil(40*rand); %randomly choose an object type
        object_type = listing(roll1+1).name; %need to add 2 since first elements are "." and ".."
        roll2 = ceil(9*rand);
        filename = "D:/ModelNet40/" + object_type + '/train/' + object_type + "_000" + string(roll2) + ".off";
%         filename = "D:/ModelNet40/" + object_type + '/train/' + object_type + "_0001.off" %for debug
    %     filename = "D:/ModelNet40/airplane/train/airplane_0006.off" %for debug
        [vertices, faces] = read_off(filename);
        vertices = vertices.';
        faces = faces.';
        scale = [0.3+2*rand(), 0.3+2*rand(), 1+5*rand()];
        rot_corr = [90, 0, 90];
        mindist = 3.5;
        %-------------------------
    end

%     %test----------
%     fn= 'C:\Users\Derm\Desktop\big\ModelNet10\toilet\train\toilet_0055.off';
%     scale = [1., 1., 1.];
%     rot_corr = [0, 0, 0];
%     mindist = 4;
%     [vertices, faces] = read_off(fn);
%     vertices = vertices.';
%     faces = faces.';
%     %-------------
    
    vel = [10*randn() 10*randn() 0.3*randn()];
    rot1 = rad2deg(2*pi*rand());
    rot2 = rad2deg(2*pi*rand());
    rot3 = rad2deg(2*pi*rand());

    %sample random  initial position, but NOT INSIDE OBJECT
    too_close = true;
    while too_close
%         pos = [3*randn(), 3*randn, 0];
        pos = [10*randn(), 10*randn(), 0.3*randn()];
        if sqrt(pos(1)^2 + pos(2)^2) > mindist
            if sqrt((pos(1) + vel(1)*0.1 )^2 + (pos(2) + vel(2)*0.1 )^2) > mindist
                too_close = false;
            end
        end
    end

    true_pos1 = [true_pos1; [pos(1), pos(2), pos(3)]];

    mesh = extendedObjectMesh(vertices,faces);     %generate extended object mesh
    mesh = rotate(mesh, rot_corr);     %rotate mesh to correct orientation (provided by each mesh)

    %init lidar unit
    SensorIndex = 1;
    sensor = monostaticLidarSensor(SensorIndex);
    sensor.MountingLocation = [0, 0, 0]; %AHHHHAHHHHAHHHH!!!! Why is this not default zero!!??!??!??!
    
    % set parameters of virtual lidar unit to match velodyne VLP-16
    sensor.UpdateRate = 10;
    sensor.ElevationLimits = [-30, 10];  %[-22, 2]; 
    sensor.RangeAccuracy = 0.0001; % was 0.01, now adding noise at end;
    sensor.AzimuthResolution = 0.2; %0.2;
    sensor.ElevationResolution = 0.4; %0.4;
    sensor.MaxRange = 100;
    
    
    % Create a tracking scenario. Add an ego platform and a target platform.
    scenario = trackingScenario;
    ego = platform(scenario, 'Position', [0, 0, 1.72]);
    target = platform(scenario,'Trajectory',kinematicTrajectory('Position', pos,'Velocity', vel, 'Orientation', quat2rotm(eul2quat([rot1, rot2, rot3])) ));

    target.Mesh = mesh;
    target.Dimensions.Length = scale(1); 
    target.Dimensions.Width = scale(2);
    target.Dimensions.Height = scale(3);
    
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


%augment data by translating scan 2 (remember to adjust solution vector
%accordingly) -------------------------------------------------------------
sam1_cum = [sam1_cum; sam1_cum];
% temp = 5*randn(size(truth_cum)); % single random translation vector
% temp = -rand(size(truth_cum)).*truth_cum; %translate some fraction of truth translation vec back to registration
% temp = (-1.0+ 0.2*rand(size(truth_cum))).*truth_cum*0.1; %translate most of the way to correct soln
temp = (-1.0+ 0.1*randn(size(truth_cum))).*truth_cum*0.1; %translate to correct solution +/- some small error

truth_cum = [truth_cum; truth_cum + 10*temp];
moved_sam2 = sam2_cum + repelem(temp, nSamples ,1);
sam2_cum = [sam2_cum; sam2_cum + repelem(temp, nSamples ,1)]; %need to tile temp 25 times
%-------------------------------------------------------------------------

%augment data 2^4 times by rotating about vertical axis
for i = 1:4
    r = eul2rotm([randn()*360, 0, 0]);
    old1 = reshape(transpose(sam1_cum), [1, 3, size(sam1_cum, 1)]);
    rot1 = pagemtimes(old1, r);
    rot1 = transpose(reshape(rot1, [3, size(sam1_cum, 1)]));
    
    old2 = reshape(transpose(sam2_cum), [1, 3, size(sam2_cum, 1)]);
    rot2 = pagemtimes(old2, r);
    rot2 = transpose(reshape(rot2, [3, size(sam2_cum, 1)]));
    
    % scatter3(rot1(:,1), rot1(:,2), rot1(:,3))
    % scatter3(rot2(:,1), rot2(:,2), rot2(:,3))
    truth_old = reshape(transpose(truth_cum), 1, 3, size(truth_cum, 1));
    rot_truth = pagemtimes(truth_old, r);
    rot_truth = transpose(reshape(rot_truth, [3, size(truth_cum, 1)]));
    
    %append rotated stuff to OG
    sam1_cum = [sam1_cum; rot1];
    sam2_cum = [sam2_cum; rot2];
    truth_cum = [truth_cum; rot_truth];
end

%last step, add gaussian noise to all range estimates
sam1_cum = sam1_cum + 0.01*randn(size(sam1_cum));
sam2_cum = sam2_cum + 0.01*randn(size(sam2_cum));
 
% % %for smaller datasets (keep in git repo)
% writematrix(sam1_cum, "training_data/scan1.txt", 'Delimiter', 'tab')
% writematrix(sam2_cum, "training_data/scan2.txt", 'Delimiter', 'tab')
% writematrix(truth_cum, "training_data/ground_truth.txt", 'Delimiter', 'tab')
% % writematrix(true_pos1, "training_data/true_pos1.txt", 'Delimiter', 'tab')

%for larger datasets (don't save with git)
writematrix(sam1_cum, "D:/TrainingData/ModelNet40/50pts_scan1_320k.txt", 'Delimiter', 'tab')
writematrix(sam2_cum, "D:/TrainingData/ModelNet40/50pts_scan2_320k.txt", 'Delimiter', 'tab')
writematrix(truth_cum, "D:/TrainingData/ModelNet40/50pts_ground_truth_320k.txt", 'Delimiter', 'tab')

%for debug
figure()
hold on
scatter3(sam1(:,1), sam1(:,2), sam1(:,3))
scatter3(sam2(:,1), sam2(:,2), sam2(:,3))
% scatter3(moved_sam2(:,1), moved_sam2(:,2), moved_sam2(:,3))

% %write scaled translated and rotated figure to new stl file for viz
% TR = triangulation( target.Mesh.Faces, target.Mesh.Vertices);
% stlwrite(TR, 'viz_model.stl')
% writematrix(sam1_cum, "viz_scan1.txt", 'Delimiter', 'tab')
% writematrix(sam2_cum, "viz_scan2.txt", 'Delimiter', 'tab')
% writematrix(truth_cum, "viz_ground_truth.txt", 'Delimiter', 'tab')
% writematrix(true_pos1, "viz_true_pos1.txt", 'Delimiter', 'tab')
