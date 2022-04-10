%read and parse data from TIERS forest trajectory rosbag file and save the
%  raw point cloud coordinates in text file


clear all
close all

bag = rosbag('forest02_straight.bag');

frames = bag.AvailableFrames;

%select a specific topic
% bSel = select(bag, 'Topic', '/os_cloud_node/points'); %entire cloud??
bSel = select(bag, 'Topic', '/velodyne_points');

%read messages
msgStructs = readMessages(bSel, 'DataFormat', 'struct');
%wow this takes forever to load


test = msgStructs{2,1};
% hold on
% rosPlot(test) %this actually works...
% rosPlot(msgStructs{2,1})

xyz = rosReadXYZ(test);

% figure()
% scatter3(xyz(:,1), xyz(:,2), xyz(:,3), 'filled')

writematrix(xyz, "rawPointClouds/scan1.txt", 'Delimiter', 'tab')