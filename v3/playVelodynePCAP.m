% display point cloud caputre dVelodyne pcap file

beep off
clear all
close all

fn = "D:/volpe/2021-03-10-16-43-50_Velodyne-VLP-16-Data_garminSignage.pcap";
veloReader = velodyneFileReader(fn, "VLP16")


%testing out MatLab Point Cloud Player Utility
xlimits = [-60 60];
ylimits = [-60 60];
zlimits = [-20 20];
player = pcplayer(xlimits,ylimits,zlimits);
while(hasFrame(veloReader) && player.isOpen() && (veloReader.CurrentTime < veloReader.StartTime + seconds(1000)))
    ptCloudObj = readFrame(veloReader);
    
    %remove ground plane
    [~,nonGroundPtCloud,groundPtCloud] = segmentGroundSMRF(ptCloudObj,MaxWindowRadius=5,ElevationThreshold=0.1,ElevationScale=0.25);

%     view(player,ptCloudObj.Location,ptCloudObj.Intensity); %with ground plane
    view(player,nonGroundPtCloud.Location,nonGroundPtCloud.Intensity);
    pause(0.01);
end