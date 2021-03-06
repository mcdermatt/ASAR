DeviceModel='PuckHiRes';
FilePath = 'C:/Users/Derm/2021-03-10-16-43-50_Velodyne-VLP-16-Data_garminSignage.pcap';
outfile = 'C:/Users/Derm/2021-03-10-16-43-50_Velodyne-VLP-16-Data_garminSignage.txt';

veloReader = velodyneFileReader(FilePath,DeviceModel);

%The first point cloud of interest is captured at 0.3 second into the file. Set the CurrentTime property to that time to being reading point clouds from there.

veloReader.CurrentTime = veloReader.StartTime; %+ seconds(100);
tStart=veloReader.CurrentTime;
tEnd=tStart+seconds(1)+seconds(300);
StartFrame=find(veloReader.Timestamps==tStart);%1501;
EndFrame=find(abs(veloReader.Timestamps-tEnd)<=seconds(.05));

L_Buff=1;%number of Frames to average
dr=.05; %define cyl coord resolution to interpolate on
daz=.2*pi/180; %define cyl coord resolution to interpolate on
dz=.3; %define cyl coord resolution to interpolate on

iFrame=StartFrame-1;

while (hasFrame(veloReader) && veloReader.CurrentTime < tEnd)

    iFrame=iFrame+1;
    NFramesProcessed=iFrame-StartFrame+1;
    ptCloudObj = readFrame(veloReader,iFrame);
    X_2D=ptCloudObj.Location(:,:,1);%16     xN. so first dimension is elevation, second is along azimuth
    Y_2D=ptCloudObj.Location(:,:,2);
    Z_2D=ptCloudObj.Location(:,:,3);
    I_2D=ptCloudObj.Intensity;
    
end

    %you can then plot, save, aggregate, or process the extracted data

%store cloud from middle of drive
ptCloudObj = readFrame(veloReader, 1000);
a = size(ptCloudObj.Location);
pts = [];
for ct = 1:a(1)
    pts = [pts, ptCloudObj.Location(ct,:,:)];
end
pts = squeeze(pts);
csvwrite(outfile, pts);

    
% test = ptCloudObj.Location(1,:,:);
% test = squeeze(test);
% csvwrite(outfile, test)


%If you want to use the Matlab player to visualize the data then this is some sample code:

%Define x-, y-, and z-axes limits for pcplayer in meters. Label the axes.
xlimits = [-150 150];
ylimits = [-150 150];
zlimits = [-20 20];

%Create the point cloud player.
player = pcplayer(xlimits,ylimits,zlimits);

%Label the axes.
xlabel(player.Axes,'X (m)');
ylabel(player.Axes,'Y (m)');
zlabel(player.Axes,'Z (m)');

% open cloud with player
view(player,ptCloudObj);


% test- create CSV from frame so I can open it in a real programming
% language...




