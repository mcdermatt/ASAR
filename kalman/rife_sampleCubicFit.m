%moving-window cubic fit example
% use BESTPOS height data 


% Re-name data read by read0em7memmap_update.m
timeRef = sow42(1); % units sec 
timeVec = sow42;    % units sec
timeIntoExperiment = timeVec-timeRef;
zGPS = bphgt;       % units m

% Build window
windowHalfWidth = 5;  % User defined parameter
windowFullWidth = 2*windowHalfWidth+1;

% Initialize fit
fitToCenter = zeros(size(zGPS))*NaN;  % Initialize fit to values of "Not a Number"
localDiff = fitToCenter;

% Cycle through each time
lengthData = length(zGPS);   
for indx = windowHalfWidth+1:lengthData-windowHalfWidth % scan over window center points
    windowedData = zGPS(indx-windowHalfWidth:indx+windowHalfWidth);
    centerData = zGPS(indx);
    tCenter = timeIntoExperiment(indx);
    t = timeIntoExperiment(indx-windowHalfWidth:indx+windowHalfWidth)-tCenter;
    oneVec = ones(windowFullWidth,1);
    A = [oneVec t t.^2 t.^3]; % If sample rate is consistent, then this matrix is constant; 
                             %  but computing this matrix at each time step
                             %  allows us to deal with jumps in the sample
                             %  time
    coef = A\windowedData; % Compute cubic coefficients
    fitToCenter(indx) = A(windowHalfWidth+1,:)*coef;
    localDiff(indx) = centerData-fitToCenter(indx); % subtract fit from the data at the center point
end

% Plot the original data and fit
figure(1); 
plot(timeIntoExperiment,zGPS,timeIntoExperiment,fitToCenter);
ylabel('Height (m)');
xlabel('Time from start (s)');

figure(2); 
plot(timeIntoExperiment,localDiff);
ylabel('Height Diff (m)');
xlabel('Time from start (s)');
zSigma = std(localDiff,'omitnan');
text(200,0.05,['\sigma_z = ' num2str(zSigma,'%.3f')]);




