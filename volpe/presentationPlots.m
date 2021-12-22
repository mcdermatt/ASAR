figure
plot(1:length(t_lidar),t_lidar+time(1), 'o')
xlabel('lidar sample number')
ylabel('world time')

figure
plot(t_gps_new, 'o')