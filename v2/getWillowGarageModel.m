%script used to retrieve Willow Garage model used in Magnusson's 3D-NDT-D2D
% paper

gzinit("192.168.198.129",14581)

List = gzmodel("list");