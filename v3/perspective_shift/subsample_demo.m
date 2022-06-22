% copyright by Jianxiong Xiao http://mit.edu/jxiao
% demo how to sample uniformly on a sphere

%{
Please cite this paper if you use this code in your publication:
J. Xiao, T. Fang, P. Zhao, M. Lhuillier, and L. Quan
Image-based Street-side City Modeling
ACM Transaction on Graphics (TOG), Volume 28, Number 5
Proceedings of ACM SIGGRAPH Asia 2009
%}

clear
clc

fn = 'C:\Users\Derm\Desktop\big\ModelNet10\toilet\train\toilet_0069.off';
[vertex, face] = read_off(fn);
vertex = vertex.';
face = face.';

mesh = extendedObjectMesh(vertex,face);
show(mesh)


tic;
points = icosahedron2sphere(0,vertex,face);
toc;
subplot(1,3,1);
plot3(points(:,1),points(:,2),points(:,3),'.')
title(sprintf('Level %d with %d points',2,size(points,1)))
axis equal
axis tight

tic;
points = icosahedron2sphere(1, vertex, face);
toc;
subplot(1,3,2);
plot3(points(:,1),points(:,2),points(:,3),'.')
title(sprintf('Level %d with %d points',3,size(points,1)))
axis equal
axis tight

tic;
points = icosahedron2sphere(2, vertex, face);
toc;
subplot(1,3,3);
plot3(points(:,1),points(:,2),points(:,3),'.')
title(sprintf('Level %d with %d points',3,size(points,1)))
axis equal
axis tight

figure()
scatter3(vertex(:,1), vertex(:,2), vertex(:,3))