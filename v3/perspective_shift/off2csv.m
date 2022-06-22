%subsample 50 points from .off files in ModelNet10 and save to csv 

clear all
close all

% fn = 'C:\Users\Derm\Desktop\big\ModelNet10\sofa\train\sofa_0145.off';
% fn = 'C:\Users\Derm\Desktop\big\ModelNet10\chair\train\chair_0010.off';
fn = 'C:\Users\Derm\Desktop\big\ModelNet10\toilet\train\toilet_0069.off';


[vertex, face] = read_off(fn);
vertex = vertex.';
face = face.';

mesh = extendedObjectMesh(vertex,face);
show(mesh)

%TODO: Figure out how to uniformly sample points from entire surfaces of
%mesh-- Or don't???

figure()
scatter3(vertex(:,1), vertex(:,2), vertex(:,3))