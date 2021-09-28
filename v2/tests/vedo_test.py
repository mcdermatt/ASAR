from vedo import *

# Load a polygonal mesh, make it white and glossy:
man = Mesh('https://vedo.embl.es/example/data/man.vtk')
man.c('white').lighting('glossy')

# Create two points:
p1 = Point([ 1,0,1], c='yellow')
p2 = Point([-1,0,2], c='red')

# Add colored light sources at the point positions:
l1 = Light(p1, c='yellow')
l2 = Light(p2, c='red')

# Show everything in one go:
show(man, l1, l2, p1, p2, "Hello World", axes=True)