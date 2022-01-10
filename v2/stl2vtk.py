import pyvista as pv
# file for converting stl meshes to vedo friendly vtk

source_file = "C:/Users/Derm/honda.stl"
stlf = pv.read(source_file)

out_file = "C:/Users/Derm/honda.vtk"
pv.save_meshio(out_file, stlf)