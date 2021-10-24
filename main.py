import mathutils;
import bpy
import numpy as np;
import os;
import re;
import sys;
directory = os.path.dirname(bpy.data.filepath);
if not dir in sys.path:
    sys.path.append(directory);
import importlib;
import Molecule;
import PointGroup;
if "Molecule" in locals():
    importlib.reload(Molecule);
if "PointGroup" in locals():
    importlib.reload(PointGroup);
from Molecule import *;
from PointGroup import *;

#The following commands tells blender to use the gpu
# enable GPUs
known_devices = [r'GeForce RTX 2070 with Max-Q Design',r'GeForce GTX 1080 Ti',r'NVIDIA GeForce RTX 2070', r'GeForce GTX 970']
cuda_devices = bpy.context.preferences.addons['cycles'].preferences.get_devices()
print(cuda_devices)
bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
gpu_found = False
for device in bpy.context.preferences.addons['cycles'].preferences.devices:
    if device.name in known_devices:
        device.use = True
        print('Enabling: %s' % device.name)
        gpu_found = True
    else:
        device.use = False
"""
    json file containing all the information of should be supplied.
    The entries of the json file are self descriptive.
    Note that if an empty list is supplied to the normals entry, a static picture
    of the molecule will be produced.
    command = cd <current directory>
    blender -b main.blend -P main.py -- data.json
"""
# Loading the json file data
argv = sys.argv;
argv = argv[argv.index("--")+1:];
filename = argv[0];
data = Molecule.load_molecule_data(filename, "molecule_data"); #This step exists to account for different ways in which files are handled between linux and windows

# Setting up the defatult rendering properties
for scene in bpy.data.scenes:
    scene.cycles.device = 'GPU'
if not gpu_found:
    scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.samples = data['samples']
    bpy.data.scenes["Scene"].view_settings.gamma = .92;
else:
    scene.render.engine = 'BLENDER_EEVEE';
    bpy.context.scene.cycles.samples = data['samples']*2
    bpy.data.scenes["Scene"].view_settings.gamma = .92;
    bpy.data.scenes["Scene"].eevee.sss_samples = 50
bpy.data.scenes["Scene"].render.film_transparent = True;
bpy.context.scene.render.resolution_x = data['resolution_x']
bpy.context.scene.render.resolution_y = data['resolution_y']
bpy.context.scene.render.tile_x = data['resolution_x']/2
bpy.context.scene.render.tile_y = data['resolution_y']/2
bpy.context.scene.render.resolution_percentage = 100
bpy.data.scenes["Scene"].render.image_settings.file_format = "PNG"
bpy.data.scenes["Scene"].render.image_settings.color_mode = "RGBA"


# Setting the camera position through its world matrix (can be tuned through the json file)
bpy.data.objects.get("Camera").matrix_world = mathutils.Matrix(data["camera_matrix"])
if "focal_length" in data:
    bpy.data.cameras[0].lens = data["focal_length"];
else:
    bpy.data.cameras[0].lens = 50;

#Loading the molecule from the xyz file supplied in the json file
m = Molecule.from_xyz_file(data["xyz"], name = os.path.splitext(data["xyz"])[0]);
m.render(stick = True, mirror_options=data["mirror_options"], rotation_axis_options = data["rotation_axis_options"]) #Adding the geometry to blender (atoms and bonds)

orbitals = "pz"
#SALC = data["SALCS"]["A1g"]["2s"]
SALC = (np.array(m.names) == "C").astype(int)
meshes = m.add_SALC(orbitals, SALC,kind = "cloud")

# Defining the normals and angles that will be used in the animations and to create the appropriate directories
normals = data["normals"]
angles = data["angles"]
pg = PointGroup(data["point_group"])
cc_list = pg.conjugacy_classes; # A directory will be created for each conjugacy class.
                                    # A subdirectory will be created for each normal.

if len(normals) == 0: # Rendering the static image in the target_directory
    m.render_image(data["target_directory"]);
else: # Rendering the images that form the clip of each symmetry operation (one for each normal)
    times, directs = m.animate_properties(cc_list, normals, angles, target_directory=data["target_directory"]);
    m.render_animation(times, directs);
