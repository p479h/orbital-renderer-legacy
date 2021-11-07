import numpy as np
import os
from PointGroup import PointGroup
from Molecule import Molecule
import shutil

filename = "data_ferrocene.json"
data = Molecule.load_molecule_data(filename, "molecule_data")
directory = data["target_directory"]
pg = PointGroup(data["point_group"])
m = Molecule.from_xyz_file(data["xyz"], name = os.path.splitext(data["xyz"])[0]);
labels = {k:i for i,k in enumerate("E C i S g".split())}; #g is in 'sig'
tt = m.transition;
sp = 1;
p = tt + sp
new_sp = 30
durations = [p, p*3, p*2, p*5, p*3]

def find_duration(cc):
    for label in labels:
        if label in cc.replace("sig", "g"):
            return durations[labels[label]]

def get_count(pic_name):
    return pic_name.replace(".png", "").replace("a_", "")

current_picture = 0
for cci, cc in enumerate(pg.conjugacy_classes):
    cc_dir = pg.normal_to_directory(cc)
    cc_path = os.path.join(directory, cc_dir)
    duration = find_duration(cc)
    for operation in os.listdir(cc_path):
        operation_dir = os.path.join(cc_path, operation)
        pics = [o for o in os.listdir(operation_dir) if o.endswith(".png")]
        if len(pics) != duration:
            print(duration)
            print(len(pics))
            print("Problem")
            break
        
        for i, pic in enumerate(pics):
            if pic[0].isalpha():
                continue
            path = os.path.join(operation_dir, pic)
            new_path = os.path.join(operation_dir, "a_"+pic)
            os.rename(path, new_path)

        for pici, pic in enumerate(pics):
            #rename the picture to its correct name
            old_path = os.path.join(operation_dir, pic)
            new_path = os.path.join(operation_dir, str(current_picture).zfill(len(get_count(pic)))+".png")
            os.rename(old_path, new_path)
            if pici%p == 0:
                for i in range(sp, new_sp):
                    current_picture += 1;
                    new_name = str(current_picture).zfill(len(get_count(pic))) + ".png"
                    new_path_copy = os.path.join(operation_dir, new_name)
                    shutil.copy(new_path, new_path_copy)
            current_picture += 1 
            
print(current_picture)
        

        
        
