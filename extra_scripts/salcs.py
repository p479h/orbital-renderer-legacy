import os;
import numpy as np;
from fractions import gcd;
import matplotlib.pyplot as plt;
from mpl_toolkits.mplot3d import Axes3D;
import numpy as np
import mathutils
from PointGroup import PointGroup
from Molecule import Molecule
import json

def search_json(path):
    dirs = []
    for d in os.listdir(path):
        p = os.path.join(path, d)
        if os.path.isdir(p):
            for pp in os.listdir(p):
                if pp.endswith(".json"):
                    dirs.append(p)
                    break
    return dirs

molpaths = [os.path.join(d, o) for d in search_json("smol_files") for o in os.listdir(d) if o.endswith(".json")]
for molpath in molpaths:
    directory, group, molname = molpath.split(os.sep)
    molname = molname.replace(".json", "")
    json_path = os.path.join(directory, group, f"{molname}.json")
    smol_path = os.path.join(directory, group, f"{molname}.smol")
    data = Molecule.load_molecule_data(json_path, None)

    m = Molecule.from_smol_file(smol_path)

    # Defining the normals and angles that will be used in the animations and to create the appropriate directories
    normals = data["normals"]
    angles = np.array(data["angles"])
    pg = PointGroup(data["point_group"])
    pg.set_normals(normals);
    print(pg.conjugacy_classes)

    orbitals = pg.create_orbitals(m.position)
    pg.make_matrices_from_normals(normals, angles)
    traces, tracep = pg.find_reducible_representation(orbitals)
    salcs = pg.print_salcs(orbitals, traces+tracep, 2)
    print(traces)

    data["SALCS"] = salcs
    Molecule.dump_molecule_data(data, json_path, None);

