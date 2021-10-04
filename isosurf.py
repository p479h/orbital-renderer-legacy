import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Molecule import Molecule
from wavefunctions import *
from numba import njit
import multiprocessing as mp
import json
import time
import os


def read_xyz(file):
    points = [];
    with open(file, "r") as f:
        for l in f.readlines():
            l = l.split();
            if len(l) == 4:
                points.append([float(i) for i in l[1:]]);
    return np.array(points);

def iso_find(p, a, field_func, ratios):
    p, a = np.array(p), np.array(a);
    d = p-a;
    phi = np.arctan2(d[:,1], d[:,0]);
    theta = np.arctan2(np.linalg.norm(d[:, :2], axis=1),d[:, 2]);
    r = np.linalg.norm(d, axis = 1);
    return (field_func(r, theta, phi)*ratios/np.linalg.norm(ratios)).sum();

def iso_find2(r, a, field_func , ratios):
    phi, theta = np.mgrid[.05:np.pi:30j, 0:np.pi*2:60j];
    x, y, z = r*np.cos(phi)*np.sin(theta), r*np.sin(theta)*np.sin(phi), r*np.cos(theta);
    x, y, z = [i.reshape(30, 60, 1) for i in (x, y, z)]
    xyz = np.concatenate((x, y, z), axis = 2).reshape(-1, 1, 3); #All points in a sphere
    d = xyz - a;
    phi = np.arctan2(d[:, :, 1], d[:, :, 0])
    theta = np.arctan2(np.linalg.norm(d[:, :, :2], axis=2), d[:, :, 2]);
    r = np.linalg.norm(d, axis = 2)
    values = (field_func(r, theta, phi)*ratios/np.linalg.norm(ratios)).sum(axis=1);
    return max(abs(values.max()), abs(values.min()))
    
#The isovalues for 0.9 probability of finding the electron are:
    # 0.38 -> px, py, pz
    # 0.02 -> 2S
    # 0.10 -> 1S


def make_mesh(name, directory = "molecule_data", radius_offset = 1, n = 10, mixed_orbitals = True):
    data = Molecule.load_molecule_data("data_"+name+".json", directory);
    molecule = read_xyz(data["xyz"]);
    ratio_data = data["SALCS"]
    parent_dir = os.path.join(os.getcwd(),f"images\\{name}");
    keys = [a for a in ratio_data];
    r = np.linalg.norm(molecule[np.abs(ratio_data[keys[0]]["2s"])>.1], axis = 1).max()+radius_offset;
    X, Y, Z = np.mgrid[-r:r:(n*1j),-r:r:(n*1j),-r:r:(n*1j)];
    x, y, z = [i.reshape(n, n, n, 1) for i in (X, Y, Z)];
    xyz = np.concatenate((x, y, z), axis = 3).astype(np.float32);
    d = xyz - molecule.reshape(-1, 1, 1, 1, 3).astype(np.float32);
    dist = np.linalg.norm(d, axis = 4);
    phi = np.arctan2(d[:, :, :, :, 1], d[:, :, :, :, 0]);
    theta = np.arctan2(np.linalg.norm(d[:, :, :, :, :2], axis = 4), d[:, :, :, :, 2]);
    if __name__ == "__main__":
        with mp.Pool(2) as pool:
            for irrep in ratio_data:
                for orbital in ratio_data[irrep]:
                    direc = os.path.join(parent_dir, irrep, orbital);
                    ratios = np.array(ratio_data[irrep][orbital]);
                    if np.abs(ratios).sum() == 0: continue;
                    orbital_func = {"2s":S1, "x":P2x, "y":P2y, "z":P2z}[orbital];
                    if mixed_orbitals:
                        value = iso_find2(r*.95, molecule, orbital_func, ratios)
                    else:
                        value = iso_find2(r*.95, molecule[0][np.newaxis, :], orbital_func, ratios)
                    s = time.time()
                    print("Got to pool");
                    print(irrep, orbital);
                    if mixed_orbitals:
                        scalarfield = (orbital_func(dist, theta, phi)*(ratios/np.linalg.norm(ratios)).reshape(-1, 1, 1, 1)).sum(axis = 0).astype(np.float32);
                    else:
                        print("USED THIS VERSION")
                        scalarfield = (orbital_func(dist, theta, phi)*(ratios/np.linalg.norm(ratios)).reshape(-1, 1, 1, 1))
                        scalarfield[np.abs(scalarfield)<value*.7] *= 0;
                        scalarfield = scalarfield.sum(axis = 0).astype(np.float32);
                    #v, f = marching_cubes(scalarfield, value, np.array(scalarfield.shape), xyz);
                    #v2, f2 = marching_cubes(scalarfield, -value, np.array(scalarfield.shape), xyz);
                    ((v, f), (v2, f2)) = pool.starmap(marching_cubes, ( (scalarfield, value, np.array(scalarfield.shape), xyz), (scalarfield, -value, np.array(scalarfield.shape), xyz) ));
                    print(time.time() - s, "s to compute the isosurfaces");
                    s = time.time();
                    print("Got the triangles")
                    if not os.path.exists(direc):
                        os.makedirs(direc);
                    j = os.path.join
                    np.save(j(direc, "positive_v.npy"), v);
                    np.save(j(direc, "positive_f.npy"), f);
                    np.save(j(direc, "negative_v.npy"), v2);
                    np.save(j(direc, "negative_f.npy"), f2);


if __name__ == "__main__":
    name= "benzene"
    names = [o.replace("data_", "").replace(".json", "") for o in os.listdir("molecule_data") if o.endswith(".json")][8:]
    for name in names:
        make_mesh(name, "molecule_data", 2, 65, True); #This function fails because ivo's pytessel is not compatible with the latest python versions
##fig = plt.figure();
##ax = fig.add_subplot(projection = "3d");
##ax.scatter(*triangles.T);
##
##plt.show()
