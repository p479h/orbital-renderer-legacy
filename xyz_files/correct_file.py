"""
This script takes the files copied from that website and corrects them"""

import os

files = [o for o in os.listdir() if o.endswith(".smol")]

def readsmol(file):
    with open(file, "r") as f:
        atoms= []
        positions =[]
        bonds = []
        bonds_bool = False
        stride = 3
        for line in f.readlines():
            if len(line.split()) == 4:
                line  = line.split()
                coordinates = line[0:3]
                atom = line[3];
                coordinates = [float(i) for i in coordinates]
                atoms.append(atom)
                positions.append(coordinates)
                bonds_bool = True
            elif bonds_bool:
                bonds.append([int(line[:3]), int(line[3:6])])
    return atoms, positions, bonds

def writexyz(atoms, coordinates, filename):
    if not filename.endswith(".xyz"):
        filename = filename + ".xyz";
    with open(filename, "w") as f:
        f.truncate(0)
        for a, c in zip(atoms, coordinates):
            f.write(a+"\t"+"\t".join([str(i) for i in c])+"\n");
        
for file in files:
    atoms, coordinates, bonds = readsmol(file);
    name = file.replace(".smol", "")
    writexyz(atoms, coordinates, name)
    
    
