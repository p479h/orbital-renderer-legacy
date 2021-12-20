from all_imports import *

class FileManager:

    @staticmethod
    def load_molecule_data(filepath):
        """
            Reads and returns data from json files used for animation of symmetry operations
            """
        with open(filepath, "r") as f:
            data = json.load(f)
        if type(data["xyz"]) == list: #Make sure that the path is compatible with different os's
            data["xyz"] = os.path.join(*data["xyz"])
        return data

    @staticmethod
    def dump_molecule_data(data, filepath):
        """
            Writes json file with information used for symmetry operations
            """
        if type(data["xyz"]) == str:
            xyzpath = os.path.normpath(data["xyz"])
            data["xyz"] = xyzpath.split(os.sep) #Now we have a list that can be reassembled in accordance with the operating system's path separator
        with open(filepath, "w") as f:
            json.dump(data, f, ensure_ascii = False)


    @classmethod
    def read_smol(cls, path):
        atoms = []
        positions = []
        bonds = []
        with open(path, "r") as f:
            found_bonds = False
            for l in f.readlines():
                if len(l.split()) == 4:
                    found_bonds = True
                    l = l.split()
                    atoms.append(l[3])
                    positions.append([float(i) for i in l[:3]])
                elif found_bonds:
                    bonds.append([int(l[:3])-1, int(l[3:6])-1])
        return atoms, positions, bonds

    @classmethod
    def read_mol(cls, path):
        atoms = []
        coordinates = []
        bonds = []
        orders = []
        with open(path, "r") as f:
            [f.readline() for i in range(3)]
            line1 = f.readline() #counting the number of atoms and bonds before entering the for loops
            n_atoms = int(line1[:3])
            n_bonds = int(line1[3:6])
            lines = f.readlines()
            for i,l in enumerate(lines):
                if i < n_atoms:
                    l = l.split()
                    atoms.append(l[3])
                    coordinates.append([float(num) for num in l[:3]])
                else:
                    if i > n_bonds + n_atoms - 1:
                        break
                    bonds.append([int(l[:3])-1, int(l[3:6])-1])
                    orders.append(int(l[6:9]))
        return atoms, coordinates, bonds, orders

    @classmethod
    def read_xyz(cls, path):
        positions = []
        names = [] #The atomic labels
        with open(path, "r") as file:
            lines = file.read().replace("\\t", " ").split("\n")
            for i, l in enumerate(lines):
                pieces = l.split()
                if len(pieces) == 4:
                    positions.append([float(coord) for coord in pieces[1:]])
                    names.append(pieces[0])
        return names, positions
