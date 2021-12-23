from all_imports import *
from FileManager import FileManager as fm

def get_rotation_around(angle, shape, axis):
    axis = np.array(axis)/np.linalg.norm(np.array(axis))
    l, m, n = axis
    diag = np.eye(3)*np.cos(angle)
    rest = (np.ones((3, 3))-np.eye(3))*np.sin(angle)
    angular_part = diag+rest
    ll = np.array(axis)*np.array(axis)[:, None]*(1-np.cos(angle))
    rr = np.array([[1, -n, m], [n, 1, -l], [-m, l, 1]])*angular_part
    if shape == 4:
        matrix = np.eye(4)
        matrix[:3, :3] = ll+rr
        return matrix
    return ll + rr

def get_reflection_across(_, shape, axis):
    axis = np.array(axis)
    axis = axis/np.linalg.norm(axis)
    N = axis[None, :]
    A = (np.eye(3)-2*N*N.T)
    if shape == 4:
        I = np.eye(4)
        I[:3, :3] = A
        return I
    return A

class PointGroup:
    def __init__(self, file: str):
        file = os.path.join(os.getcwd(), "pointgroup_data", file)
        if not os.path.splitext(file)[-1] == ".txt":
            file = file + ".txt"
        with open(file, "r") as f:
            self.text = open(file, "r").read()
        self.pg_name = self.read_pg_name(file)
        self.conjugacy_classes = self.read_conjugacy_classes(file) #Symmetry operations with labels
        self.conjugacy_counts = self.count_conjugacy(self.conjugacy_classes) #Number of elements for each operation
        self.h = self.conjugacy_counts.sum() #This should be equal to the length of the vectors in the table!
        self.irreps = self.read_irreps(file) #Irrep labels
        self.characters = self.read_characters(file)

    @classmethod
    def normal_to_directory(self, term: str) -> str:
        """
            Takes term and writes it in the same way as in the directory"""
        return term.replace("'", "_prime")

    @classmethod
    def directory_to_normal(self, term: str) -> str:
        """
            Takes a directory name and writes it as it will be read from the point group files"""
        return term.replace("_prime","'")

    def read_conjugacy_classes(self, file: str) -> list:
        with open(file, "r") as f:
            return f.readline().split()[1:]

    def read_pg_name(self, file: str) -> list:
        with open(file, "r") as f:
            return f.readline().split()[0]

    @classmethod
    def find_angles(cls, conjugacy_classes: list) -> np.ndarray: #Find angles of rotation for the conjugacy_classes
        n = []
        pattern = r"[C|S]\'*([0-9]+)\)?([0-9]+)?"
        for cc in conjugacy_classes:
            results = re.search(pattern, cc)
            if results:
                n_ = int(results.group(1))
                if results.group(2):
                    n_ /= int(results.group(2))
            else:
                n_ = 1
            n.append(n_)
        angles = np.pi*2/np.array(n)
        return angles

    def make_matrices_from_normals(self, normals: list, angles: np.ndarray) -> list:
        self.matrices = []
        for ci, cc in enumerate(self.conjugacy_classes):
            if "E" == cc:
                self.matrices.append(self.make_identity()[np.newaxis, :])
                continue
            elif "i" == cc:
                self.matrices.append(self.make_inversion()[np.newaxis, :])
                continue
            elif "C" in cc or "S" in cc:
                func, args = (self.make_rotation if "C" in cc else self.make_improper_rotation), (np.array(normals[ci]), angles[ci])
            elif cc.count("sig")>0:
                func, args = self.make_reflection, (np.array(normals[ci]),)
            self.matrices.append(func(*args))
        return self.matrices

    def get_so_matrices(self, normals: list) -> list:
        return self.make_matrices_from_normals(normals, self.find_angles(self.conjugacy_classes))


    def expand(self): #Was used for some early animations. I think it is a good idea to keep it
        s = []
        pattern = re.compile("([0-9]+)([A-Z]|sig|Sig|sigma|Sigma)(\S+)")
        for cls, count in zip(self.conjugacy_classes, self.conjugacy_counts):
            s = s + [pattern.sub("\2\3",cls)]*count
        return s

    def make_rotation(self, normal: np.ndarray, angle: float) -> np.ndarray: #This and the following "tranformation functions" generate 4x4 matrices that can be appplied to the orbitals
        if len(normal.shape) == 1:
            return get_rotation_around(angle, 4, normal)
        else:
            return np.array([get_rotation_around(angle, 4, norm) for norm in normal])

    def make_reflection(self, normal: np.ndarray, angle: float = 0) -> np.ndarray:
        if len(normal.shape) == 1:
            return np.array(get_reflection_across(-1, 4, normal))
        else:
            return np.array([get_reflection_across(-1, 4, norm) for norm in normal])

    def make_inversion(self, normal: list = [], angle: float = 0) -> np.ndarray:
        i = np.eye(4)*-1
        i[3, 3] = 1
        return i

    def make_identity(self, normal: list = [], angle: float = 0) -> np.ndarray:
        return np.eye(4)

    def make_improper_rotation(self, normal: np.ndarray, angle: float)->np.ndarray:
        if len(normal.shape) == 1:
            return np.array(self.make_reflection(normal)@self.make_rotation(normal, angle))
        else:
            rotations = self.make_rotation(normal, angle)
            reflections = self.make_reflection(normal)
            return np.array([r@rot for r, rot in zip(reflections, rotations)])

    def get_irreps(self, trace: np.ndarray) -> list:
        counts = (trace*self.characters).sum(axis = 1)/self.h
        return [str(int(round(c, 0)))+n for c, n in zip(counts, self.irreps) if c > 0 ]


    def count_mos(self, trace: np.ndarray) -> float:
        return ((trace*self.characters).sum(axis=1)/self.h * self.characters[:, 0]).sum()

    def read_irreps(self, file: str) -> list:
        with open(file, "r") as f:
            f.readline()
            return [line.split()[0] for line in f.readlines()]

    def read_characters(self, file: str) -> np.ndarray:
        with open(file, "r") as f:
            f.readline()
            c = []
            for l in f.readlines():
                l = l.replace("pi", "*np.pi").replace("2c", "2*np.c").replace("3c", "3*np.c").replace(")½", ")**(0.5)").replace("Â½", "**2")
                try:
                    c.append([eval(i)for i in l.split()[1:]])
                except:
                    c.append(np.zeros(len(l.split())))
                    print(self.pg_name)
                    print("problem with ", l, "while collecting characters")
            return np.array(c)

    def count_conjugacy(self, classes: list) -> np.ndarray:
        counts = []
        pattern = re.compile("([0-9]+)\(?([A-Z]|sig|Sig|sigma|Sigma)")
        for cl in classes:
            result = 1 if not pattern.search(cl) else int(pattern.search(cl).group(1))
            counts.append(result)
        return np.array(counts)


    def find_irreps(self, reducible_representation: np.ndarray) -> float:
        return (self.characters*reducible_representation.ravel()).sum(axis=1)/self.h


    def set_normals(self, normals: list) -> None: #All normals!!! For all conjugacy classes
        for i, n_group in enumerate(normals):
            self.normals[i] = n_group

    def set_naxis(self, n, conj_cls = 0):
        self.naxis[conj_cls] = n

    def get_header(self) -> str:
        return " | ".join([self.pg_name]+self.conjugacy_classes)

    def permute_atoms(self, orbitals: np.ndarray, matrix: np.ndarray) -> np.ndarray: # Used for finding the linearly independent SALCS
        transformed_positions = (matrix[:3, :3]@orbitals[:, :3, 3].T).T
        new_indices = []
        indices = np.arange(len(transformed_positions))
        for i, tp in enumerate(transformed_positions):
            d = np.linalg.norm(transformed_positions[i] - orbitals[:, :3, 3], axis=1)
            new_indices.append(indices[d<1e-1])

        return np.array(new_indices).flatten()


    def linear_independenceE(self, vector, label, orbitals, matrices):
        matrix = matrices[0]
        v1 = vector
        v2 = vector[self.permute_atoms(orbitals, matrix)]
        if abs((v1+v2)@(v1-v2))>1e-5:
            print("PROBLEM")
            print("Linear independence E not working")
        labels = [l+["+", "-"][i] for i, l in enumerate([label]*2)]
        return (labels, np.array([v1+v2, v1-v2]))

    def linear_independenceT(self, vector, label, orbitals, matrices):
        v = np.array(vector[np.newaxis, :].tolist()+[vector[self.permute_atoms(orbitals, matrices[i])] for i in range(len(matrices))])
        u = self.GramS(v[:3])
        labels = [label + l for l in "12 13 23".split()]
        return (labels, u)

    def linear_independenceG(self, vector, label, orbitals, matrices):
        v = np.array(vector[np.newaxis, :].tolist()+[vector[self.permute_atoms(orbitals, matrices[i])] for i in range(len(matrices))])
        u = self.GramS(v[:4])
        labels = [label + l for l in "12 13 14 23".split()]
        return (labels, u)

    def linear_independenceH(self, vector, label, orbitals, matrices):
        v = np.array(vector[np.newaxis, :].tolist()+[vector[self.permute_atoms(orbitals, matrices[i])] for i in range(len(matrices))])
        u = self.GramS(v[:5])
        labels = [label + l for l in "12 13 14 23".split()]
        return (labels, u)

    def latexify_term(self, t: str) -> str:
        t = t.replace("sigma", "sig").replace("Sigma", "sig").replace("infty", "inf").replace("Infty", "inf")
        #Takes a term t and makes it latex friendly (Could be improved I think)
        pattern = r"([0-9]+)?(\(?)([A-Z]|Sig|sig|sigma|Sigma)('*)([a-z0-9]+|Inf|inf|infty|Infty)?(\))?([0-9]+)?"
        pattern = re.compile(pattern)
        subbed = pattern.sub(r"\1\2\3_{\5}\6^{\4\7}", t)
        if subbed.find(")") != -1:
            subbed = pattern.sub(r"\1\2\3^{\4}_{\5}\6^{\7}", t)
        return subbed.replace("^{}", "").replace("_{}", "").replace("sig", r"\sigma ").replace("inf", r"\infty").replace("'", r"\prime ")

    def latexify_terms(self, terms: list) -> list:
        return [self.latexify_term(t) for t in terms]


    def GramS(self, X, row_vecs=True, norm = False): #I got this one online. Writen by ingmarschuster
        if not row_vecs:
            X = X.T
        Y = X[0:1,:].copy()
        for i in range(1, X.shape[0]):
            proj = np.diag((X[i,:].dot(Y.T)/np.linalg.norm(Y,axis=1)**2).flat).dot(Y)
            Y = np.vstack((Y, X[i,:] - proj.sum(0)))
        if norm:
            Y = np.diag(1/np.linalg.norm(Y,axis=1)).dot(Y)
        if (True in np.isnan(Y)): #This means that the vectors were filled with zeros
            Y = np.zeros(Y.shape)
        if row_vecs:
            return Y
        else:
            return Y.T

    def latex_table(self, text = None, frame = False)->None:
        pattern = r"([0-9]+)?(\(?)([A-Z]|Sig|sig|sigma|Sigma)('*)([a-z0-9]+|Inf|inf|infty|Infty)?(\))?([0-9]+)?"
        pattern = re.compile(pattern)
        subbed = pattern.sub(r"$\1\2\3_{\5}\6^{\4\7}$", self.text if not text else text).replace("sig", r"\sigma").replace("inf", r"\infty").replace("_{}", "").replace("^{}", "")
        lines = subbed.split("\n")
        cols = len(lines[0].split('\t'))
        print("\\begin{center}")
        print("\\begin{tabular}{" + ("|" if frame else "") +"|".join(list("c"*cols))+ ("|" if frame else "")+"}")
        if frame:
            print("\\hline")
        for i, line in enumerate(lines):
            print(line.replace("\t", " & ") + "\\\\" + ( "\\hline" if i < len(lines)-1 or frame else ""))
        print("\\end{tabular}\n\\end{center}")

    def latex_complete_table(self, text = None, frame = False)->None:
        pattern = r"([0-9]+)?(\(?)([A-Z]|Sig|sig|sigma|Sigma)('*)([a-z0-9]+|Inf|inf|infty|Infty)?(\))?([0-9]+)?"
        pattern = re.compile(pattern)
        header = "\t".join(self.get_expanded_conjugacy_classes())
        header = pattern.sub(r"$\1\2\3_{\5}\6^{\4\7}$", header).replace("sig", r"\sigma").replace("inf", r"\infty").replace("_{}", "").replace("^{}", "")
        subbed = pattern.sub(r"$\1\2\3_{\5}\6^{\4\7}$", self.text if not text else text).replace("sig", r"\sigma").replace("inf", r"\infty").replace("_{}", "").replace("^{}", "")
        lines = subbed.split("\n")[1:]
        cols = len(lines[0].split('\t'))
        print("\\begin{center}")
        print("\\begin{tabular}{" + ("|" if frame else "") +"|".join(list("c"*(sum(self.conjugacy_counts)+1)))+ ("|" if frame else "")+"}")
        if frame:
            print("\\hline")
        print(f"${self.latexify_term(self.pg_name)}$ & " + header.replace("\t", " & "))
        print("\\\\" + "\\hline")
        for i, line in enumerate(lines):
            for ie, element in enumerate(line.split("\t")):
                if ie == 0:
                    print(element, " & ", end = "")
                else:
                    for ic, cc in enumerate(range(self.conjugacy_counts[ie-1])):
                        print(element, end = "")
                        if ie > 0 and ie < len(line.split("\t"))-1:
                            print(" & ", end = "")
                        elif (ie == len(line.split("\t"))-1) and ic < self.conjugacy_counts[ie-1]-1:
                            print(" & ", end = "")

            print("\\\\" + ( "\\hline" if i < len(lines)-1 or frame else ""))
        print("\\end{tabular}\n\\end{center}")


    def get_expanded_conjugacy_classes(self)->list:
        """ Brings all transformations individually e.g. E C C C instead of E 3C"""
        new_columns = []
        for i, cc in enumerate(self.conjugacy_counts):
            kind = re.search(r"([ECSi]['0-9']*)|sig[vdh]*", self.conjugacy_classes[i]).group(0)
            for j in range(cc):
                new_columns.append(kind)
        return new_columns

    def get_expanded_characters(self)->np.ndarray:
        new_columns = np.zeros((self.characters.shape[0], self.conjugacy_counts.sum()))
        j = 0 #At any point in time, j is cumsum(cc[:i]), but thats too wordy... :D
        for i, cc in enumerate(self.conjugacy_counts):
            col = np.tile(self.characters[:, [i]], (1, cc))
            new_columns[:, np.arange(j, j+cc)] = col
            j+=cc
        return np.array(new_columns)

class SObject:
    #Symmetry object
    def __init__(self, positions = [], world_matrices = None, pg = None, normals = None):
        self.positions = np.array(positions).reshape(-1, 3)
        if world_matrices is None: #4x4 world matrices to hold position and orientation
            m = np.tile(np.eye(4), (len(self.positions), 1, 1))
            m[:, :3, -1] = self.positions
            self.world_matrices = m
        else:
            self.world_matrices = np.array(world_matrices)
        if not pg is None:
            self.pg = PointGroup(pg)
            self.so_matrices = self.pg.get_so_matrices(normals) #Normals is relative to symmetry operations. MUST be provided
            self.expanded_so_matrices = np.concatenate(self.so_matrices, axis = 0) #Transformation matrices associated with each conjugacy class
            self.expanded_conjugacy_classes = self.pg.get_expanded_conjugacy_classes() # Each
            self.get_expanded_characters = self.pg.get_expanded_characters()                                             #(irrep,  so)
            self.transformed_world_matrices = np.einsum("acd,bde->abce", self.expanded_so_matrices, self.world_matrices) #(so, atom, 4, 4)


    @classmethod
    def from_file(cls, filepath: str):
        data = fm.load_molecule_data(filepath)
        atoms, positions = fm.from_file(data["xyz"])[:2]
        return cls(positions, world_matrices = None, pg = data["point_group"], normals = data["normals"])

    def find_landing_spot_projection(self, index: int) -> np.ndarray:
        p = self.transformed_world_matrices[..., :3, [-1]][:, [index], ...] #Final positions #(so, atom, 3, 1)
        d = self.world_matrices[..., :3, [-1]] - p #Displacement
        r = np.linalg.norm(d, axis = -2, keepdims = True) #Distance
        o = np.einsum("abcd,bcd->abd", self.transformed_world_matrices[:, [index], :3, :3], self.world_matrices[:, :3, :3]) #orientation per axes (via dot product along dimension 2)(so, atom, [px,py,pz])
        o = np.insert(o, 0, 1, axis = -1)#orientation per axes (so, atom, [s,px,py,pz])
        landing = (r < 0.1).astype(int) #0 where the atom does not land. 1 where it does
        proj = np.einsum("abcd,abe->abcde",landing,o).squeeze() #(so, atom, [s,px,py,pz])
        return proj

    def find_projection(self, index: int = 0) -> dict:
        proj = self.find_landing_spot_projection(index) #(so, atom, orbitals)
        SALCS = np.einsum("ba,acd->bcd",self.get_expanded_characters, proj) #(Irrep, atom, orbital)
        SALCS = np.round(SALCS, 3)
        result = {}
        for ii, irrep in enumerate(self.pg.irreps):
            result[irrep] = {}
            for io, orbital in enumerate("s px py pz".split()):
                if sum(np.abs(SALCS[ii,:,io])) < .1:
                    continue
                result[irrep][orbital] = SALCS[ii,:,io]
            if len(result[irrep]) == 0:
                del result[irrep]
        return result

    def find_SALCS(self, index: int = 0) -> dict:
        letters = "".join(self.pg.irreps)
        degeneracy = {"A":1,"B":1,"E":2,"T":3,"G":4,"H":5}
        letters = [re.search(r"[ABETGH]",a).group() for a in self.pg.irreps]
        itterations = list(map(lambda a: degeneracy[a], letters))
        n = max(itterations) + 2
        dicts = []
        w = self.world_matrices.copy()
        for i in range(n):
            dicts.append(self.find_projection(index))
            self.world_matrices = np.einsum("bc,acd->abd",self.expanded_so_matrices[i+2], w)
        return dicts

    def filter_SALCS(self, dicts: list) -> dict:
        """ Removes repeated salcs and finds linearly independent combinations where needed"""
        if len(dicts) == 1:
            return dicts[0]
        d0 = dicts[0]
        degeneracy = {"A":1,"B":1,"E":4,"T":5,"G":6,"H":7} # Extra degeneracy added for safety
        letters = [re.search(r"[ABETGH]",a).group() for a in self.pg.irreps]
        itterations = list(map(lambda a: degeneracy[a], letters))
        degenerates = np.array(self.pg.irreps)[np.array(itterations) > 1]
        relevant_counts = np.array(itterations)[np.array(itterations) > 1]
        for (i, d), n in zip(enumerate(degenerates), relevant_counts): #For each element with degeneracy different than 1
            for j in range(n-1): #Add vectors until degeneracy is accounted for
                irrep = dicts[j+1][d]
                for label in irrep: #Group coefficients by label for easy access
                    d0[d][label] = np.vstack((d0[d][label], irrep[label]))
            for label in irrep: #Apply gram smidt orthogonality after all vectors are added and remove non-orthogonals
                d0[d][label] = np.round(self.pg.GramS(d0[d][label]),3)
                d0[d][label] = d0[d][label][np.linalg.norm(d0[d][label], axis=1) > 0.1]
        return d0

    def get_SALCS(self, index: int = 0) -> dict:
        return self.filter_SALCS(self.find_SALCS(index))

if __name__ == "__main__":
    file = os.path.join("molecule_data", "data_dodecahedrane.json")
    pg = PointGroup("D3h")
    benzene = SObject.from_file(file)
    SALC = benzene.get_SALCS(0)
    # print(SALC)
    # a = np.array([te.split("\t") for te in pg.latexify_term(pg.text).split("\n")]).flatten().tolist()
    # a = np.array([f"${e}$" for e in a]).reshape(-1, 13)
    # print(a.tolist())

    pg.latex_complete_table(frame = True)
    print(pg.irreps)
