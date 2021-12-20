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
    def __init__(self, file):
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
    def normal_to_directory(self, term):
        """
            Takes term and writes it in the same way as in the directory"""
        return term.replace("'", "_prime")

    @classmethod
    def directory_to_normal(self, term):
        """
            Takes a directory name and writes it as it will be read from the point group files"""
        return term.replace("_prime","'")

    def read_conjugacy_classes(self, file):
        with open(file, "r") as f:
            return f.readline().split()[1:]

    def read_pg_name(self, file):
        with open(file, "r") as f:
            return f.readline().split()[0]

    @classmethod
    def find_angles(cls, conjugacy_classes): #Find angles of rotation for the conjugacy_classes
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

    def make_matrices_from_normals(self, normals, angles):
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

    def get_so_matrices(self, normals):
        return self.make_matrices_from_normals(normals, self.find_angles(self.conjugacy_classes))


    def expand(self): #Was used for some early animations. I think it is a good idea to keep it
        s = []
        pattern = re.compile("([0-9]+)([A-Z]|sig|Sig|sigma|Sigma)(\S+)")
        for cls, count in zip(self.conjugacy_classes, self.conjugacy_counts):
            s = s + [pattern.sub("\2\3",cls)]*count
        return s

    def find_landing_spot(self, orbitals, matrix, orbital_index = 0):
        orbital = orbitals[orbital_index]
        diff = (matrix@orbital)[:3, 3] - orbitals[:, :3, 3]
        distance = np.linalg.norm(diff, axis = 1, keepdims = False)
        ending_orbital = np.argmin(distance)
        return ending_orbital

    def find_landnig_spots(self, orbitals, matrices, orbital_index = 0):
        final_orbitals = np.einsum("abc,acd->abd", matrices.reshape(-1, 4, 4), orbitals[[orbital_index], ...])

        ## To be continued

    def find_projected_p_coeffs(self, orbital, matrix):
        mask = np.eye(4, dtype = bool)
        mask[3, 3] = False
        return (orbital.T@matrix)[mask]

    def count_remaining_p_orbitals(self, orbitals, matrix):
        transformed_orbitals = np.einsum('ij,kjl->kil',matrix, orbitals)
        error = np.linalg.norm(transformed_orbitals[:, :3, 3] - orbitals[:, :3, 3], axis = 1)
        projected = np.einsum('ij,kil->kjl',matrix, orbitals)
        projected[error>1e-1]*=0
        mask = np.eye(4, dtype = bool)
        mask[3, 3] = False
        mask = np.tile(mask, (len(projected), 1, 1))
        s = projected[mask].sum()
        return int(round(s, 0)) if np.abs(float(s) - round(s, 0)) < error.max()/1000 else s #Easier to read afterwards

    def count_remaining_s_orbitals(self, orbitals, matrix):
        new_positions = (matrix[:3, :3]@orbitals[:, :3, 3].T).T
        distance = np.linalg.norm(orbitals[:,:3,3] - new_positions, axis = 1, keepdims = False)
        return np.ones(len(new_positions))[distance<1e-1].sum()

    def create_orbitals(self, vertices, orientations = [None]):
        orbitals = np.tile(np.eye(4), (len(vertices), 1, 1))
        for i, v in enumerate(vertices):
            orbitals[i, :3, 3] = v[:3]
            orbitals[i, :3, :3] = np.eye(3) if orientations[0] == None else orientations[i]
        return orbitals

    def find_reducible_representation(self, orbitals):
        """
            Orbitals must be 4x4 matrices"""
        traces = np.zeros(len(self.matrices))
        tracep = np.zeros(len(self.matrices))
        for ci, c in enumerate(self.matrices): #For each  conjugacy class
            for ti, t in enumerate(c): #For each transformation in that conjugacy class
                traces[ci] += self.count_remaining_s_orbitals(orbitals, t)
                tracep[ci] += self.count_remaining_p_orbitals(orbitals, t)
        return traces, tracep

    def apply_projection_operator(self, orbitals, orbital_index = 0):
        projx = np.zeros((len(orbitals), len(self.conjugacy_counts)))
        projy = projx.copy()
        projz = projx.copy()
        projs = projx.copy()

        proj = [projx, projy, projz]
        for cci, cc in enumerate(self.matrices): #For each conjugacy class
            for t in cc: #For each transformation in that cclss
                landing = self.find_landing_spot(orbitals, t, orbital_index)
                coeffs = self.find_projected_p_coeffs(orbitals[orbital_index], t)
                projs[landing, cci] += 1 # For the s orbital
                for i in range(3):
                    proj[i][landing, cci] += coeffs[i]

        return projs, projx, projy, projz

    def print_salcs(self, orbitals, trace, orbital_index = 0):
        projs, projx, projy, projz = self.apply_projection_operator(orbitals, orbital_index)
        SALCS = {}
        labels = []
        dot = (trace*self.characters).sum(axis = 1)/self.h
        for ci, c in enumerate(self.characters):
            if dot[ci] < 0.1:
                continue
            salcs = (projs*c).sum(axis = 1)
            salcx = (projx*c).sum(axis = 1)
            salcy = (projy*c).sum(axis = 1)
            salcz = (projz*c).sum(axis = 1)
            if "E" in self.irreps[ci]:
                f = self.linear_independenceE
            elif "T" in self.irreps[ci]:
                f = self.linear_independenceT
            elif "G" in self.irreps[ci]:
                f = self.linear_independenceG
            elif "H" in self.irreps[ci]:
                f = self.linear_independenceH
            else:
                print(self.irreps[ci])
                SALCS[self.irreps[ci]] = {}
                for i, salc in enumerate([salcs, salcx, salcy, salcz]):
                    label = "2s x y z".split()[i]
                    salc = np.around(salc, 6).tolist()
                    SALCS[self.irreps[ci]][label] = salc
                    print(label, salc)
                    print()
                continue
            salcs_and_labels = [f(s, self.irreps[ci], orbitals, self.matrices[1]) for s in [salcs,salcx,salcy,salcz]]
            print(salcs_and_labels[0][0])
            for i in range(len(salcs_and_labels[0][0])):
                print(salcs_and_labels[0][0][i])
                SALCS[salcs_and_labels[0][0][i]] = {}
                for j in range(4):
                    label = "2s x y z".split()[j]
                    salc = np.around(salcs_and_labels[j][1][i], 6).tolist()
                    SALCS[salcs_and_labels[0][0][i]][label] = salc
                    print(label,salc)
                    print()
        return SALCS

    def make_rotation(self, normal, angle): #This and the following "tranformation functions" generate 4x4 matrices that can be appplied to the orbitals
        if len(normal.shape) == 1:
            return get_rotation_around(angle, 4, normal)
        else:
            return np.array([get_rotation_around(angle, 4, norm) for norm in normal])

    def make_reflection(self, normal, angle = 0):
        if len(normal.shape) == 1:
            return np.array(get_reflection_across(-1, 4, normal))
        else:
            return np.array([get_reflection_across(-1, 4, norm) for norm in normal])

    def make_inversion(self, normal = [], angle = 0):
        i = np.eye(4)*-1
        i[3, 3] = 1
        return i

    def make_identity(self, normal = [], angle = 0):
        return np.eye(4)

    def make_improper_rotation(self, normal, angle):
        if len(normal.shape) == 1:
            return np.array(self.make_reflection(normal)@self.make_rotation(normal, angle))
        else:
            rotations = self.make_rotation(normal, angle)
            reflections = self.make_reflection(normal)
            return np.array([r@rot for r, rot in zip(reflections, rotations)])

    def get_irreps(self, trace):
        counts = (trace*self.characters).sum(axis = 1)/self.h
        return [str(int(round(c, 0)))+n for c, n in zip(counts, self.irreps) if c > 0 ]


    def count_mos(self, trace):
        return ((trace*self.characters).sum(axis=1)/self.h * self.characters[:, 0]).sum()

    def read_irreps(self, file):
        with open(file, "r") as f:
            f.readline()
            return [line.split()[0] for line in f.readlines()]

    def read_characters(self, file):
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

    def count_conjugacy(self, classes):
        counts = []
        pattern = re.compile("([0-9]+)\(?([A-Z]|sig|Sig|sigma|Sigma)")
        for cl in classes:
            result = 1 if not pattern.search(cl) else int(pattern.search(cl).group(1))
            counts.append(result)
        return np.array(counts)


    def find_irreps(self, reducible_representation):
        return (self.characters*reducible_representation.ravel()).sum(axis=1)/self.h


    def set_normals(self, normals): #All normals!!! For all conjugacy classes
        for i, n_group in enumerate(normals):
            self.normals[i] = n_group

    def set_naxis(self, n, conj_cls = 0):
        self.naxis[conj_cls] = n

    def get_header(self):
        return " | ".join([self.pg_name]+self.conjugacy_classes)

    def permute_atoms(self, orbitals, matrix): # Used for finding the linearly independent SALCS
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

    def latexify_term(self, t):
        t = t.replace("sigma", "sig").replace("Sigma", "sig").replace("infty", "inf").replace("Infty", "inf")
        #Takes a term t and makes it latex friendly (Could be improved I think)
        pattern = r"([0-9]+)?(\(?)([A-Z]|Sig|sig|sigma|Sigma)('*)([a-z0-9]+|Inf|inf|infty|Infty)?(\))?([0-9]+)?"
        pattern = re.compile(pattern)
        subbed = pattern.sub(r"\1\2\3_{\5}\6^{\4\7}", t)
        if subbed.find(")") != -1:
            subbed = pattern.sub(r"\1\2\3^{\4}_{\5}\6^{\7}", t)
        return subbed.replace("^{}", "").replace("_{}", "").replace("sig", r"\sigma ").replace("inf", r"\infty").replace("'", r"\prime ")

    def latexify_terms(self, terms):
        return [self.latexify_term(t) for t in terms]

    def find_normals(self, m, collinear_counts = [1, 0, 0, 2, 0, 0, 0, 0, 2, 0], collinear_bonds = []): #This one is for dodecahedrane
        """
            Finds all relevant normals of molecule
            collinear_counts are the number of atoms through the normal that describes each transformatio"""

        #Creating scattered initial guesses
        thetas = np.linspace(0, np.pi, 60)
        phis = np.linspace(0, np.pi*2, 80)
        starting_angles = np.array([[theta, phi] for theta in thetas for phi in phis])
        np.random.shuffle(starting_angles)

        r = []
        bonds_positions = []
        for p in m.position:
            r.append(p.tolist())
        try:
            for c in m.connections:
                r.append(((m.position[c[0]]+m.position[c[1]])/2).tolist())
                bonds_positions.append(r[-1])
        except:
            print("Could not use the connections")
        bonds_positions = np.array(bonds_positions)
        r = np.array(r)
        r_angles = np.zeros((len(r),2))
        r_angles[:, 0] = np.arctan2(r[:, 2], np.linalg.norm(r[:, :2],axis = 1))
        r_angles[:, 1] = np.arctan2(r[:, 1], r[:, 0])
        r_angles[r_angles<0] = r_angles[r_angles<0] + np.pi
        starting_angles = np.array(r_angles.tolist() + starting_angles.tolist())
        np.random.shuffle(starting_angles)

        #Defining functions used by the solver
        def error_func(angles, atom_pos, rot_angle, make_matrix):
            theta, phi = angles
            axis = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])
            matrix = make_matrix(axis, rot_angle)[:3, :3]
            new_pos = (matrix@atom_pos.T).T
            permuted = new_pos*0
            for i, p in enumerate(atom_pos):
                permuted[i, :] = new_pos[np.argmin(np.linalg.norm(p - new_pos, axis = 1))]
            error = np.linalg.norm(permuted - atom_pos, axis = 1).sum()
            return error

        def count_collinear_atoms(normal, atoms): #Atoms is any array of points!
            atoms = atoms/np.linalg.norm(atoms, axis = 1, keepdims = True)
            atoms = np.nan_to_num(atoms)
            count= 0
            for ia, a in enumerate(atoms):
                if ((a*normal).sum()>.99) or ((a*normal).sum()<-.99 or np.abs(a).sum() < 1e-6):
                    count +=1
            return count

        #The matrix used depends on the kind of operation
        matrix_makers = {
                        "C": self.make_rotation,
                        "sig": self.make_reflection,
                        "S": self.make_improper_rotation}
        axes = []
        kind_history = []
        for ci, cc in enumerate(self.conjugacy_classes):
            sas = []
            kind = re.search(r"[E|C|S|i]|sig", cc).group(0)
            kind_history.append(kind)
            if kind == "E" or kind == "i":
                axes.append([])
                continue
            angles = self.find_angles()
            matrix_maker = matrix_makers[kind]
            print(cc, " Just started")
            for sai, sa in enumerate(starting_angles):
                results = scipy.optimize.minimize(error_func, sa, (m.position, angles[ci], matrix_maker), bounds = ((0, np.pi*1.1),(0, np.pi*2.2)), method = "Nelder-Mead", tol = 1e-7*len(m.position))
                if results.success and results.fun < .1:
                    print(results.fun, "accepted error")
                    theta, phi = results.x
                    axis = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])
                    if kind == "sig":
                        results = np.array([np.dot(axis, v) for ind, vs in enumerate(axes) if kind_history[ind] == "sig" for v in vs])
                        if np.any(((results<-.98)|(results>.98))):
                            continue
                    if len(sas) == 0:
                        if (count_collinear_atoms(axis, m.position) == collinear_counts[ci]):
                            if len(collinear_bonds)> 0 and len(bonds_positions)>0:
                                if not (count_collinear_atoms(axis, bonds_positions) == collinear_bonds[ci]):
                                    continue
                            sas.append(list(axis))
                        continue
                    if ((np.array(sas)*axis).sum(axis = 1) < .98).all() and (count_collinear_atoms(axis, m.position) == collinear_counts[ci]):
                        if len(collinear_bonds)>0 and len(bonds_positions)>0:
                            if not (count_collinear_atoms(axis, bonds_positions) == collinear_bonds[ci]):
                                continue
                        if abs(angles[ci]-np.pi)<1e-4 or kind == "sig":
                            # We have a C2 rotation. So we can not allow antiparallel
                            axis = self.reorient_normals([axis])[0]
                            if (True in (((np.array(sas)*axis).sum(axis = 1) < -.98)|((np.array(sas)*axis).sum(axis = 1) > .98))):
                                continue
                        sas.append(list(axis))
                    if len(sas) == self.conjugacy_counts[ci]:
                        print("Found all prematurely yeet!(this is good btw)")
                        break
                if sai == len(starting_angles)-1:
                    print("Failure with ", cc)
            sas = np.array(sas)
            sas[np.abs(sas)<1e-10] = 0
            sas[np.abs(sas)>.99] = np.around(sas[np.abs(sas)>.99], 1)
            sas = np.around(sas, 6)
            axes.append(self.pair_up(self.sort_normals(sas)).tolist())
        return axes

    def reorient_normals(self, normals):
        """
            Places all normals above the xy plane"""
        n = np.array(normals)
        mask = n[:, 2]<0
        n[mask] = n[mask]*-1
        return n

    def sort_by_height(self, normals, n_levels = 20):
        n = np.array(normals)
        copy = n.tolist()
        zs = n[:, 2]
        zmax = zs.max()
        zmin = zs.min()
        if zmax-zmin < 1:
            zmax = zmax + 2
            zmin = zmin - 1
        bins = np.linspace(zmax, zmin, n_levels)
        levels = [[] for i in range(n_levels)]
        for n_, z in zip(copy, zs):
            levels[np.argmin(np.abs(z-bins))].append(n_)
        levels = [l for l in levels if len(l) > 0]
        return levels

    def sort_by_angle(self, normals, n_angles = 20):
        n = np.array(normals)
        copy = n.tolist()
        xy = n[:, :2]/np.linalg.norm(n[:, :2], axis = 1, keepdims = True)
        xy = np.nan_to_num(xy)
        bins = np.linspace(0, np.pi*2, n_angles)
        levels = [[] for i in range(n_angles)]
        for xy_, n_ in zip(xy, n):
            ang = np.arctan2(xy_[1], xy_[0]) + np.pi
            levels[np.argmin(np.abs(bins - ang))].append(n_)
        levels = [l for l in levels if len(l) > 0]
        return levels

    def pair_up(self, normals):
        print(normals)
        print("given")
        normals2 = []
        indices_to_avoid = [] #This only works well because all of the vectors are unit vectors!!! More stuff needs to be added if we work with non unit vectors
        for i, n in enumerate(normals):
            if i in indices_to_avoid:
                continue
            normals2.append(n)
            if i < len(normals)-1:
                dots = (n.reshape(1, 3)@normals[i+1:, :].reshape(-1, 3).T).flatten()
                if np.any(dots<-.99):
                    index = np.where(dots<-.99)[0][0] + i + 1
                    indices_to_avoid.append(index)
                    normals2.append(normals[index])

        print(np.array(normals2))
        print("Returned")
        return np.array(normals2)


    def sort_normals(self, normals):
        levels = self.sort_by_height(normals)
        levels = [self.sort_by_angle(level) for level in levels]
        normals = np.array([subsub for level in levels for sub in level for subsub in sub])
        return normals

    def GramS(self, X, row_vecs=True, norm = True): #I got this one online. Writen by ingmarschuster
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

    def latex_table(self, text = None, frame = False):
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


    def get_expanded_conjugacy_classes(self):
        """ Brings all transformations individually e.g. E C C C instead of E 3C"""
        new_columns = []
        for i, cc in enumerate(self.conjugacy_counts):
            kind = re.search(r"([ECSi]['0-9']*)|sig[vdh]*", self.conjugacy_classes[i]).group(0)
            for j in range(cc):
                new_columns.append(kind)
        return new_columns

    def get_expanded_characters(self):
        new_columns = np.zeros((self.characters.shape[0], self.conjugacy_counts.sum()))
        j = 0
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
    def from_datafile(cls, filepath):
        data = fm.load_molecule_data(filepath)
        atoms, positions = fm.from_file(data["xyz"])[:2]
        return cls(positions, world_matrices = None, pg = data["point_group"], normals = data["normals"])

    def find_landing_spot_projection(self, index):
        p = self.transformed_world_matrices[..., :3, [-1]][:, [index], ...] #Final positions #(so, atom, 3, 1)
        d = self.world_matrices[..., :3, [-1]] - p #Displacement
        r = np.linalg.norm(d, axis = -2, keepdims = True) #Distance
        o = np.einsum("abcd,bcd->abd", self.transformed_world_matrices[:, [index], :3, :3], self.world_matrices[:, :3, :3]) #orientation per axes (via dot product along dimension 2)(so, atom, [px,py,pz])
        o = np.insert(o, 0, 1, axis = -1)#orientation per axes (so, atom, [s,px,py,pz])
        landing = (r < 0.1).astype(int) #0 where the atom does not land. 1 where it does
        proj = np.einsum("abcd,abe->abcde",landing,o).squeeze() #(so, atom, [s,px,py,pz])
        return proj

    def find_projection(self, index = 0):
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
        return result


if __name__ == "__main__":
    file = os.path.join("molecule_data", "data_benzene.json")
    pg = PointGroup("D6h")
    benzene = SObject.from_datafile(file)
    p = benzene.find_projection(0)
    for a in p:
        if "s" in p[a]:
            print(a)
            print(p[a])
    # a = np.array([te.split("\t") for te in pg.latexify_term(pg.text).split("\n")]).flatten().tolist()
    # a = np.array([f"${e}$" for e in a]).reshape(-1, 13)
    # print(a.tolist())
    # pg.latex_table()
