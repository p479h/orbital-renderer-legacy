import numpy as np
import wavefunctions
try:
    import bpy
    import mathutils
except:
    None
from skimage.measure import marching_cubes
from scipy.interpolate import interp1d
from scipy.special import binom
from Bobject import Bobject
from numba import njit
import time

def fibonacci_sphere(samples=1000):
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
    i = np.arange(samples)
    y = 1 - (i/(samples-1))*2
    r = np.sqrt(1 - y**2)
    x, z = np.cos(phi*i)*r, np.sin(phi*i)*r
    return np.array([x, y, z]).T

class Isosurface(Bobject):
    fib_sphere = fibonacci_sphere(1000)

    def __init__(self, *args, **kwargs):
        if "name" not in kwargs:
            kwargs["name"] = "Orbital"
            self.name = kwargs["name"]
        super().__init__(*args, **kwargs);
        self.collection = self.make_collection(name = self.name)

    @staticmethod
    def bezier(start, finish, n = 30) -> np.ndarray:
        t = np.linspace(0, 1, n);
        x = np.array([start, start, finish, finish])[..., np.newaxis];
        i = np.arange(4).reshape(4, *[1 for i in range(x.ndim-1)])
        P = np.sum(binom(3, i)*(1-t)**(3-i) * t**i * x, axis = 0);
        return P; #Note that n plays the role of the frames

    @staticmethod
    def linear(start, finish, n = 30) -> np.ndarray:
        return np.linspace(start, finish, n)

    @staticmethod
    def cart_to_spherical(xyz):
        "xyz is a vector with xyz coordinates in the last dimension"
        phi = np.arctan2(xyz[..., 1], xyz[..., 0]);
        theta = np.arctan2(np.linalg.norm(xyz[..., :2], axis=-1), xyz[..., 2]);
        r = np.linalg.norm(xyz, axis = -1);
        return r, phi, theta


    @classmethod
    def iso_find2(cls, r, field_func, ratios = 1, atoms = None) -> float:
        """
            Finds maximum isovalue in a sphere when field_func is applied"""
        if type(ratios) in (float, int):
            ratios = np.array([ratios])
        ratios = np.array(ratios).flatten()
        xyz = cls.fib_sphere.reshape(-1, 3)*r; #All points in a sphere
        if not (atoms is None):
            d = xyz - np.array(atoms).reshape(-1, 1, 3);
        else:
            d = xyz[np.newaxis,...]
        r, phi, theta = cls.cart_to_spherical(d)
        values = (field_func(r, theta, phi)*ratios[:, None]).sum(axis=0);
        return np.max(np.abs(values))


    @staticmethod
    def apply_field(grid, molecule, orbital_func, molecule_mat = np.eye(3), inv = [], SALC = [], orbital_orientation_function = lambda a: np.eye(3)) -> np.ndarray:
        """
        xyz: coordinates where field is evaluated 4d np.ndarray
        molecule: coordinates of atoms 2d np.ndarray
        mat: matrix applied to the molecule (such as a rotation)
        orbital_func: function to be applied to the molecule to generate field
        orbital_orientation_function is a function which takes the position of each atom as an argument and returns a corresponding linear transformation for its orbital to be applied to it's field

        returns 4d array with the following indices (atom, x, y, z, xyz) -> field value at cooresponding index in xyz
        to get the overall field, add along dimension 0
        """
        molecule = np.array(molecule)
        if molecule.ndim == 1:
            molecule = molecule.reshape(-1, 3)
        if len(SALC) == 0:
            SALC = np.ones(len(molecule)).astype(np.float32);
        orientation_matrices = np.array([np.linalg.inv(i) for i in map(orbital_orientation_function, molecule)])
        grid_transformed = np.einsum("aef,abcdfe->abcde", orientation_matrices, (grid-molecule.reshape(-1, 1, 1, 1, 3))[..., np.newaxis])+molecule.reshape(-1, 1, 1, 1, 3)
        d = grid_transformed - (molecule_mat@molecule.T).T.reshape(-1, 1, 1, 1, 3).astype(np.float32);
        dist = np.linalg.norm(d, axis = 4)
        phi = np.arctan2(d[..., 1], d[..., 0]);
        theta = np.arctan2(np.linalg.norm(d[..., :2], axis = 4), d[..., 2])
        return orbital_func(dist, theta, phi)*(SALC/np.linalg.norm(SALC)).reshape(-1, 1, 1, 1)

    @staticmethod
    def generate_grid(r: float, n: int) -> np.ndarray:
        """
            returns 4d grid with points inside square of side length 2r and n points along each dimension (x,y,z)
            indexing follows (x, y, z, [x,y,z])
        """
        grids = np.meshgrid(*[np.linspace(-r, r, n) for i in range(3)], indexing = "ij")
        grids = [i[..., None] for i in grids]
        return np.concatenate(grids, axis = 3).astype(np.float32);

    def make_orbital_material(self, sign, copy = True, ivo = False):
        A = {"p":"positive", "n": "negative"}[sign]
        try:
            bpy.data.materials.get(A)#So we can use this class without being in
        except:
            None
        if not bpy.data.materials.get(A) or not copy:
            m = bpy.data.materials.new(name=A);
            m.use_nodes = True;
            if A == "positive":
                rgb = (.16, .47, .8, 1)
                if ivo:
                    rgb = (0.05, 0.24, 0.91, 1)
            else:
                rgb = (.8, .33, .11, 1)
                if ivo:
                    (0.1, 0.419, 0.07, 1)

            m.node_tree.nodes["Principled BSDF"].inputs[0].default_value = rgb;
            if bpy.data.scenes["Scene"].render.engine == "BLENDER_EEVEE":
                m.node_tree.nodes["Principled BSDF"].inputs[5].default_value = .3;
            m.node_tree.nodes["Principled BSDF"].inputs[7].default_value = .1;
            m.node_tree.nodes["Principled BSDF"].inputs[14].default_value = 1;
            m.node_tree.nodes["Principled BSDF"].inputs[15].default_value = .5;
            m.node_tree.nodes["Principled BSDF"].inputs[16].default_value = .5;
            m.use_screen_refraction= True
            m.blend_method = "BLEND"
            m.show_transparent_back = False
        else:
            m = bpy.data.materials.get(A);
        return m;

    def make_mesh(self, verts, faces, sign = "p", offset = [0, 0, 0], scale = 1): #p is "positive"
        """
            makes a mesh from verts and faces"""
        self.deselect_all()
        new_mesh = bpy.data.meshes.new(self.name);
        new_mesh.from_pydata((verts*scale+offset).tolist(), [], faces.tolist());
        new_mesh.update()
        new_object = bpy.data.objects.new('Orbital', new_mesh)
        self.link_obj(new_object, self.collection)
        self.obj = new_object
        self.set_active(new_object)
        self.smooth()
        self.set_origin(new_object);
        return new_object;

    def generate_isosurface(self, material_copy = True, center_origin = True, transition_field = False):
        """
        Applies the marching cubes algorithm to scalarfield with many atoms' vector fields added
        scalarfield: values of wavefunction in the corresponding places in the grid
        grid: (natoms, n, n, n, 3) array with the positions where the wavefunction is evaluated
        isovalue: float with the value where the isosurface is constructed
        material_copy: boolean specifying if every orbital will have it's own instance of the material
        """
        self.deselect_all()
        pair = []
        r = self.r;
        n = self.n;
        grid = self.grid
        isovalue = self.isovalue if transition_field is False else self.current_isovalue
        scalarfield = self.scalarfield if transition_field is False else self.current_scalarfield
        spacing = np.full(3, r*2/(n-1));
        for sign, val in zip(["p", "n"],[1, -1]):
            isovalue*=val;
            try:
                vertices, faces, normals, values = marching_cubes(scalarfield, level = isovalue, spacing = spacing)
            except Exception as f: #If the isovalue is too high/low to begin with:
                print(f, "Could not render orbital. Consider changing isovalue.")
                vertices = np.array([[0, 0, 0]]);
                normals = np.array([]);
                faces = np.array([]);
                values = np.array([]);
            orb = self.make_mesh(vertices - r, faces, sign)
            orb.active_material = self.make_orbital_material(sign, copy = material_copy);
            pair.append(orb)
        self.select(*pair)
        self.set_active(orb)
        bpy.ops.object.join();
        self.set_origin(orb);
        self.obj = orb
        self.mesh = orb
        return orb

class AtomicOrbital(Isosurface):
    def __init__(self, r = 10, n = 30, isovalue = None, position = [0, 0, 0], coeff = 1, field_func = wavefunctions.pz, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.r = r
        self.n = n
        self.coeff = coeff #Coefficient that multiplies the wavefunction
        self.grid = self.generate_grid(self.r, self.n)
        self._field_func = field_func #Wavefunction that takes r theta and phi as arguments
        self._transform_function = lambda atom_pos: np.eye(3)
        self._transform = np.eye(3)
        self.position = np.array(position)
        if isovalue is None:
            self.isovalue = self.iso_find2(self.r, self.field_func, ratios = 1, atoms = self.position)
        else:
            self.isovalue = isovalue
        self.scalarfield = None
        self.current_scalarfield = self.scalarfield
        self.current_isovalue = self.isovalue
        self.transitions = []
        self.transitions2 = False
        self.add_updater(self.updater)

    @property
    def field_func(self):
        return lambda *args, **kwargs: self.coeff*self._field_func(*args, **kwargs)

    @field_func.setter
    def field_func(self, func):
        self._field_func = func
        self._scalarfield = self.apply_field(self.field_func)

    @property
    def scalarfield(self):
        return self._scalarfield

    @scalarfield.setter
    def scalarfield(self, value):
        if value is None:
            self._update_scalarfield()
        else:
            self._scalarfield = value

    def _update_scalarfield(self):
        self._scalarfield = self.apply_field(self.field_func)

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, value):
        if type(value) == np.ndarray or type(value) == mathutils.Matrix:
            self._transform = np.array(value)
        elif value is None:
            self._transform = np.eye(3)
        else:
            self.transform_function = value
            self._transform = value(self.position)
        self._update_scalarfield()

    def mean_inside_cloud(self, cloud, isovalue):
        return np.abs(cloud[np.abs(cloud) > isovalue]).mean()

    def add_transition(self, isovalue = None, duration = 59, wavefunction = wavefunctions.dz2, interpolation = None):
        if isovalue is None:
            isovalue = self.iso_find2(self.r, wavefunction, 1, self.position)
        final_frame = self.get_current_frame() + duration
        if len(self.transitions) > 0:
            last_transition = self.transitions[-1]
            f1, i1, sf1, ff1 = self.get_current_frame(), \
                            last_transition["isovalues"][-1], \
                            last_transition["scalarfields"][-1], \
                            last_transition["wavefunctions"][-1];
        else:
            f1, i1, sf1, ff1 = self.get_current_frame(), \
                             self.isovalue, \
                             self.scalarfield, \
                             self.field_func
        transition = {
            "frames": [f1, final_frame],
            "isovalues": [i1, isovalue],
            "scalarfields": np.array([sf1, self.apply_field(wavefunction)]),
            "interpolation": self.linear if interpolation is None else interpolation,
            "wavefunctions": [ff1, wavefunction]}
        self.transitions.append(transition)
        self.set_frame(final_frame)


    def find_isosign(self, scalarfield, isovalue = None):
        if isovalue is None:
            isovalue = self.isovalue
        is_iterable = False
        try:
            isovalue[0]
            is_iterable = True
        except:
            None
        if is_iterable and scalarfield.ndim == 4:
            isovalue = isovalue.reshape(-1, *np.ones(len(scalarfield)-1).astype(int))

        return (scalarfield > isovalue)+(scalarfield < -isovalue)*-1


    def updater(self, frame):
        transition_function = self.update_mesh
        for active_transition in self.find_active_transitions(frame):
            transition_function(frame, active_transition)

    def find_active_transitions(self, frame):
        transition_ranges = np.array([np.array(t.get("frames"))[[0, -1]] for t in self.transitions])
        within_ranges = (frame<=transition_ranges[:, 1])&(frame>=transition_ranges[:, 0])
        return [t for t, w in zip(self.transitions, within_ranges) if w]


    def update_mesh(self, frame, transition):
        print("THis update mesh is being called")
        limits = transition["frames"]
        duration = np.diff(limits)[0]
        if frame not in range(*[limits[0], limits[1]-1]):
            if frame == limits[-1]-1:
                self.isovalue = transition["isovalues"][-1]
                self.field_func = transition["wavefunctions"][-1]
                self.scalarfield = transition["scalarfields"][-1]
            return
        interp = transition["interpolation"]
        fields = np.fft.fftn(transition["scalarfields"], axes = (-3, -2, -1))
        factor_start = interp(1, 0, duration)[frame-limits[0]]
        factor_end = 1-factor_start
        factors = np.array([factor_start, factor_end]).reshape(2, *[1 for i in range(fields.ndim-1)])
        self.current_isovalue = interp(*transition["isovalues"], duration)[frame-limits[0]]
        self.current_scalarfield = np.real(np.fft.ifftn((factors*fields).sum(0), axes = (-3, -2, -1)))
        self.delete_obj(self.obj, delete_collection = False)
        self.generate_isosurface(transition_field = True)


    def apply_field(self, wavefunction):
        transform = np.linalg.inv(self.transform) #Transform space relative to orbital
        d = np.einsum("de,abce->abcd",transform,self.grid) - self.position
        dist = np.linalg.norm(d, axis = -1)
        phi = np.arctan2(d[..., 1], d[..., 0]);
        theta = np.arctan2(np.linalg.norm(d[..., :2], axis = -1), d[..., 2])
        return wavefunction(dist, theta, phi)



class MolecularOrbital(AtomicOrbital):
    def __init__(self, r, n, field_func, position = np.zeros(3), atom_positions = np.zeros((1, 3)), LC = None, *args, **kwargs):
        super().__init__(r, n, field_func=field_func, position = position, *args, **kwargs)
        self.position = np.array(position)
        self.atom_positions = np.array(atom_positions)
        self.field_func = field_func
        if LC is None:
            LC = np.ones(len(self.atom_positions))
        self.LC = LC
        self.atomic_orbitals = [
            AtomicOrbital(r = r,
                          n = n,
                          isovalue = None,
                          position = p,
                          coeff = coeff,
                          field_func = field_func, *args, **kwargs)
            for p, coeff in zip(atom_positions, LC)
            ]
        self.scalarfield = np.array([
            a.scalarfield for a in self.atomic_orbitals]).sum(0)
        self.isovalue = self.iso_find2(r, atom_positions, field_func, ratios = LC)

def apply_field(grid, orbital_func):
    dist = np.linalg.norm(grid, axis = -1)
    phi = np.arctan2(grid[..., 1], grid[..., 0]);
    theta = np.arctan2(np.linalg.norm(grid[..., :2], axis = -1), grid[..., 2])
    return orbital_func(dist, theta, phi)
