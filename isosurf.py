import numpy as np
import wavefunctions
import bpy
import mathutils
from skimage.measure import marching_cubes
from scipy.optimize import minimize, brute
from Bobject import Bobject
from numba import njit
import time

class Isosurface(Bobject):
    def __init__(self, *args, **kwargs):
        if "name" not in kwargs:
            kwargs["name"] = "Orbital"
            self.name = kwargs["name"]
        super().__init__(self, *args, **kwargs);
        self.collection = self.make_collection(name = self.name)

    @staticmethod
    def bezier(start, finish, n = 30) -> np.ndarray:
        t = np.linspace(0, 1, n);
        x = np.array([start, start, finish, finish]);
        P = (1-t)**3*x[0] + 3*(1-t)**2*t*x[1] +3*(1-t)*t**2*x[2] + t**3*x[3];
        return P; #Note that n plays the role of the frames

    @staticmethod
    def iso_find2(r, a, field_func, ratios = 1) -> float:
        print(ratios)
        if type(ratios) in (float, int):
            ratios = np.array([ratios])
        ratios = np.array(ratios).flatten()
        phi, theta = np.mgrid[.05:np.pi:30j, 0:np.pi*2:60j];
        x, y, z = r*np.cos(phi)*np.sin(theta), r*np.sin(theta)*np.sin(phi), r*np.cos(theta);
        x, y, z = [i.reshape(30, 60, 1) for i in (x, y, z)]
        xyz = np.concatenate((x, y, z), axis = 2).reshape(-1, 1, 3); #All points in a sphere
        d = xyz - a;
        phi = np.arctan2(d[..., 1], d[..., 0])
        theta = np.arctan2(np.linalg.norm(d[..., :2], axis=2), d[..., 2]);
        r = np.linalg.norm(d, axis = 2)
        values = (field_func(r, theta, phi)*ratios/np.linalg.norm(ratios)).sum(axis=1);
        return max(abs(values.max()), abs(values.min()))

    @classmethod
    def iso_find_mean(cls, grid, molecule, orbital_func, molecule_mat = np.eye(3), inv = [], SALC = [], orbital_orientation_function = lambda a: np.eye(3)) -> float:
        return np.abs(cls.apply_field(grid, molecule, orbital_func, molecule_mat, inv, SALC, orbital_orientation_function)).mean();

    @classmethod
    def iso_find_mean2(cls, grid, molecule, orbital_func, molecule_mat = np.eye(3), inv = [], SALC = [], orbital_orientation_function = lambda a: np.eye(3)) -> float:
        return (np.abs(cls.apply_field(grid, molecule, orbital_func, molecule_mat, inv, SALC, orbital_orientation_function))**.5).mean()**2;

    @classmethod
    def iso_find_vert(cls, Npoints, grid, molecule, orbital_func, molecule_mat = np.eye(3), inv = [], SALC = [], orbital_orientation_function = lambda a: np.eye(3), resolution = 100) -> float:
        scalarfield = cls.apply_field(grid, molecule, orbital_func, molecule_mat=molecule_mat, inv=inv, SALC=SALC, orbital_orientation_function = orbital_orientation_function)
        r = grid.max();
        n = len(grid);
        spacing = np.full(3, r*2/(n-1));
        def errorFunc(iso_guess):
            try:
                vertices, faces, normals, values = marching_cubes(scalarfield.sum(0), level = iso_guess[0], spacing = spacing);
                n_points = len(vertices)
            except Exception as f:
                n_points = -1;
            return n_points
        iso_guesses = np.linspace(0, 1, resolution).reshape(-1, 1);
        counts = np.array([errorFunc(i) for i in iso_guesses])
        lims = iso_guesses[counts>0, :].min(), iso_guesses[counts>0, :].max()
        iso_guesses2 = np.linspace(*lims, resolution).reshape(-1, 1);
        counts = np.array([errorFunc(i) for i in iso_guesses2])
        return iso_guesses2[np.argmin(np.abs(counts - Npoints))]
    
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
        phi = np.arctan2(d[:, :, :, :, 1], d[:, :, :, :, 0]);
        theta = np.arctan2(np.linalg.norm(d[:, :, :, :, :2], axis = 4), d[:, :, :, :, 2])
        return orbital_func(dist, theta, phi)*(SALC/np.linalg.norm(SALC)).reshape(-1, 1, 1, 1)



    @staticmethod
    def generate_grid(r: float, n: int) -> np.ndarray:
        """
            returns 4d grid with points inside square of side length 2r and n points along each dimension (x,y,z)
            indexing follows (x, y, z, [x,y,z])
        """
        X, Y, Z = np.mgrid[-r:r:(n*1j),-r:r:(n*1j),-r:r:(n*1j)];
        x, y, z = [i.reshape(n, n, n, 1) for i in (X, Y, Z)];
        return np.concatenate((x, y, z), axis = 3).astype(np.float32);
    
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
        self.set_obj(new_object)
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
                print(f)
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
        self.set_obj(orb)
        self.mesh = orb
        return orb        

class AtomicOrbital(Isosurface):
    def __init__(self, r = 10, n = 30, isovalue = None, position = [0, 0, 0], coeff = 1, field_func = wavefunctions.pz, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.r = r
        self.n = n
        self.coeff = coeff #Coefficient that multiplies the wavefunction
        self.grid = self.generate_grid(self.r, self.n)
        self.position = np.array(position)
        self.field_func = field_func #Wavefunction that takes r theta and phi as arguments
        self.transform_function = lambda atom_pos: np.eye(3)
        self.transform = None
        if isovalue is None:
            self.isovalue = self.iso_find2(self.r, self.position, self.field_func, 1)
        else:
            self.isovalue = isovalue
        self.scalarfield = None
        self.current_scalarfield = self.scalarfield
        self.current_isovalue = self.isovalue
        self.transitions = []
        self.add_updater(self.updater)

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
            
        self.scalarfield = None #update scalarfield

    def add_transition(self, isovalue = None, duration = 59, wavefunction = wavefunctions.dz2):
        if isovalue is None:
            isovalue = self.iso_find2(self.r, self.position, wavefunction, 1)
            final_frame = self.get_current_frame() + duration
        if len(self.transitions) > 0:
            last_transition = self.transitions[-1]
            f1, i1, sf1, ff1 = last_transition["frames"][1], \
                            last_transition["isovalues"][1], \
                            last_transition["scalarfields"][1], \
                            last_transition["wavefunctions"][1];
        else:
            f1, i1, sf1, ff1 = self.get_current_frame(), \
                             self.isovalue, \
                             self.scalarfield, \
                             self.field_func
        transition = {
            "frames": [f1, final_frame],
            "isovalues": [i1, isovalue],
            "scalarfields": np.array([sf1, self.apply_field(wavefunction)]),
            "interpolation": self.bezier,
            "wavefunctions": [ff1, wavefunction]}
        self.transitions.append(transition)
        self.set_frame(final_frame)

    def updater(self, frame):
        for active_transition in self.find_active_transitions(frame):
            self.update_mesh(frame, active_transition)

    def find_active_transitions(self, frame):
        transition_ranges = np.array([t.get("frames") for t in self.transitions])
        within_ranges = (frame<=transition_ranges[:, 1])&(frame>=transition_ranges[:, 0])
        return [t for t, w in zip(self.transitions, within_ranges) if w]
        

    def update_mesh(self, frame, transition):
        limits = transition["frames"]
        duration = np.diff(limits)[0]
        if frame not in range(*[limits[0], limits[1]-1]):
            print("GOT HERE")
            print(limits)
            if frame == limits[-1]-1:
                self.isovalue = transition["isovalues"][1]
                self.field_func = transition["wavefunctions"][1]
                self.scalarfield = transition["scalarfields"][1]
            return
        interp = transition["interpolation"]
        fields = transition["scalarfields"]
        factor_start = interp(1, 0, duration)[frame-limits[0]]
        factor_end = 1-factor_start
        factors = np.array([factor_start, factor_end]).reshape(2, *[1 for i in range(fields.ndim-1)])
        self.current_isovalue = interp(*transition["isovalues"], duration)[frame-limits[0]]
        self.current_scalarfield = (factors*fields).sum(0)
        self.delete_obj(self.obj, delete_collection = False)
        self.generate_isosurface(transition_field = True)
        

    def apply_field(self, wavefunction):
        transform = np.linalg.inv(self.transform) #Transform space relative to orbital
        d = np.einsum("de,abce->abcd",transform,self.grid) - self.position
        dist = np.linalg.norm(d, axis = -1)
        phi = np.arctan2(d[..., 1], d[..., 0]);
        theta = np.arctan2(np.linalg.norm(d[..., :2], axis = -1), d[..., 2])
        return wavefunction(dist, theta, phi)
        

    @property
    def field_func(self):
        return self._field_func

    @field_func.setter
    def field_func(self, func):
        self._field_func = lambda *args, **kwargs: self.coeff*func(*args, **kwargs)

    @property
    def scalarfield(self):
        return self._scalarfield

    @scalarfield.setter
    def scalarfield(self, value):
        if value is None:
            self._scalarfield = self.apply_field(self.field_func)
        else:
            self._scalarfield = value

    

