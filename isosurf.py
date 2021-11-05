import numpy as np
from wavefunctions import *
from skimage.measure import marching_cubes
from scipy.optimize import minimize, brute
from numba import njit

class Isosurface:
    def __init__(self):
        pass

    @staticmethod
    def bezier(start, finish, n = 30) -> np.ndarray:
        t = np.linspace(0, 1, n);
        x = np.array([start, start, finish, finish]);
        P = (1-t)**3*x[0] + 3*(1-t)**2*t*x[1] +3*(1-t)*t**2*x[2] + t**3*x[3];
        return P; #Note that n plays the role of the frames

    @staticmethod
    def iso_find2(r, a, field_func , ratios) -> float:
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

    @classmethod
    def iso_find_mean(cls, grid, molecule, orbital_func, molecule_mat = np.eye(3), inv = [], SALC = [], orbital_orientation_function = lambda a: np.eye(3)) -> float:
        return np.abs(cls.apply_field(grid, molecule, orbital_func, molecule_mat, inv, SALC, orbital_orientation_function)).mean();

    @classmethod
    def iso_find_mean2(cls, grid, molecule, orbital_func, molecule_mat = np.eye(3), inv = [], SALC = [], orbital_orientation_function = lambda a: np.eye(3)) -> float:
        return (np.abs(cls.apply_field(grid, molecule, orbital_func, molecule_mat, inv, SALC, orbital_orientation_function))**.5).mean()**2;

    @classmethod
    def iso_find_vert(cls, Npoints, grid, molecule, orbital_func, molecule_mat = np.eye(3), inv = [], SALC = [], orbital_orientation_function = lambda a: np.eye(3)) -> float:
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
                return-1
            return np.abs(n_points-Npoints)
        iso_guesses = np.linspace(0, 1, 300).reshape(-1, 1);
        counts = np.array([errorFunc(i) for i in iso_guesses])
        lims = iso_guesses[counts>0, :].min(), iso_guesses[counts>0, :].max()
        return brute(errorFunc, ranges = [lims], Ns = 10)[0]

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
