from all_imports import *
from scipy.integrate import tplquad
from scipy.optimize import minimize
from scipy.special import assoc_laguerre, lpmv, sph_harm
from math import factorial

try:
    from pytessel import PyTessel
except:
    print("Pytessel could not be imported")
import time

if __name__ == "__main__":
    print(np.__file__)


def s1(r, theta, phi):
    a0 = 0.53 #Angstroms!
    return 1/np.sqrt(np.pi*a0**3) * np.exp(-r/a0)

def s2(r, theta, phi):
    a0 = 0.53 #Angstroms
    return 1/4 * 1/np.sqrt(2*np.pi*a0**3) * (2 - r/a0) * np.exp(-r/(2*a0))


def pz(r, theta, phi):
    a0 = 0.53
    return 1/(4*np.sqrt(2*np.pi*a0**3))*r/a0*np.exp(-r/(2*a0))*np.cos(theta)


def px(r, theta, phi):
    a0 = 0.53
    return 1/(4*np.sqrt(2*np.pi*a0**3))*r/a0*np.exp(-r/(2*a0))*np.sin(theta)*np.cos(phi)


def py(r, theta, phi):
    a0 = 0.53
    return 1/(4*np.sqrt(2*np.pi*a0**3))*r/a0*np.exp(-r/(2*a0))*np.sin(theta)*np.sin(phi)


def dz2(r, theta, phi):
    a0 = 0.53
    rho = 2*r/3/a0
    radial = (1/(9*30**.5))*rho**2*np.exp(-rho/2)
    angular = np.sqrt(5/(16*np.pi))*(3*np.cos(theta)**2 - 1)
    return radial * angular

def dxz(r, theta, phi):
    a0 = 0.53
    rho = 2*r/3/a0
    radial = (1/(9*30**.5))*rho**2*np.exp(-rho/2)
    angular = np.sqrt(15/(4*np.pi))*np.sin(theta)*np.cos(theta)*np.cos(phi)
    return radial * angular

def dyz(r, theta, phi):
    a0 = 0.53
    rho = 2*r/3/a0
    radial = (1/(9*30**.5))*rho**2*np.exp(-rho/2)
    angular = np.sqrt(15/(4*np.pi))*np.sin(theta)*np.cos(theta)*np.sin(phi)
    return radial * angular

def dx2y2(r, theta, phi):
    a0 = 0.53
    rho = 2*r/3/a0
    radial = (1/(9*30**.5))*rho**2*np.exp(-rho/2)
    angular = np.sqrt(15/(16*np.pi))*np.sin(theta)**2*np.cos(2*phi)
    return radial * angular

def dxy(r, theta, phi):
    a0 = 0.53
    rho = 2*r/3/a0
    radial = (1/(9*30**.5))*rho**2*np.exp(-rho/2)
    angular = np.sqrt(15/(16*np.pi))*np.sin(theta)**2*np.sin(2*phi)
    return radial * angular

#Functions from https://en.wikipedia.org/wiki/Table_of_spherical_harmonics
def fy3x2y2(r, theta, phi):
    xr = np.sin(theta)*np.cos(phi)
    yr = np.sin(theta)*np.sin(phi)
    zr = np.cos(theta)
    a0 = 0.53
    rho = 2*r/4/a0
    R = (1/96*np.sqrt(35)) *rho**3*1/2*np.exp(-rho/2)
    return R*1/4*np.sqrt(35/(2*np.pi))*(3*xr**2 - yr**2)*yr

def fxyz(r, theta, phi):
    xr = np.sin(theta)*np.cos(phi)
    yr = np.sin(theta)*np.sin(phi)
    zr = np.cos(theta)
    a0 = 0.53
    rho = 2*r/4/a0
    R = (1/96*np.sqrt(35)) *rho**3*1/2*np.exp(-rho/2)
    return R*1/2*np.sqrt(105/np.pi)*xr*yr*zr

def fyz2(r, theta, phi):
    xr = np.sin(theta)*np.cos(phi)
    yr = np.sin(theta)*np.sin(phi)
    zr = np.cos(theta)
    a0 = 0.53
    rho = 2*r/4/a0
    R = (1/96*np.sqrt(35)) *rho**3*1/2*np.exp(-rho/2)
    return R*1/4*np.sqrt(21/(2*np.pi))*yr*(4*zr**2 - xr**2 - yr**2)

def fz3(r, theta, phi):
    xr = np.sin(theta)*np.cos(phi)
    yr = np.sin(theta)*np.sin(phi)
    zr = np.cos(theta)
    a0 = 0.53
    rho = 2*r/4/a0
    R = (1/96*np.sqrt(35)) *rho**3*1/2*np.exp(-rho/2)
    return R*1/4*np.sqrt(7/np.pi)*zr*(2*zr**2 - 3*xr**2 - 3*yr**2)

def fxz2(r, theta, phi):
    xr = np.sin(theta)*np.cos(phi)
    yr = np.sin(theta)*np.sin(phi)
    zr = np.cos(theta)
    a0 = 0.53
    rho = 2*r/4/a0
    R = (1/96*np.sqrt(35)) *rho**3*1/2*np.exp(-rho/2)
    return R*1/4*np.sqrt(21/(np.pi*2))*xr*(4*zr**2 - xr**2 - yr**2)


def fzx2y2(r, theta, phi):
    xr = np.sin(theta)*np.cos(phi)
    yr = np.sin(theta)*np.sin(phi)
    zr = np.cos(theta)
    a0 = 0.53
    rho = 2*r/4/a0
    R = (1/96*np.sqrt(35)) *rho**3*1/2*np.exp(-rho/2)
    return R*1/4*np.sqrt(105/np.pi)*(xr**2 - yr**2)*zr

def fx3yx2(r, theta, phi):
    xr = np.sin(theta)*np.cos(phi)
    yr = np.sin(theta)*np.sin(phi)
    zr = np.cos(theta)
    a0 = 0.53
    rho = 2*r/4/a0
    R = (1/96*np.sqrt(35)) *rho**3*1/2*np.exp(-rho/2)
    return R*1/4*np.sqrt(35/(np.pi*2))*(xr**2 - 3*yr**2)*xr


functions = [s1, px, py, pz, dz2, dxz, dyz, dx2y2, dxy, px, dz2, fy3x2y2, fxyz, fyz2, fz3, fxz2, fzx2y2, fx3yx2]

##Now we do some ivo stuff
def from_quantum_numbers(n,l,m):
    """
    Construct the wave function
    """
    return lambda r, phi, theta: radial(n,l,r) * angular(l,m,theta,phi) * r

def angular(l,m,theta,phi):
    """
    Construct the angular part of the wave function
    """
    # see: https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form
    #
    # this create so-called Tesseral spherical harmonics
    #
    if m == 0:
        return np.real(sph_harm(m,l,theta,phi))
    elif m > 0:
        return np.real(sph_harm(-m,l,theta,phi) + (-1)**m * sph_harm(m,l,theta,phi)) / np.sqrt(2.0)
    return np.imag(sph_harm(m,l,theta,phi) - (-1)**m * sph_harm(-m,l,theta,phi)) / np.sqrt(2.0)

def radial(n,l,r):
    """
    This is the formulation for the radial wave function as encountered in
    Griffiths "Introduction to Quantum Mechanics 3rd edition"
    """
    a = 1.0
    rho = 2.0 * r / (n * a)
    val =  np.sqrt((2.0 / (n * a))**3) * \
           np.sqrt(factorial(n - l - 1) / (2 * n * factorial(n + l))) * \
           np.exp(-0.5 * rho) * \
           rho**l * \
           assoc_laguerre(rho, n-l-1, 2*l+1)
    return val

def azimuthal(m, phi):
    """
    Construct azimuthal part of the angular wave function
    """
    pre = 1.0 / np.sqrt(4.0 * np.pi)
    if m == 0:
        return pre
    if(m > 0):
        return pre * np.cos(m * phi)
    return pre * np.sin(-m * phi)

def polar(l,m,theta,phi):
    """
    Construct polar part of the angular wave function
    """
    pre = np.sqrt((2 * l + 1) * factorial(l - np.abs(m)) /\
                  factorial(l + np.abs(m)))
    return pre * lpmv(np.abs(m), l, np.cos(theta))



def binatodeci(binary):
    return np.array([binary[i]*2**i for i in range(len(binary))]).sum()



def marching_cubes(field, value, shape, xyz):
    lookuptable = np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1],
    [3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1],
    [3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1],
    [3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1],
    [9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1],
    [9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
    [2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1],
    [8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1],
    [9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
    [4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1],
    [3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1],
    [1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1],
    [4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1],
    [4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
    [5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1],
    [2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1],
    [9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
    [0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
    [2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1],
    [10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1],
    [4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1],
    [5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1],
    [5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1],
    [9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1],
    [0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1],
    [1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1],
    [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1],
    [8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1],
    [2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1],
    [7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1],
    [2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1],
    [11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1],
    [5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1],
    [11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1],
    [11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
    [1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1],
    [9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1],
    [5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1],
    [2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
    [5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1],
    [6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1],
    [3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1],
    [6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1],
    [5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1],
    [1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
    [10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1],
    [6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1],
    [8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1],
    [7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1],
    [3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
    [5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1],
    [0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1],
    [9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1],
    [8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1],
    [5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1],
    [0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1],
    [6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1],
    [10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1],
    [10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1],
    [8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1],
    [1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1],
    [0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1],
    [10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1],
    [3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1],
    [6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1],
    [9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1],
    [8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1],
    [3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1],
    [6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1],
    [0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1],
    [10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1],
    [10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1],
    [2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1],
    [7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1],
    [7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1],
    [2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1],
    [1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1],
    [11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1],
    [8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1],
    [0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1],
    [7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
    [10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
    [2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
    [6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1],
    [7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1],
    [2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1],
    [1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1],
    [10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1],
    [10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1],
    [0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1],
    [7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1],
    [6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1],
    [8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1],
    [9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1],
    [6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1],
    [4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1],
    [10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1],
    [8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1],
    [0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1],
    [1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1],
    [8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1],
    [10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1],
    [4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1],
    [10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
    [5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
    [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1],
    [9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
    [6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1],
    [7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1],
    [3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1],
    [7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1],
    [3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1],
    [6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1],
    [9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1],
    [1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1],
    [4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1],
    [7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1],
    [6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1],
    [3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1],
    [0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1],
    [6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1],
    [0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1],
    [11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1],
    [6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1],
    [5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1],
    [9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1],
    [1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1],
    [1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1],
    [10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1],
    [0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1],
    [5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1],
    [10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1],
    [11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1],
    [9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1],
    [7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1],
    [2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1],
    [8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1],
    [9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1],
    [9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1],
    [1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1],
    [9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1],
    [9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1],
    [5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1],
    [0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1],
    [10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1],
    [2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1],
    [0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1],
    [0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1],
    [9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1],
    [5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1],
    [3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1],
    [5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1],
    [8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1],
    [0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1],
    [9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1],
    [1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1],
    [3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1],
    [4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1],
    [9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1],
    [11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1],
    [11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1],
    [2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1],
    [9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1],
    [3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1],
    [1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1],
    [4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1],
    [4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1],
    [3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1],
    [0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1],
    [9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1],
    [1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]], dtype = np.int32)

    edge_table = np.array([
        [0, 1], #0
        [1, 2], #1
        [2, 3], #2
        [3, 0], #3
        [4, 5], #4
        [5, 6], #5
        [6, 7], #6
        [7, 4], #7
        [4, 0], #8
        [5, 1], #9
        [6, 2], #10
        [7, 3]], dtype = np.int32) #11

    cube_i = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1]], dtype = np.int32)

    corner1, corner2 = np.zeros(3), np.zeros(3)
    iso1, iso2 = 0, 0
    corner1_index, corner2_index = 0, 0
    local_cube = np.zeros((8, 3), dtype = np.float32) #Coordinates of the cube vertices
    cube_values = np.zeros(8, dtype = np.float32) #Function values at the cube edges
    in_range_indices = np.zeros(8, dtype = np.int32)
    edge_indices = np.zeros(16, dtype = np.int32)
    range_8 =np.arange(8, dtype = np.int32)
    current_f = np.zeros(3, dtype = np.int32)
    edge_value = 0
    n_triangles = 0
    n_vertices = 0

    #First we check how much memory will be needed for v and f
    for i in range(shape[0]-1):
        for j in range(shape[1]-1):
            for k in range(shape[2]-1):
                #We assign the cube to the proper isovalues
                in_range_indices *= 0
                for ci in range(8): #Edge indices
                    edge_value = field[i+cube_i[ci, 0],j+cube_i[ci, 1],k+cube_i[ci, 2]]
                    #Check for the indices that satisfy the boundary condition
                    if edge_value<value:
                        in_range_indices[ci] = 1

                index = (in_range_indices*2**range_8).sum()
                edge_indices[:] = lookuptable[index, :]
                for ei in range(0, 15, 3):
                    if -1 == edge_indices[ei]:
                        break
                    n_triangles += 1
                    n_vertices += 1
    print(n_triangles)
    print(n_vertices)

    f = np.zeros((n_triangles, 3), dtype=np.int32)
    v = np.zeros((n_vertices, 3), dtype=np.float32)-1 #Worst case scenario we have the 3 vertices per triangle (not true because some triangles will share vertices
    d2 = np.zeros(n_triangles*3, dtype = np.float32)-1
    thress = 1e-12 if shape[0]>50 else 1e-10
    v_count = 0
    f_count = 0
    for i in range(shape[0]-1):
        for j in range(shape[1]-1):
            for k in range(shape[2]-1):
                #We assign the cube to the proper isovalues
                in_range_indices *= 0
                for ci in range(8): #Edge indices
                    edge_value = cube_values[ci] = field[i+cube_i[ci, 0],j+cube_i[ci, 1],k+cube_i[ci, 2]]
                    local_cube[ci, :] = xyz[i+cube_i[ci, 0],j+cube_i[ci, 1],k+cube_i[ci, 2], :] #We will need these coordinates later
                    #Check for the indices that satisfy the boundary condition
                    if edge_value>value:
                        in_range_indices[ci] = 1

                index = (in_range_indices*2**range_8).sum()
                edge_indices[:] = lookuptable[index, :]
                for ei in range(0, 15, 3):
                    if -1 == edge_indices[ei]:
                        break
                    current_f*=0
                    for ti in range(3): #Index of each triangle vertex
                        #Compute the isovalues in both corners and interpolate to find ideal location to place the triangle
                        corner1_index, corner2_index = edge_table[edge_indices[ti+ei]][0], edge_table[edge_indices[ti+ei]][1]
                        iso1, iso2 = cube_values[corner1_index], cube_values[corner2_index]
                        corner1[:], corner2[:] = local_cube[corner1_index, :], local_cube[corner2_index, :]

                        #For the interpolation we need the actual coordinates
                        interp = (value - iso1)/(iso2 - iso1)*(corner2 - corner1) + corner1

                        if v_count == 0:
                            v[0, :] = interp[:]
                            current_f[ti] = 0
                            v_count +=1
                        else:
                            d2[:v_count] = ((v[:v_count, :]-interp)**2).sum(axis = 1)
                            if np.any(d2[:v_count]<thress):
                                current_f[ti] = np.where(d2<thress)[0][0]
                            else:
                                v[v_count, :] = interp[:]
                                current_f[ti] = v_count
                                v_count += 1

                    f[f_count, :] = current_f[:]
                    f_count += 1

        if f_count == n_triangles:
            print("BROKE CAUSE THERE WERE ENOUGH TRIANGLES")
            print(f_count, "IS the number of faces")
            print(v_count, "IS the number of vertices")
            break
        if v_count == n_vertices:
            print("Broke cause there were enough vertices")
            print(f_count, "IS the number of faces")
            print(v_count, "IS the number of vertices")
            break

    v = v[:v_count]
    f = f[:f_count]
    return v, f


def make_isosurface(xyz, value, ratios, binatodeci, field_func, orbital_func, molecule):
    dX, dY, dZ = xyz[1, 0, 0, 0]-xyz[0, 0, 0, 0], xyz[0, 1, 0, 1]-xyz[0, 0, 0, 1], xyz[0, 0, 1, 2]-xyz[0, 0, 0, 2]
    thress = 1e-10
    cube = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1]])*np.array([[dX, dY, dZ]])

    with open("edges.txt", "r") as f:
        tri_table = np.array(eval(f.read().replace("\t", "")))[:, :-1]

    verts = cube.reshape(8, 1, 1, 1, 3) + xyz
    d = verts - molecule.reshape(-1, 1, 1, 1, 1, 3)
    r = np.linalg.norm(d, axis = 5)
    phi = np.arctan2(d[:, :, :, :, :, 1], d[:, :, :, :, :, 0])
    theta = np.arctan2(np.linalg.norm(d[:, :, :, :, :, :2], axis = 5), d[:, :, :, :, :, 2])
    values_ = (orbital_func(r, theta, phi)*(ratios/np.linalg.norm(ratios)).reshape(-1, 1, 1, 1, 1)).sum(axis = 0)

    edge_table = [
        [0, 1], #0
        [1, 2], #1
        [2, 3], #2
        [3, 0], #3
        [4, 5], #4
        [5, 6], #5
        [6, 7], #6
        [7, 4], #7
        [4, 0], #8
        [5, 1], #9
        [6, 2], #10
        [7, 3]] #11
    f = []
    v = []
    n = []
    for i in range(xyz.shape[0]-1):
        for j in range(xyz.shape[1]-1):
            for k in range(xyz.shape[2]-1):
                values = values_[:, i, j, k]
                if value > 0:
                    within_range = (values<=value).astype(int)
                else:
                    within_range = (values>=value).astype(int)
                if 1 in within_range:
                    tri_index = binatodeci(within_range)
                    edge_indices = tri_table[tri_index][tri_table[tri_index]!=-1]
                    for trio in edge_indices.reshape(-1, 3):
                        indices_f = []
                        for tri_edge in np.array(trio):
                            indices = edge_table[tri_edge]
                            v1, v2 = verts[indices, i, j, k]
                            iso1, iso2 = field_func(np.array([v1, v2]), molecule, orbital_func, ratios)
                            interp = (value - iso1)/(iso2 - iso1)*(v2 - v1) + v1
                            d2 = ((v-interp)**2).sum(axis=1) if len(v) > 0  else None
                            if len(v) == 0 or (not (True in (d2<thress))):
                                v.append(interp)
                                indices_f.append(len(v)-1)
                            else:
                                indices_f.append(np.where(d2<thress)[0][0])
                        p = (v[indices_f[0]]+v[indices_f[1]]+v[indices_f[2]])/3
                        gradient = grad(p, molecule, orbital_func, ratios)
                        if (np.cross(v[indices_f[0]]-v[indices_f[1]], v[indices_f[0]]-v[indices_f[2]])@gradient > 0):
                            indices_f = indices_f[::-1]
                        n.append(gradient)
                        f.append(indices_f)

    return [np.array(v), np.array(f), np.array(n)]
