import numpy as np
#Formula from:
#https://en.wikipedia.org/wiki/Transformation_matrix#Rotation

def get_rotation_around(angle, axis):
    axis = np.array(axis)/np.linalg.norm(np.array(axis))
    l, m, n = axis
    diag = np.eye(3)*np.cos(angle)
    rest = (np.ones((3, 3))-np.eye(3))*np.sin(angle)
    angular_part = diag+rest
    ll = np.array(axis)*np.array(axis)[:, None]*(1-np.cos(angle))
    rr = np.array([[1, -n, m], [n, 1, -l], [-m, l, 1]])*angular_part
    return ll + rr

def get_reflection_across(_, axis):
    axis = np.array(axis)
    axis = axis/np.linalg.norm(axis)
    N = axis[None, :]
    A = (np.eye(3)-2*N*N.T)
    return A

print(get_reflection_across(1, [1, 2, 1]))
