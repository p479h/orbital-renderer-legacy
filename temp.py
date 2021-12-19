import numpy as np


def fibonacci_sphere(samples=1000):

    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y
        theta = phi * i  # golden angle increment
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append((x, y, z))
    return np.array(points)

def fibonacci_sphere2(samples=1000):
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
    i = np.arange(samples)
    y = 1 - (i/(samples-1))*2
    r = np.sqrt(1 - y**2)
    x, z = np.cos(phi*i)*r, np.sin(phi*i)*r
    return np.array([x, y, z]).T

a = fibonacci_sphere() - fibonacci_sphere2()
print(a)
