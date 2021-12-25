try: #We are inside blender
    import mathutils
    import bpy
except:#We are on matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import interp1d
from scipy.special import binom
from skimage import measure
from numba import njit
import numpy as np
import itertools
import importlib
import json
import time
import os
import re
