from all_imports import *

#First we setup a system that allows for reloading of this script
import inspect
for frame in inspect.stack()[1:]:
    if frame.filename[0] != '<':
        caller = os.path.basename(frame.filename)
        break
this_script = os.path.basename(__file__) #The first time we run it we get access to blender_imports! The second time, nothing hapens hahah. I BEAT PYTHON+BLENDER IN THEIR OWN SICK GAME!
if "caller" in locals() and (caller  != this_script): #THis will only be called if this file is imported through blender!
    import blender_imports

    #Then we do the actual imports without worrying about recursion
    import importlib
    import Molecule
    import PointGroup
    import BText
    import Bobject
    import isosurf
    import wavefunctions
    LIBs = (Molecule, PointGroup, wavefunctions, isosurf, Bobject, BText, blender_imports)
    def reload_libs(LIBs):
        def closure():
            for LIB in LIBs:
                importlib.reload(LIB)
        return closure #Has LIB trapped inside
    reload_libs = reload_libs(LIBs) #This needs to be called from within blender for technical reasons
    from importlib import *
    from Molecule import *
    from PointGroup import *
    from BText import *
    from Bobject import *
    from isosurf import *
    from wavefunctions import *
