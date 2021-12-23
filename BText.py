from all_imports import *
from Bobject import Bobject

class BText(Bobject): #Blender Text
    def __init__(self, text = "+", font = "", camera_align = True, *args, **kwargs):
        super().__init__(None, *args, **kwargs)
        self.text = text
        self.font = font
        if camera_align:
            self.add_updater(self.align_to_camera)

    def draw(self, assign_collection = True):
        self.font_curve = bpy.data.curves.new(type="FONT", name="Font Curve")
        self.font_curve.body = self.text
        self.obj = bpy.data.objects.new(name=self.text, object_data=self.font_curve)
        if assign_collection:
            self.assign_collection(self.obj)
        return self.obj

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, text):
        self._text = text
        if self.obj:
            self.obj.data.body = text

    def align_to_camera(self, frame = None): #The second argument is needed for the update function caller 
        if not self.obj: return
        camera = self.find_camera() #Method belonging to Bobject
        loc, rot, cale = camera.matrix_world.decompose()
        #sloc, srot, scale = self.obj.matrix_world.decompose()
        if self.obj.rotation_mode == "QUATERNION":
            self.obj.rotation_quaternion = rot.to_quaternion()
        else:
            self.obj.rotation_euler = rot.to_euler()
