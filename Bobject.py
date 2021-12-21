from all_imports import *

class Bobject: #Blender object
    def __init__(self, obj = None, pause = 20, transition = 59, short_pause = 1, name = "bob", parent_collection = None, collection = None):
        print("Object is ", obj)
        self.obj = obj #Blender object that obj refers to
        self.parent = None #
        self.pause = pause #Pause duration between transitions in frames
        self.transition = transition #Transition duration in frames
        self.short_pause = short_pause
        self.name = name
        self.updaters = []
        self.transitions = []
        self.parent_collection = parent_collection
        self.collection = collection
        if obj:
            self.assign_collection(obj)

    def assign_collection(self, obj):
        self.unlink_obj(obj)
        self.link_obj(obj, collection = self.collection)
        if not self.parent_collection is None:
            self.set_collection_parent(self.parent_collection, self.collection)

    def add_updater(self, func):
        """
            Every function in updaters gets run every frame and takes frame as argument.
            """
        self.updaters.append(func)

    @property
    def parent(self):
        if self.obj:
            return self.obj.parent
        return None

    @parent.setter
    def parent(self, parent):
        if self.obj:
            self.obj.parent = parent

    @property
    def obj(self):
        if hasattr(self,"_obj"):
            return self._obj


    @obj.setter
    def obj(self, obj):
        """ Sets the object that Bobject wraps"""
        self._obj = obj

    def get_angles(self, obj = None):
        if obj is None:
            obj = self.obj
        rot = "rotation_"+("quaternion" if obj.rotation_mode == "QUATERNION" else "euler")
        return [*getattr(obj, rot)]


    #This is a helper class that offers basic blender functionality
    @staticmethod
    def hex2rgb(_hex,_tuple=False): #Ivos function.
        """
        @brief      Converts RGB code to numeric values

        @param      angle  the hex code
        @param      angle  whether to create a tuple

        @return     Numeric color values
        """
        string = _hex.lstrip('#') # if hex color starts with #, remove # char
        rgb255 = list(int(string[i:i+2], 16) for i in (0, 2 ,4))
        rgb_r = rgb255[0]/255.0
        rgb_g = rgb255[1]/255.0
        rgb_b = rgb255[2]/255.0
        if _tuple == False:
            return [rgb_r, rgb_g, rgb_b]
        elif _tuple == True:
            return (rgb_r, rgb_g, rgb_b, 1.0)

    def to_euler(self, obj = None):
        self.set_active(obj if obj else self.obj)
        if not self.obj is None:
            self.obj.rotation_mode = "XYZ"
        elif not obj is None:
            obj.rotation_mode = "XYZ"
        else:
            print("NO VALID OBJECT TO_EULER")

    def to_quaternion(self, obj = None):
        if not self.obj is None:
            self.obj.rotation_mode = "QUATERNION"
        elif not obj is None:
            obj.rotation_mode = "QUATERNION"
        else:
            print("NO VALID OBJECT TO_EULER")

    def play(self, *objs):
        self.keyframe_state(*objs)
        self.set_frame(self.get_current_frame() + self.transition)

    def wait(self, *objs):
        self.keyframe_state(*objs)
        self.set_frame(self.get_current_frame() + self.pause)

    def short_wait(self, *objs):
        self.keyframe_state(*objs)
        self.set_frame(self.get_current_frame() + self.short_pause)

    def set_scale_obj(self, obj, v):
        if type(v) in (int, float):
            v = [v]*3
        obj.scale = mathutils.Vector(v)
        return obj

    def set_location_obj(self, obj, v):
        if type(v) in (int, float):
            v = [v]*3
        obj.location = mathutils.Vector(v)
        return obj

    def set_position_obj(self, *args, **kwargs):
        return self.set_location_obj(*args, **kwargs)

    def set_rotation_obj(self, obj, angle, axis):
        rmode = obj.rotation_mode
        if rmode == "QUATERNION":
            obj.rotation_quaternion = mathutils.Quaternion(axis, angle)
        else:
            if type(angle) in (int, float):
                angle = [angle]*3
            obj.rotation_euler = mathutils.Euler(angle, rmode)
        return obj

    def set_position(self, v):
        return self.set_position_obj(self.obj, v)

    def set_location(self, v):
        return self.set_position(self,v)

    def set_scale(self, v):
        return self.set_scale_obj(self.obj, v)

    def set_rotation(self, angle, axis=[0, 0, 1]):
        return self.set_rotation_obj(self.obj, angle, axis)

    def scale(self, v):
        self.scale_obj(self.obj, v)
        return self

    def rotate(self, angle, axis):
        self.rotate_obj(self.obj, angle, axis)
        return self

    def translate(self, v):
        self.translate_obj(self.obj, v)
        return self


    def translate_obj(self, obj, v):
        """
            Translates the molecule by v"""
        obj.location = obj.location + mathutils.Vector(v)
        return obj


    def scale_obj(self, obj, v):
        """
            scales the molecule by v"""
        if type(v) == int or type(v) == float:
            v = [v]*3
        obj.scale = obj.scale * mathutils.Vector(v)
        return obj


    def rotate_obj(self, obj, angle, axis = [0, 0, 1]):
        """
            rotates the molecule by angle about axis
            if using quaternion, angle should be a float or int.
            if using Euler, angle should be a tuple or list."""
        rmode = obj.rotation_mode
        if rmode == "QUATERNION":
            obj.rotation_quaternion.rotate(mathutils.Quaternion(axis, angle))
        else:
            if type(angle) in (int, float):
                angle = [angle]*3
            obj.rotation_euler.rotate(mathutils.Euler(angle, rmode))
        return obj


    def add_modifier_obj(self, obj, modifier = "SUBSURF"):
        obj.modifiers.new(modifier.lower(), modifier)
        obj.modifiers[modifier.lower()].levels = 2
        return obj.modifiers[modifier.lower()]

    def add_modifier(self, modifier = "SUBSURF"):
        return self.add_modifier_obj(self.obj, modifier)

    def smooth_obj(self, obj, modifier = False):
        self.deselect_all()
        self.set_active(obj)
        bpy.ops.object.shade_smooth()
        if modifier:
            self.add_modifier("SUBSURF")

    def smooth(self, *args, **kwargs):
        self.smooth_obj(self.obj, *args, **kwargs)

    @staticmethod
    def select(*objs):
        [obj.select_set(True) for obj in objs]

    @staticmethod
    def deselect_all():
        bpy.ops.object.select_all(action="DESELECT")
        bpy.context.view_layer.objects.active = None

    @classmethod
    def set_active(cls, obj):
        if type(obj) == Bobject: #We can now use it for an object that is not a BlenderObject
            obj = obj.obj
        cls.select(obj)
        bpy.context.view_layer.objects.active = obj

    @staticmethod
    def get_active():
        return bpy.context.view_layer.objects.active

    @staticmethod
    def find_material(m):
        return bpy.data.materials.get(m)

    def get_material(self):
        return self.material

    @classmethod
    def set_parent(cls, parent, child):
        cls.deselect_all()
        cls.select(parent, child)
        cls.set_active(parent)
        bpy.ops.object.parent_set(type="OBJECT", keep_transform = True)

    @staticmethod
    def set_frame(frame):
        bpy.context.scene.frame_current = frame

    @staticmethod
    def get_current_frame():
        return bpy.context.scene.frame_current

    @classmethod
    def skip_frames(cls, n):
        self.set_frame(cls.get_current_frame()+n)

    @classmethod
    def wait(cls, n):
        cls.skip_frames(n)

    @classmethod
    def set_parents(cls, parent, children):
        [cls.set_parent(parent, child) for child in children]

    @classmethod
    def set_origin(cls, obj, type = "ORIGIN_GEOMETRY"):
        """
            Types: ORIGIN_GEOMETRY, GEOMETRY_ORIGIN, ORIGIN_CURSOR, ORIGIN_CENTER_OF_MASS, ORIGIN_CENTER_OF_VOLUME
            """
        cls.deselect_all()
        cls.set_active(obj)
        bpy.ops.object.origin_set(type = type)


    def keyframe_state(self, *objs, property = "all", frame = None):
        property = property.lower()
        if objs[0].__class__ != bpy.types.Object: #In case we are dealing with wrapped objects
            objs = [o.obj for o in objs]
        modes = ["rotation_euler" if "X" in o.rotation_mode else "rotation_quaternion" for o in objs]
        if property == "all":
            properties = [("location", "scale", modes[i]) for i, o in enumerate(objs)]
        elif property == "scale" or property == "location":
            properties = [(property,) for o in objs]
        elif property == "rotation":
            properties = modes
        else:
            print("Problem in Bobject.keyframe_state due to unknown property")

        if frame is None:
            frame = self.get_current_frame()

        for obj, props in zip(objs, properties):
            self.set_active(obj)
            for p in props:
                obj.keyframe_insert(data_path = p, frame = frame)

    @classmethod
    def add_transition(cls, property, obj, t0, transition, init, end):
        cls.deselect_all()
        obj.select_set(True)
        cls.set_active(obj)
        setattr(obj, property, init)
        obj.keyframe_insert(data_path = property, frame = t0)
        setattr(obj, property, end)
        obj.keyframe_insert(data_path = property, frame = t0+transition)

    def add_material_transition(self, property_name, obj, t0, transition, init, end, slot = "all"):
        property_index = {"base_color": 0, "transmission": 15, "emission": 17, "alpha": 19}[property_name]
        self.deselect_all()
        obj.select_set(True)
        self.set_active(obj)
        for m in obj.material_slots:
            if type(slot) == int:
                if m != obj.material_slots[slot]:
                    continue
            material = m.material
            imp = material.node_tree.nodes["Principled BSDF"].inputs[property_index]
            imp.default_value = init
            imp.keyframe_insert("default_value", frame = t0)
            imp.default_value = end
            imp.keyframe_insert("default_value", frame = t0 + transition)

    @staticmethod
    def unlink_obj(obj, collections = None):
        if collections is None:
            collections = bpy.data.collections

        for collection in collections:
            try:
                collection.objects.unlink(obj)
            except:
                None #Sometimes an object is not part of a collection, which would throw an error.

    @staticmethod
    def link_obj(obj, collection):
        collection.objects.link(obj)


    @staticmethod
    def set_camera(
        engine: str = "CYCLES",
        samples: int = 60,
        resolution_xy: (int, int) = (512, 512),
        tile: (int, int) = (256, 256),
        resolution_percentage: int = 100,
        file_format: str = "PNG",
        color_mode: str = "RGBA",
        background_transparent: bool = True,
        focal_length: float = 75,
        camera_matrix = [[ 0.707, -0.395,  0.586,  8.965],[ 0.707,  0.395, -0.586, -8.982],[ 0.   ,  0.829,  0.559,  8.75 ],[ 0.   ,  0.   ,  0.   ,  1.   ]]
        ) -> None:
        bpy.data.scenes["Scene"].render.film_transparent = background_transparent
        bpy.context.scene.render.resolution_x = resolution_xy[0]
        bpy.context.scene.render.resolution_y = resolution_xy[1]
        bpy.context.scene.render.tile_x = tile[0]
        bpy.context.scene.render.tile_y = tile[1]
        bpy.context.scene.render.resolution_percentage = resolution_percentage
        bpy.data.scenes["Scene"].render.image_settings.file_format = file_format
        bpy.data.scenes["Scene"].render.image_settings.color_mode = color_mode
        bpy.data.cameras[0].lens = focal_length
        bpy.data.objects.get("Camera").matrix_world = mathutils.Matrix(camera_matrix)

    @staticmethod
    def switch_render_engine(engine = "CYCLES"):
        """ CYCLES, BLENDER_EEVEE"""
        bpy.data.scenes["Scene"].render.engine = engine

    def make_collection(self, parent = None, name = None):
        if not name:
            if hasattr(self, "name"):
                name = self.name
            else:
                name = "obj"
                self.name = name
        if not parent: #If there is no parent we just make the collection
            if not bpy.data.collections.get(name):
                c = bpy.data.collections.new(name  = name)
                bpy.context.scene.collection.children.link(c)
                return c
            return bpy.data.collections.get(name)
        else:
            if parent.children.get(name):#Else we make the collection under that parent
                return parent.children.get(name)
            c = bpy.data.collections.new(name = name)
            self.set_collection_parent(parent, c)
            return c

    def set_collection_parent(self, parent, child):
        parent.children.link(child)

    @staticmethod
    def copy_material(obj):
        """
        obj: blender object/mesh with an active material
        return
            copy of material for animation"""
        return obj.active_material.copy()

    @staticmethod
    def make_outline_material():
        """ Makes black material with use_backface_culling for outline"""
        if bpy.data.materials.get("AtomBondOutline"):
            return bpy.data.materials.get("AtomBondOutline")
        m = bpy.data.materials.new(name="AtomBondOutline")
        m.use_nodes = True
        emmNode = m.node_tree.nodes.new(type="ShaderNodeEmission") # creates Emission shader node.
        origNode = m.node_tree.nodes["Principled BSDF"]
        OutputNode = m.node_tree.nodes["Material Output"]
        m.node_tree.links.new(OutputNode.inputs[0], emmNode.outputs[0])
        m.node_tree.nodes.remove(origNode)
        emmNode.inputs[0].default_value = (0, 0, 0, 1)
        m.use_backface_culling = True
        return m

    @classmethod
    def delete_obj(cls, *objs, delete_collection = True):
        """
            deletes objs
            if delete_collection == True, the empty collections linked to the object that is about to be deleted are deleted as well"""
        for obj in objs:
            collections = obj.users_collection
            cls.deselect_all()
            cls.unhide(obj)
            cls.set_active(obj)
            bpy.ops.object.delete(use_global = True)
            if delete_collection:
                for c in collections:
                    if len(c.objects) == 0:
                        bpy.data.collections.remove(c)

    def erase(self):
        self.delete_obj(self.obj, delete_collection = True)


    @classmethod
    def apply_transform(cls, ob, location=False, rotation=False, scale=False):
        Matrix = mathutils.Matrix
        mb = ob.matrix_basis
        I = Matrix()
        loc, rot, scale = mb.decompose()

        T = Matrix.Translation(loc)
        R = mb.to_3x3().normalized().to_4x4()
        S = Matrix.Diagonal(scale).to_4x4()

        transform = [I, I, I]
        basis = [T, R, S]

        def swap(i):
            transform[i], basis[i] = basis[i], transform[i]
        swaps = [location, rotation, scale]
        [swap(i) for i, b in enumerate(swaps) if b] #apply the swaps

        M = transform[0] @ transform[1] @ transform[2]
        if hasattr(ob.data, "transform"):
            ob.data.transform(M)
        for c in ob.children:
            c.matrix_local = M @ c.matrix_local

        ob.matrix_basis = basis[0] @ basis[1] @ basis[2]

    @classmethod
    def hide(cls, obj, hide_render = False):
        obj.hide_set(True)
        obj.hide_render = hide_render

    @classmethod
    def unhide(cls, obj, unhide_render = True):
        obj.hide_set(False)
        obj.hide_render = unhide_render

    def render_animation_group_theory(self, times, directs):
        """
        Used_for_symmetry operations
        Times are the ending times of each animation (and beggining of the first)
        directs is the list of directories for each time range!
        """
        zfill = len(str(bpy.data.scenes["Scene"].frame_end)) #Ensures correct ordering!!!
        bpy.data.scenes["Scene"].render.image_settings.file_format = "PNG"
        ranges = [list(range(times[i], times[i+1])) for i in range(len(times[:-1]))]
        for ir, r in enumerate(ranges):
           for frame in r:
               if len(self.updaters) > 0:
                   [u(frame) for u in self.updaters]
               self.setup_dir(directs[ir])
               self.render_image(directs[ir], frame = frame)


    def render_animation(self, frames = None, *args, **kwargs):
        """
        arguments are the same as render_image"""
        if frames is None:
            frames = [0, self.get_current_frame()]
        for frame in range(*frames):
            if len(self.updaters) > 0:
                [u(frame) for u in self.updaters]
            kwargs["frame"] = frame
            self.render_image(*args, **kwargs)


    def render_image(self, directory = None, name = None, file_format = 'PNG', frame = 0):
        if not name:
            name = str(frame).zfill(4)
        if not directory:
            directory = "."
        bpy.data.scenes["Scene"].render.image_settings.file_format = file_format
        bpy.context.scene.render.filepath = os.path.join(os.getcwd(), directory, name)
        self.set_frame(frame)
        bpy.ops.render.render(use_viewport = True, write_still=True)
