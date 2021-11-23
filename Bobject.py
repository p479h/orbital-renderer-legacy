try:
    import mathutils;
except:
    print("Could not import mathutuils");
try:
    import bpy;
except:
    None;
from PointGroup import PointGroup
import numpy as np;
import os;
import json;
import re;


def load_atoms_data(): 
    with open("atoms.json", "r") as j_file:
        return json.load(j_file);


class Bobject: #Blender object 
    data = load_atoms_data();
    radii_list = data["radii"];
    colors = data["atoms"];
    def __init__(self):
        self.pause = 20; #These three are timers for the animations and can easily be integrated into the json file.
        self.transition = 59;#It would be concise to make this a class property. But I am not sure yet.
        self.short_pause = 1;
        self.frame_current = 0;

    #This is a helper class that offers basic blender functionality
    
    def play(self, *objs):
        self.keyframe_state(*objs)
        self.frame_current += self.transition
        self.set_frame(self.get_current_frame() + self.transition)

    def wait(self, *objs):
        self.keyframe_state(*objs)
        self.frame_current += self.pause
        self.set_frame(self.get_current_frame() + self.pause)

    def short_wait(self, *objs):
        self.keyframe_state(*objs)
        self.frame_current += self.short_pause
        self.set_frame(self.get_current_frame() + self.short_pause)

    
    @staticmethod
    def smooth_obj(obj, modifier = False):
        bpy.ops.object.select_all(action='DESELECT');
        obj.select_set(True);
        bpy.context.view_layer.objects.active = obj;
        bpy.ops.object.shade_smooth();
        if modifier:
            obj.modifiers.new("Smoother", "SUBSURF");
            obj.modifiers["Smoother"].levels = 2;

    @staticmethod
    def deselect_all():
        bpy.ops.object.select_all(action="DESELECT");
        bpy.context.view_layer.objects.active = None;

    def set_active(self, obj):
        self.select(obj);
        bpy.context.view_layer.objects.active = obj;

    @staticmethod
    def select(*objs):
        [obj.select_set(True) for obj in objs];
        
    def set_parent(self, parent, child):
        self.deselect_all();
        self.select(parent, child);
        self.set_active(parent);
        bpy.ops.object.parent_set(type="OBJECT", keep_transform = True);

    @staticmethod
    def set_frame(frame):
        bpy.context.scene.frame_current = frame;
        
    @staticmethod
    def get_current_frame():
        return bpy.context.scene.frame_current;

    def set_parents(self, parent, children):
        [self.set_parent(parent, child) for child in children];

    def set_origin(self, obj, type = "ORIGIN_GEOMETRY"):
        """
            Types: ORIGIN_GEOMETRY, GEOMETRY_ORIGIN, ORIGIN_CURSOR, ORIGIN_CENTER_OF_MASS, ORIGIN_CENTER_OF_VOLUME
            """
        self.deselect_all()
        self.set_active(obj)
        bpy.ops.object.origin_set(type = type);


    def keyframe_state(self, *objs, property = "all", frame = None):
        property = property.lower()
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


    def add_transition(self, property, obj, t0, transition, init, end):
        self.deselect_all();
        obj.select_set(True);
        self.set_active(obj);
        setattr(obj, property, init);
        obj.keyframe_insert(data_path = property, frame = t0);
        setattr(obj, property, end);
        obj.keyframe_insert(data_path = property, frame = t0+transition);

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
                collection.objects.unlink(obj);
            except:
                None; #Sometimes an object is not part of a collection, which would throw an error.

    @staticmethod                
    def link_obj(obj, collection):
        collection.objects.link(obj);


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
        bpy.data.scenes["Scene"].render.film_transparent = background_transparent;
        bpy.context.scene.render.resolution_x = resolution_xy[0];
        bpy.context.scene.render.resolution_y = resolution_xy[1];
        bpy.context.scene.render.tile_x = tile[0];
        bpy.context.scene.render.tile_y = tile[1];
        bpy.context.scene.render.resolution_percentage = resolution_percentage;
        bpy.data.scenes["Scene"].render.image_settings.file_format = file_format;
        bpy.data.scenes["Scene"].render.image_settings.color_mode = color_mode;
        bpy.data.cameras[0].lens = focal_length;
        bpy.data.objects.get("Camera").matrix_world = mathutils.Matrix(camera_matrix);
    
    @staticmethod
    def switch_render_engine(engine = "CYCLES"):
        """ CYCLES, BLENDER_EEVEE"""
        bpy.data.scenes["Scene"].render.engine = engine;

    def make_collection(self, parent = None, name = None):
        if not name:
            if hasattr(self, "name"):
                name = self.name
            else:
                name = "obj";
                self.name = name;
        if not parent: #If there is no parent we just make the collection
            if not bpy.data.collections.get(name):
                c = bpy.data.collections.new(name  = name);
                bpy.context.scene.collection.children.link(c)
            return c;
        else:
            if parent.children.get(name):#Else we make the collection under that parent
                return parent.children.get(name);
            c = bpy.data.collections.new(name = name);
            parent.children.link(c);
            return c

    @staticmethod
    def copy_material(obj):
        """
        obj: blender object/mesh with an active material
        return
            copy of material for animation"""
        return obj.active_material.copy()

    @staticmethod
    def make_outline_material():
        if bpy.data.materials.get("AtomBondOutline"):
            return bpy.data.materials.get("AtomBondOutline");
        m = bpy.data.materials.new(name="AtomBondOutline");
        m.use_nodes = True;
        emmNode = m.node_tree.nodes.new(type="ShaderNodeEmission") # creates Emission shader node.
        origNode = m.node_tree.nodes["Principled BSDF"];
        OutputNode = m.node_tree.nodes["Material Output"]
        m.node_tree.links.new(OutputNode.inputs[0], emmNode.outputs[0])
        m.node_tree.nodes.remove(origNode);
        emmNode.inputs[0].default_value = (0, 0, 0, 1);
        m.use_backface_culling = True;
        return m;


    def delete_obj(self, *objs, delete_collection = True):
        """
            deletes objs
            if delete_collection == True, the empty collections linked to the object that is about to be deleted are deleted as well"""
        for obj in objs:
            collections = obj.users_collection
            self.deselect_all();
            self.unhide(obj);
            self.set_active(obj);
            print(collections)
            bpy.ops.object.delete(use_global = True);
            print(collections)
            if delete_collection:
                for c in collections:
                    if len(c.objects) == 0:
                        bpy.data.collections.remove(c)
            
        

    @classmethod
    def translate_obj(cls, obj, v):
        """
            Translates the molecule by v"""
        obj.location = obj.location + mathutils.Vector(v)
        return obj

    @classmethod
    def scale_obj(cls, obj, v):
        """
            scales the molecule by v"""
        if type(v) == int or type(v) == float:
            v = [v]*3
        obj.scale = obj.scale * mathutils.Vector(v)
        return obj
        
    @classmethod
    def rotate_obj(cls, obj, angle, axis = [0, 0, 1]):
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
            
    
    def hide(self, obj, hide_render = False):
        obj.hide_set(True);
        obj.hide_render = hide_render;

    def unhide(self, obj, unhide_render = True):
        obj.hide_set(False);
        obj.hide_render = unhide_render;
