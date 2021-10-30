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


class Arrow:
    def __init__(self, p1, p2, material = None, bodyW = 1/4, fracH = 1.5/5, ratioW = 1.5):
        self.arrow = self.make_arrow(p1, p2, material, bodyW, fracH, ratioW);
        
    @classmethod
    def make_arrow(self, p1, p2, material, bodyw, fracH, ratioW):
        #body w is the fraction of the length which the radius of the cylinder has
        p1, p2 = [np.array(p) for p in [p1, p2]];
        r = p2 - p1;
        l = np.linalg.norm(r); #Total length
        l_body = l*(1-fracH);
        l_head = l*fracH;
        r_body = l_body*bodyw;
        r_head = r_body*ratioW;
        
        axis = np.cross([0, 0, 1], r/l);
        angle = np.arccos(np.dot([0, 0, 1], r/l));
        quat = mathutils.Quaternion(axis.tolist(), angle);
        
        bpy.ops.mesh.primitive_cylinder_add(
            radius = r_body, location = p1+r/l*l_body/2,
            rotation = quat.to_euler(), depth = l_body);
        body = bpy.context.active_object;
        
        bpy.ops.mesh.primitive_cone_add(
            radius1 = r_head, depth = l_head, 
            rotation = quat.to_euler(), location = p1+r/l*(l_body+l_head/2),);
        head = bpy.context.active_object;
        body.select_set(True);
        
        bpy.ops.object.mode_set(mode='OBJECT');
        bpy.ops.object.join();
        bpy.ops.object.origin_set(type="ORIGIN_CURSOR");
        bpy.ops.object.shade_smooth();
        
        obj = bpy.context.active_object;
        obj.data.use_auto_smooth = True;
        
        if material:
            obj.active_material = material;
        return obj;
    
    @classmethod
    def make_arrow_material(self,axis = "x", color = None):
        if bpy.data.materials.get(axis+"axis"):
            return bpy.data.materials.get(axis+"axis");
        #else 
        bpy.data.materials.new(name = axis+"axis");
        mat = bpy.data.materials.get(axis+"axis");
        mat.use_nodes = True;
        mat.node_tree.nodes["Principled BSDF"].inputs[7].default_value = .3;
        addres = mat.node_tree.nodes["Principled BSDF"].inputs[0];
        if not color:
            if axis == "z":
                addres.default_value = (0.2, 0.3, 0.7, 1);
            elif axis == "x":
                addres.default_value = (1, 0.19, 0.14, 1);
            elif axis == "y":
                addres.default_value = (0.04, 0.5, 0.22, 1);
        return mat;
    
    @classmethod
    def add_text(self, text = "x", location = [1.2,0,0], material = None, font = "georgia"):
        font = bpy.data.fonts.load(rf"C:\Windows\Fonts\{font}.ttf");
        font_curve = bpy.data.curves.new(type="FONT", name="Font Curve")
        font_curve.body = text;
        font_obj = bpy.data.objects.new(name="Font Object", object_data=font_curve);
        font_obj.data.font = font;
        bpy.context.scene.collection.objects.link(font_obj);
        font_obj.location = mathutils.Vector(location);
        
        if material:
            font_obj.active_material = material;
        return font_obj;
    
    @classmethod
    def coordinate_system(self, p=[0, 0, 0], l=1, kind = None, **kwargs):
        if kind == "thic":
            kwargs["fracH"]=1.5/5; kwargs["ratioW"]=1.5;
            kwargs["bodyw"] = 1/4;
        elif kind == "usual":
            kwargs["fracH"]=1/5; kwargs["ratioW"]=2;
            kwargs["bodyw"] = .5/7;
        elif kind == "malnurished":
            kwargs["fracH"]=.7/5; kwargs["ratioW"]=3;
            kwargs["bodyw"] = .2/7;
        else: #You are feeling lucky!
            print("Ohohoho");
        
        print(kwargs)
        p = np.array(p);
        materials = [self.make_arrow_material(m) for m in "x y z".split()]
        texts = [self.add_text(axisname, p+loc, mat) for axisname, loc, mat in zip(["x","y","z"], np.eye(3)*1.2*l, materials)]
        arrowx = self.make_arrow(p, p+[l, 0, 0], materials[0], **kwargs);
        arrowy = self.make_arrow(p, p+[0, l, 0], materials[1], **kwargs);
        arrowz = self.make_arrow(p, p+[0, 0, l], materials[2], **kwargs);
        
        arrowx.select_set(True);
        arrowy.select_set(True);    
        bpy.ops.object.mode_set(mode='OBJECT');
        bpy.ops.object.join();
        bpy.ops.object.origin_set(type="ORIGIN_CURSOR");
        
        for t in texts:
            t.select_set(True);
        bpy.ops.object.parent_set(type="OBJECT");
        return bpy.context.active_object, texts;
    
    @staticmethod
    def hide_text(textobj):
        textobj.hide_set(True)
        textobj.hide_render = True;


def hex2rgb(_hex,_tuple=False): #Ivos function.
    """
    @brief      Converts RGB code to numeric values

    @param      angle  the hex code
    @param      angle  whether to create a tuple

    @return     Numeric color values
    """
    string = _hex.lstrip('#') # if hex color starts with #, remove # char
    rgb255 = list(int(string[i:i+2], 16) for i in (0, 2 ,4))
    rgb_r = rgb255[0]/255.0;
    rgb_g = rgb255[1]/255.0;
    rgb_b = rgb255[2]/255.0;
    if _tuple == False:
        return [rgb_r, rgb_g, rgb_b]
    elif _tuple == True:
        return (rgb_r, rgb_g, rgb_b, 1.0)

def load_atoms_data(): 
    with open("atoms.json", "r") as j_file:
        return json.load(j_file);


class Object:
    data = load_atoms_data();
    radii_list = data["radii"];
    colors = data["atoms"];
    def __init__(self):
        pass;
    #This is a helper class that offers basic blender functionality
    
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
        self.select(child, parent);
        self.set_active(parent);
        bpy.ops.object.parent_set(type="OBJECT");

    @staticmethod
    def set_frame(frame):
        bpy.data.scenes["Scene"].frame_current = frame;

    def set_parents(self, parent, children):
        [self.set_parent(parent, child) for child in children];

    def set_origin(self, obj, type = "ORIGIN_GEOMETRY"):
        """
            Types: ORIGIN_GEOMETRY, GEOMETRY_ORIGIN, ORIGIN_CURSOR, ORIGIN_CENTER_OF_MASS, ORIGIN_CENTER_OF_VOLUME
            """
        self.deselect_all()
        self.set_active(obj)
        bpy.ops.object.origin_set(type = type);

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
    def unlink_obj(obj):
        for collection in bpy.data.collections:
            try:
                bpy.context.scene.collection.objects.unlink(obj);
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
        camera_matrix = [[0.707, -0.395, 0.586, 10.822], [0.707, 0.395, -0.586, -10.038], [0.0, 0.829, 0.559, 9.932], [0.0, 0.0, 0.0, 1.0]]
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
        if not parent:
            if not bpy.data.collections.get(name):
                bpy.data.collections.new(name  = name)
                bpy.context.scene.collection.children.link(bpy.data.collections[name])
            return bpy.context.scene.collection.children.get(name);
        else:
            if parent.children.get(name):
                return parent.children.get(name);
            bpy.data.collections.new(name = name);
            parent.children.link(bpy.data.collections[name]);
        return bpy.data.collections[name];

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

    def delete_obj(self, *objs):
        for obj in objs:
            self.deselect_all();
            self.unhide(obj);
            self.set_active(obj);
            bpy.ops.object.delete(use_global = True);
        

    def scale_obj(self, obj, axes_factors, apply = False, orient_type = "LOCAL"):
        """
            Scales the obj along defined axes.
            Perfect for making bonds thinner along their xy planes.
            LOCAL, GLOBAL... 
            """
        self.deselect_all()
        self.set_active(obj);
        bpy.ops.transform.resize(value=axes_factors[:3], orient_type=orient_type)
        if apply:
            bpy.ops.object.transform_apply(scale = True);
        return obj;
    
    def hide(self, obj, hide_render = False):
        obj.hide_set(True);
        obj.hide_render = hide_render;

    def unhide(self, obj, unhide_render = True):
        obj.hide_set(False);
        obj.hide_render = unhide_render;
