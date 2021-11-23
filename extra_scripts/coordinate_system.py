import numpy as np
import bpy
import mathutils

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
