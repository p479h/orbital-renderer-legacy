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

        

class Atom:
    # The color and radii will be stored by the class for compatibility with old projects!
    data = load_atoms_data();
    radii_list = data["radii"];
    colors = data["atoms"];
    def __init__(self, position = None, radius = None, material = None, name = None):
        """
            position: 1x3 matrix  with cartesian coordinates of the center of the atom
            radius: float with radius of the atom
            material: blender object with the material of the atom
            """
        self.name = self.set_name(name);
        self.position = self.set_position(position);
        self.radius = self.set_radius(radius);
        try:
            self.material = self.set_material(material);
        except:
            None #set_material only works from inside blender

    def set(self, argument, attribute): #I did this in case at some point we wish to add underscores or some other prefix to property names
        #This makes it possible to specify the positions and such at a later stage after the creation of the atom
        setattr(self, attribute, argument);

    def set_name(self, name):
        self.set(name, "name");
        return self.name;

    def set_position(self, position):
        if type(position) == type(None):#The reason this is allowed is because Molecule can make use of the math here described
            return None;
        self.set(position, "position");
        self.position = np.array(list(position));
        return self.position;

    def set_radius(self, radius):
        self.set(radius, "radius");
        if not self.radius:
            if self.name:
                self.radius = self.radii_list[self.name];
        return self.radius;

    def set_material(self, material):
        self.set(material, "material");
        if not self.material:
            if self.name:
                self.material = self.make_material(self.name);
        return self.material;

    def makeAtomMesh(self):
        if not self.material or not self.radius or type(self.position) == type(None):
            return False;
        bpy.ops.mesh.primitive_ico_sphere_add(radius=self.radius, location=self.position);
        self.atomMesh = bpy.context.active_object;
        self.atomMesh.active_material = self.material;
        self.smooth_obj(self.atomMesh, modifier = True);
        return self.atomMesh;

    def render(self): #This function is overwritten for the Molecule class
        return self.makeAtomMesh();

    @staticmethod
    def smooth_obj(obj, modifier = False):
        bpy.ops.object.select_all(action='DESELECT');
        obj.select_set(True);
        bpy.context.view_layer.objects.active = obj;
        bpy.ops.object.shade_smooth();
        if modifier:
            obj.modifiers.new("Smoother", "SUBSURF");
            obj.modifiers["Smoother"].levels = 2;

    def deselect_all(self):
        bpy.ops.object.select_all(action="DESELECT");
        bpy.context.view_layer.objects.active = None;

    def set_active(self, obj):
        bpy.context.view_layer.objects.active = obj;

    def set_parent(self, parent, child):
        self.deselect_all();
        self.select([child, parent]);
        self.set_active(parent);
        bpy.ops.object.parent_set(type="OBJECT");

    def set_parents(self, parent, children):
        [self.set_parent(parent, child) for child in children];

    def print_properties(self):
        for p in self.__dict__:
            print(p);
            print(getattr(self, p));
            print("\n");


    def make_material(cls, A):
        print(A)
        print(cls.colors[A])
        print(hex2rgb(cls.colors[A], _tuple=True))
        """ A is the atom name. Each one has a different color"""
        if not bpy.data.materials.get(A):
            #Note that these colors were not generated by hand. That would have been suicide;
            m = bpy.data.materials.new(name=A);
            m.use_nodes = True;
            color = hex2rgb(cls.colors[A], _tuple=True)
            print(color, cls.data["atoms"][A])
            m.node_tree.nodes["Principled BSDF"].inputs[0].default_value = color;
            m.node_tree.nodes["Principled BSDF"].inputs[7].default_value = .1;
        else:
            print("Material in data.materials", A)
            m = bpy.data.materials.get(A);
        return m;

    def make_orbital_material(self, sign, copy = True, ivo = False):
        print("This function is getting called")
        A = {"p":"positive", "n": "negative"}[sign]
        try:
            bpy.data.materials.get(A)#So we can use this class without being in 
        except:
            None
        if not bpy.data.materials.get(A) or not copy:
            m = bpy.data.materials.new(name=A);
            m.use_nodes = True;
            if A == "positive":
                rgb = (.16, .47, .8, 1)
                if ivo:
                    rgb = (0.05, 0.24, 0.91, 1)
            else:
                rgb = (.8, .33, .11, 1)
                if ivo:
                    (0.1, 0.419, 0.07, 1)
            
            m.node_tree.nodes["Principled BSDF"].inputs[0].default_value = rgb;
            if bpy.data.scenes["Scene"].render.engine == "BLENDER_EEVEE":
                m.node_tree.nodes["Principled BSDF"].inputs[5].default_value = .3;
            m.node_tree.nodes["Principled BSDF"].inputs[7].default_value = .1;
            m.node_tree.nodes["Principled BSDF"].inputs[14].default_value = 1;
            m.node_tree.nodes["Principled BSDF"].inputs[15].default_value = .5;
            m.node_tree.nodes["Principled BSDF"].inputs[16].default_value = .5;
            m.use_screen_refraction= True
            m.blend_method = "BLEND"
            m.show_transparent_back = False
        else:
            print("Material in data.materials", A)
            m = bpy.data.materials.get(A);
        return m;
        
class Molecule(Atom):
    """
        This class unites many atoms, allowing them to bond and be rendered together
        Here each atom becomes associated to an index in the arrays with info about the molecule"""

    def __init__(self, atoms, radius_factor = 1,connections = None, name = None,
                 prettify_radii = True, collection = None, orders = []):
        super().__init__();
        self.prettify_radii = prettify_radii; #If turned off, the actuall atomic radii will be used
        self.position = self.get_positions(atoms); #Note how we use the same property name for Atoms and Molecule. This is because then we can rely on the "setters" already defined in Atoms.
        self.radius = self.get_radii(atoms)*radius_factor;
        self.material = self.get_materials(atoms);
        self.names = self.set_names(atoms); #names of all atoms in the same order as positions.
        self.name = self.set_molecule_name(name); #Name of the molecule
        self.connections = connections; # Bond indices
        self.meshes = []; #All of this molecules meshes
        self.atomMeshes = [];
        self.cylinderMeshes = [];
        self.cylinderRadii = [];
        self.bondOrder = orders; #The order of the bonds!
        self.mules = []; #Used for the animations
        self.current_frame = 0;
        self.pause = 20; #These three are timers for the animations and can easily be integrated into the json file.
        self.transition = 59;#It would be concise to make this a class property. But I am not sure yet.
        self.short_pause = 1;
        self.collection = self.make_collection(collection);
        self.orbital_indices = np.array([0 for i in range(len(self.position))]).astype(bool);
        self.orbital_meshes = [None for i in range(len(self.position))]; #Meshes of molecular orbitals
        self.orbital_matrices = np.tile(np.eye(4), (len(self.position), 1, 1)); #Matrices of molecular orbitals
        self.orbital_names = [None for i in range(len(self.position))];
        self.orbital_kinds = [None for i in range(len(self.position))];
        try:
            self.mule = self.make_empty()
        except:
            None#Only works from inside blender, even though this class can be used outside

    def unlink_obj(self, obj):
        for collection in bpy.data.collections:
            try:
                bpy.context.scene.collection.objects.unlink(obj);
            except:
                None; #Sometimes an object is not part of a collection, which would throw an error.
                    
    def link_obj(self, obj, collection):
        collection.objects.link(obj);


    def find_normal_from_points(self, center, corners):
        """
            Center 1x3 array with the coordinates of the center of the figure
            corners nx3 array with the coordinates of the edges of the figure
            returns 1x3 array with "average" normal.
            Note this can only be used if there is more than 1 corner!!!!
            """
        r = (center - corners)/np.linalg.norm(center-corners, keepdims = True, axis = 1);
        normals = np.array([np.cross(r[0], r[i]) for i in range(1, len(r))]);
        dots = np.array([np.dot(normals[0], n) for n in normals]);
        normals[dots<0]*=-1; #Invert antiparallel.
        avg_norm = np.sum(normals, axis = 0)/np.sqrt(len(normals))
        return avg_norm
        

    def make_double_bond(self, bond_index, atom_pair_indices, neighbour_indices = []):
        """
            bond_index: int with the index of the single bond which is to be doubled
            atom_pair_indices: list (0,2) int indices of atoms which form the bond
            neighbour_indices: list (0,N) int indices of 1st gen neighbouts to atom_pair_indices
            """
        c = self.cylinderMeshes[bond_index];
        cmat = np.array(c.matrix_world);
        cmat[:3, :3] = cmat[:3, :3]@(np.eye(3)*[[.4],[.4],[1]]);
        rad = self.cylinderRadii[bond_index];
        d = rad/1.9
        if len(neighbour_indices) < 2:
            r = np.array(cmat[:3, 0]);
            r = r/np.linalg.norm(r);
            
        else:
            r_center = np.mean(self.position[atom_pair_indices, :], axis = 0)
            r_neighbours = self.position[neighbour_indices, :]
            normal = self.find_normal_from_points(r_center, r_neighbours)
            r = np.array(cmat[:3, 2]);
            r = r/np.linalg.norm(r);
            r = np.cross(normal, r); #Now we have a vector that points perpendicular to the bonding axis!

        c.matrix_world = mathutils.Matrix.Translation(r*d)@mathutils.Matrix(cmat);
        c_copy = c.copy();
        c_copy.matrix_world = mathutils.Matrix.Translation(-r*d)@mathutils.Matrix(cmat);
        self.link_obj(c_copy, self.collection)
        self.deselect_all()
        self.select([c, c_copy])
        self.set_active(c)
        bpy.ops.object.join();#The double bond now occupies the same slot as the former single bond.

            
    def find_neighbours(self, pair):
        connections = np.array(self.connections)
        contains_first = {*connections[np.any(pair[0] == connections, axis = 1), :].flatten()}
        contains_second = {*connections[np.any(pair[1] == connections, axis = 1), :].flatten()}
        return np.array(list({*contains_first, *contains_second}-{*pair}))

        
    def make_collection(self, name):
        if not name:
            name = self.name
        if not bpy.data.collections.get(name):
            bpy.data.collections.new(name  = name)
            bpy.context.scene.collection.children.link(bpy.data.collections[name])
        return bpy.context.scene.collection.children.get(name);
        

    def copy_material(self, obj):
        """
        obj: blender object/mesh with an active material
        return
            copy of material for animation"""
        return obj.active_material.copy()

    def find_static_atoms(self, matrix, tol = 2e-1): # Lower tolerances require better xyz files... But this is good enough for most purpuses
        new_positions = (matrix@self.position.T).T
        r = np.linalg.norm(self.position - new_positions, axis = 1, keepdims = False)
        return r<tol #Returns logical array indexing atoms that do not get out of place during symmtry operation

    def find_static_orbitals(self, matrix, tol = 2e-1):
        static_atoms = self.find_static_atoms(matrix, tol).astype(bool);
        return static_atoms&(self.orbital_indices!=-1)&(np.array(self.orbital_meshes)!=None) #Returns logical array indexing orbitals that do not get out of place during symmtry operation

    def find_final_projections(self, mat, orb = "pz", tol = 2e-1):
        """
            Finds the coefficients that multiply the size of the molecular orbitals based on their orientation after the tranformation relative to before
            This funcition is more important for pxyz than s orbitals
            For example:
                if a transformation rotates a px orbital about the y axis 90 degrees, we get 0 in the end, since there is no overlap with itself anymore
                if a transformation rotates a pz orbital about the z axis, the result will always be 1
                if a transformation rotates a py orbital about the x axis 60 degress, you get an overlap of 0.5. Here I neglect the sign because this will be used for scaling the orbital, which will be rotated already by the animation.
            """
        remaining = self.find_static_atoms(mat, tol)
        if orb in ("2s", "s"):
            return remaining.astype(int)
        if mat.shape[-1] == 3:
            eye = np.eye(4);
            eye[:3, :3] = mat[:,:];
            mat = eye;
        final_orientations = np.einsum('ij,kjl->kil',mat,self.orbital_matrices);
        projection = (self.orbital_matrices*final_orientations).sum(axis = 1)[:,{"px":0, "x":0,"py":1, "y":1,"pz":2, "z":2}[orb]]
        return remaining*projection #We multiply by remaining because the orbitals that leave their atoms must be set to 0
        
    def add_SALC(self, orbitals, SALC, norm = True, orientation_function = None, *args, **kwargs):
        """
            Receives a list of coefficients per atomic index
            non-zero coefficients will be used to construct atomic orbitals. The coefficients will act scaling the orbitals
            norm is a boolean used to make the size of the orbitals more visually appealing
            orientation_function is a function that takes in coordinates ([xyz]) and uses those to construct a rotation_matrix to apply to the unit vectors of the world matrix of the orbital in blender! You would use this for xyz orbitals of an Oh symmetric molecule for example
            """
        orbital_meshes = []
        if norm:
            SALC = np.array(SALC)
            SALC = SALC / np.abs(SALC).max()
            SALC[SALC !=0 ] = SALC[SALC !=0 ] / np.abs(SALC[SALC !=0 ])**.3 # Dividing by the actual maxima can lead to very disproportional orbitals
        for i, coeff in enumerate(SALC):
            if np.abs(coeff) > 1e-4: # Due to numerical errors we may end up with 1e-4 to 1e-6 where there should be 0 in the salcs
                obj = self.add_atomic_orbital(orbitals[i] if type(orbitals) == list else orbitals, atom_index = i, scale = coeff, *args, **kwargs)
                self.set_parent(self.mule, obj)
                if orientation_function:
                    obj.matrix_world[:3, :3] = mathutils.Matrix(orientation_function(self.position[i]))@obj.matrix_world[:3, :3]
                self.orbital_matrices[i, :, :] = np.array(obj.matrix_world)
                orbital_meshes.append(obj)
        return orbital_meshes

    def add_atomic_orbital(self, name = "2s", directory = "ao_meshes", offset = [0, 0, 0], atom_index = None, scale = 1, kind = "cloud", material_copy = False, small = True):
        """
        Adds atomic orbital to molecule at specified position.
        data comes from .npy file in specified directory and one may choose to provide the location of the orbital OR the index where it resides.
        kind means arrow or cloud
        copy = True will use the same material for all orbitals (Harder to animate)
        copy = False will use a separate material for each orbital (colors can be animated)
        """
        meshes = []
        if type(atom_index) == int:
            offset = self.position[atom_index]
            self.orbital_indices[atom_index] = True;
            self.orbital_kinds[atom_index] = kind;
        for sign in ["p", "n"]:
            if not small:
                vertsfile = f"{name}_v_{sign}.npy"
                facesfile = f"{name}_f_{sign}.npy"
            else:
                try:
                    vertsfile = f"c_{name}_v_{sign}.npy"
                    facesfile = f"c_{name}_f_{sign}.npy"
                except:
                    vertsfile = f"{name}_v_{sign}.npy"
                    facesfile = f"{name}_f_{sign}.npy"
            if kind == "cloud":
                verts, faces = self.read_verts_and_faces(vertsfile, facesfile, directory)
                meshes.append(self.add_mesh(verts, faces, sign, offset, scale))
            elif kind == "arrow":
                l = scale*.75
                p0, p1 = np.array(offset) + [[0, 0, -l],[0, 0, l*1.2]]
                wid = self.radius.mean()/8
                meshes.append(Arrow(p0, p1, None, wid, .15, 1.8).arrow)
                if name == "px":
                    bpy.ops.transform.rotate(value = -np.pi/2, orient_axis = "Y")
                if name == "py":
                    bpy.ops.transform.rotate(value = np.pi/2, orient_axis = "X")
            else:
                raise("Kind of orbital is not recognized")
            if sign:
                meshes[-1].active_material = self.make_orbital_material(sign, copy = material_copy);
            if kind == "arrow":
                break
        [o.select_set(True) for o in meshes]
        if len(meshes)>1:
            self.set_active(meshes[0])
            bpy.ops.object.join()
        self.orbital_meshes[atom_index] = bpy.context.view_layer.objects.active
        self.meshes.append(self.orbital_meshes[atom_index])
        self.orbital_names[atom_index] = name
        #self.orbital_matrices[-1][3, :3] = offset[:]
        return bpy.context.active_object

    def read_verts_and_faces(self, vertsfile, facesfile, directory = "ao_meshes"):
        if directory:
            vertsfile = os.path.join(directory, vertsfile)
            facesfile = os.path.join(directory, facesfile)
        return np.load(vertsfile), np.load(facesfile)
        

    def add_mesh(self, verts, faces, sign = "p", offset = [0, 0, 0], scale = 1): #p is "positive"
        """
            makes a mesh from verts and faces"""
        bpy.ops.object.select_all(action='DESELECT')
        new_mesh = bpy.data.meshes.new("MO"+str(len(self.orbital_meshes)));
        new_mesh.from_pydata((verts*scale+offset).tolist(), [], faces.tolist());
        new_mesh.update()
        new_object = bpy.data.objects.new('Orbital', new_mesh)
        self.collection.objects.link(new_object)
        new_object.select_set(True)
        bpy.context.view_layer.objects.active = new_object
        bpy.ops.object.shade_smooth()
        return new_object;

    def erase_MO(self, MO): #MO is the mesh of the orbital
        index = self.orbital_meshes.index(MO)
        self.orbital_meshes[index] = None
        self.orbital_names[index] = None
        self.orbital_matrices[index, :, :] = np.eye(4)
        self.orbital_indices[index] = False
        index = self.meshes.index(MO)
        self.meshes.pop(index)
        bpy.data.objects.remove(MO, do_unlink=True)

    def erase_MOS(self):
        for MO in [o for o in self.orbital_meshes if o]: #Without this list comprehension the loop breaks prematurely!
            self.erase_MO(MO)

    def erase(self):
        for obj in self.meshes+self.mules:
            self.delete_obj(obj);
        self.meshes = []
        self.mules = []

    def delete_obj(self, obj):
        self.deselect_all();
        self.unhide(obj);
        obj.select_set(True);
        self.set_active(obj);
        bpy.ops.object.delete(use_global = True);

    def set_names(self, atoms):
        self.names = [a.name for a in atoms];
        return self.names;
    
    def set_molecule_name(self, name):
        if name:
            self.name = name;
            return name;
        self.name = "".join(self.names);
        return self.name;

    def select(self, objs):
        [obj.select_set(True) for obj in objs];

    def get_positions(self, atoms):
        return np.array([a.position for a in atoms]);

    def get_radii(self, atoms):
        return np.array([a.radius for a in atoms]);

    def get_materials(self, atoms):
        return [a.material for a in atoms];

    @staticmethod
    def load_molecule_data(filename, directory = "molecule_data"):
        if directory:
            path = os.path.join(os.getcwd(), directory, filename)
        else:
            path = os.path.join(os.getcwd(), filename)
        with open(path, "r") as f:
            data = json.load(f);
        if type(data["xyz"]) == list:
            data["xyz"] = os.path.join(*data["xyz"]);
        return data;

    @staticmethod
    def dump_molecule_data(data, filename, directory = "molecule_data"):
        path = os.path.join(directory, filename) if directory else filename;
        if type(data["xyz"]) == str:
            xyzpath = os.path.normpath(data["xyz"])
            data["xyz"] = xyzpath.split(os.sep) #Now we have a list that can be reassembled in accordance with the operating system's path separator
        with open(path, "w") as f:
            json.dump(data, f, ensure_ascii = False);
        


    #Now onto the rendering part
    def getBondRadius(self, rad):
        r = rad.mean()/2.3;
        if any(r > rad):
            r = min(rad);
        return r;

    @classmethod
    def read_mol(cls, path, first_index = 0):
        atoms = []
        coordinates = []
        bonds = []
        orders = []
        with open(path, "r") as f:
            i = 0;
            while i<100:
                if f.readline()[1:4].isalpha():
                    f.readline();
                    break
                i+=1;
            line1 = f.readline()
            n_atoms = int(line1[:3])
            n_bonds = int(line1[3:6])
            lines = f.readlines()
            for i,l in enumerate(lines):
                if i < n_atoms:
                    l = l.split()
                    atoms.append(l[3])
                    coordinates.append([float(num) for num in l[:3]])
                else:
                    if i > n_bonds + n_atoms - 1:
                        break
                    bonds.append([int(l[:3])-1, int(l[3:6])-1])
                    orders.append(int(l[6:9]))
        return atoms, coordinates, bonds, orders
            
    @classmethod
    def from_mol_file(cls, path, first_index = 0, *args, **kwargs):
        a, c, b, o = cls.read_mol(path, first_index)
        atoms = [Atom(position = coord, name = name) for coord, name in zip(c, a)];
        return cls(atoms, connections = b, orders = o, *args, **kwargs);

    @classmethod
    def from_file(cls, path, *args, **kwargs):
        if path.endswith(".xyz"):
            return cls.from_xyz_file(path, *args, **kwargs)
        elif path.endswith(".mol"):
            return cls.from_mol_file(path, *args, **kwargs)
        elif path.endswith(".smol"):
            return cls.from_smol_file(path, *args, **kwargs)
        else:
            print("File type not suported")
            print("Supported file types are: ")
            print("xyz\nmol\nsmol")
            return None

    @classmethod
    def read_smol(cls, path):
        atoms = []
        positions = []
        bonds = []
        with open(path, "r") as f:
            found_bonds = False
            for l in f.readlines():
                if len(l.split()) == 4:
                    found_bonds = True
                    l = l.split()
                    atoms.append(l[3])
                    positions.append([float(i) for i in l[:3]])
                elif found_bonds:
                    bonds.append([int(l[:3])-1, int(l[3:6])-1])
        return atoms, positions, bonds;

    @classmethod
    def from_smol_file(cls, path, *args, **kwargs):
        names, positions, bonds = cls.read_smol(path);
        atoms = [Atom(position = coord, name = name) for coord, name in zip(positions, names)];
        return cls(atoms, connections = bonds, *args, **kwargs);

    @classmethod
    def read_xyz(cls, path):
        positions = [];
        names = []; #The atomic labels
        with open(path, "r") as file:
            lines = file.read().replace("\\t", " ").split("\n");
            for i, l in enumerate(lines):
                pieces = l.split();
                if len(pieces) == 4:
                    positions.append([float(coord) for coord in pieces[1:]]);
                    names.append(pieces[0]);
        return names, positions

    @classmethod
    def from_xyz_file(cls, path,*args, **kwargs):
        if not path.endswith(".xyz"):
            path = path + ".xyz";
        names, positions = cls.read_xyz(path);
        atoms = [Atom(position = coord, name = name) for coord, name in zip(positions, names)];
        return cls(atoms, connections = cls.detect_bonds_xyz(atoms), *args, **kwargs);

    def make_atom_meshes(self, stick, smooth = True, prettify_radii = True):
        self.atomMeshes = [];
        rad = self.radius.copy();
        if prettify_radii:
            rad = self.get_smaller_radii(self.radius)
        for p, r, m in zip(self.position, rad, self.material):
            bpy.ops.mesh.primitive_ico_sphere_add(radius=(r if not stick else r**.3/5), location=p);
            obj = bpy.context.active_object;
            obj.active_material = m;
            if smooth:
                self.smooth_obj(obj, modifier = True);
            self.atomMeshes.append(obj);
            self.meshes.append(obj);
            self.unlink_obj(obj);
            self.link_obj(obj, self.collection);
        return self.meshes
            
    def get_smaller_radii(self, radius, s = 0.70):
        rad=np.array(radius).copy();
        rad[rad>self.radii_list["Fe"]*s] = self.radii_list["Fe"]*s;
        rad[np.array(self.names) == "H"] = 0.25
        return rad;

    def make_cylinder_meshes(self, stick, prettify_radii = True):
        if type(self.connections) == type(None):
            return;

        #self.cylinderMeshes = [];
        #self.cylinderRadii = [];
        if prettify_radii:
            rad = self.get_smaller_radii(self.radius);
        else:
            rad = self.radius
        for connection in self.connections:
            i0, i1 = connection;
            p0, p1 = self.position[i0, :], self.position[i1, :];
            m0, m1 = self.material[i0], self.material[i1];
            r0, r1 = rad[i0], rad[i1];
            r = self.getBondRadius(rad[np.array([i0, i1])]);
            if stick:
                r =  r**.5/10 + .02
            p1p0 = (p1-p0)/np.linalg.norm(p1-p0);
            middle = (p1+p1p0*(r0-r1) + p0)/2;
            if self.names[i1] != self.names[i0]: #If they are different atoms we make half the bond of each color. Else we make a single long cyllinder
                c0 = self.cylinder_between(*middle, *p0, r);
                self.smooth_obj(c0, modifier = False);
                c0.data.use_auto_smooth = True;
                c1 = self.cylinder_between(*middle, *p1, r);
                self.smooth_obj(c1, modifier = False);
                c1.data.use_auto_smooth = True;
                c0.active_material = m0;
                c1.active_material = m1;
                self.select([c0]);
                bpy.ops.object.join();
            else:#We make a single long bond
                c0 = self.cylinder_between(*p1, *p0, r);
                self.smooth_obj(c0, modifier = False);
                c0.data.use_auto_smooth = True;
                c0.active_material = m0;
            self.cylinderRadii.append(r);
            obj = bpy.context.active_object;
            self.cylinderMeshes.append(obj);
            self.meshes.append(obj);
            self.unlink_obj(obj)
            self.link_obj(obj, self.collection)
        

    def render(self, mirror_options={}, rotation_axis_options={}, stick = False, mirror = True, rotation_axis = True, prettify_radii = True, show_double_bonds = True):
        """
            Creates all the meshes.
            mirror_options and rotaiton_axis_options are the arguments that can be suppplied
            to the functions (respectively):
                bpy.ops.mesh.primitive_circle_add
                bpy.ops.mesh.primitive_cylinder_add
            supplied as a dictionary.
            stick means stick and ball model. It is used to give emphasis to the orbitals instead of atoms themselves
        """
        self.make_atom_meshes(stick, prettify_radii = prettify_radii);
        self.make_cylinder_meshes(stick, prettify_radii = prettify_radii);

        if show_double_bonds:
            for i, o in enumerate(self.bondOrder):
                if o == 1:
                    continue
                self.make_double_bond(i, self.connections[i], self.find_neighbours(self.connections[i]))

        #These are used for animations
        if not hasattr(self, "mule"):
            self.make_empty(True); # This objects parents the entire molecule. This way we can have unjoined meshes that all move together.
        else:
            self.deselect_all();
            self.select(self.meshes+[self.mule]);
            self.set_active(self.mule);
            bpy.ops.object.parent_set(type="OBJECT");
            
        #Note that the mirror and rotation axis are not parented as they would suffer deformations during the animations.
        if mirror:
            self.make_mirror(mirror_options, prettify_radii = prettify_radii);
        if rotation_axis:
            self.make_rotation_axis(rotation_axis_options, prettify_radii = prettify_radii);

    def render_mpl(self, ax):
        for b in self.connections:
            ax.plot(*(self.position[np.array(b)]).T, c = "black");
            
        for a in set(self.names):
            indices = [i for i, n in enumerate(self.names) if n == a];
            points = np.array([self.position[i] for i in indices])
            color = self.colors[a];
            color = hex2rgb(color);
            radius = self.radii_list[a];
            ax.scatter(*points.T, c = color, s = radius*80);

    def make_empty(self):
        bpy.ops.object.empty_add();
        e = bpy.context.active_object;
        if not hasattr(self, "empty"): #The first created mule will parent all atoms and bonds
            self.empty = self.mule = e;
            self.deselect_all();
            self.select(self.meshes+[e]);
            self.set_active(e);
            bpy.ops.object.parent_set(type="OBJECT");
        self.mules.append(e);
        e.rotation_mode = "QUATERNION";
        return e;


    #Adding the cylinders
    def cylinder_between(self, x1, y1, z1, x2, y2, z2, r): #this can be made shorter with numpy. But it is more readable this way.
      dx = x2 - x1;
      dy = y2 - y1;
      dz = z2 - z1 ;
      dist = np.sqrt(dx**2 + dy**2 + dz**2);
      bpy.ops.mesh.primitive_cylinder_add(
          radius = r,
          depth = dist,
          location = (dx/2 + x1, dy/2 + y1, dz/2 + z1)
      );
      phi = np.arctan2(dy, dx);
      theta = np.arccos(dz/dist);
      bpy.context.object.rotation_euler[1] = theta;
      bpy.context.object.rotation_euler[2] = phi;
      obj = bpy.context.active_object;
      return obj;

    def centralize(self): # Used for experimenting with molecules. Centers the center of mass
        self.position = self.position - np.array(self.position).mean(axis = 0)
        return self.position;

    def rotate_about(self, axis, theta): # Used for experimenting with molecules.
        axis, points = np.array(axis), self.position;
        M = np.array(mathutils.Matrix.Rotation(theta, 3, axis));
        return (M@points if len(points.shape) < 2 else (M@points.T).T);

    def center_facet(self, face_indices, reference = "z"): #Was used to help orient molecules.
        coordinates = self.position;
        p1, p2, p3 = coordinates[np.array(face_indices[:3])];
        v1, v2 = (p2 - p1), (p3 - p1);
        normal = np.cross(v2, v1)/np.linalg.norm(np.cross(v1, v2));
        axis = np.eye(3)[{"x":0, "y":1, "z":2}[reference]];
        angle = np.arccos(np.dot(axis, normal));
        self.position = self.rotate_about(np.cross(normal, axis), angle);
        return self.position;

    @classmethod #Crude method of determing bonds based on their covalent radii. Works well on mol files but not xyz files.
    def detect_bonds(cls, coordinates, radii, ratio = 1.2, constant = None):
        coordinates, radii = np.array(coordinates), np.array(radii);
        pairs = [];
        for i, (c1, r1) in enumerate(zip(coordinates[:-1], radii[:-1])):
            for j, (c2, r2) in enumerate(zip(coordinates[i+1:], radii[i+1:])):
                min_dist = (r1+r2)*ratio if not constant else constant;
                if np.linalg.norm(c2-c1) < min_dist:
                    pairs.append([i, j+i+1]);
        return pairs;

    @classmethod # I got this from ivo. Not sure where these exact numbers come from, but it works well on xyz files.
    def detect_bonds_xyz(cls, atoms):
        #First we set the limits for each kind of bond
        cls.distances = np.full((86, 86), 3.0);
        for i in range(0, 86):
            cls.distances[i][1] = 1.4
            cls.distances[1][i] = 1.4

            if i > 20:
                for j in range(2,20):
                    cls.distances[i][j] = 2.8
                    cls.distances[j][i] = 2.8
            else:
                for j in range(2,20):
                    cls.distances[i][j] = 2.0
                    cls.distances[j][i] = 2.0

        bonds = [];
        for i, a1 in enumerate(atoms[:-1]):
            for j, a2 in enumerate(atoms[i+1:]):
                a1id = cls.data["atomid"][a1.name]
                a2id = cls.data["atomid"][a2.name]
                a1id, a2id = sorted([a1id, a2id]);
                d = np.linalg.norm(a1.position - a2.position);
                if d < cls.distances[a1id, a2id]:
                    bonds.append([i, j+i+1]);
        return bonds;

    def find_closest_indices(self, point, number_elements_per_face = 3): #Can be used to find indices of a specific face of a plane in a molecule. Can later be used to help orient double bonds!
        coordinates = self.position;
        return np.argpartition(np.linalg.norm(coordinates-point,axis=1), number_elements_per_face)[:number_elements_per_face];

    def make_mirror(self, mirror_options, prettify_radii = True): #Makes the mirror that shows up in the animations. Also makes the material if it is not already defined in the scene.
        if "fill_type" not in mirror_options:
            mirror_options["fill_type"] = "NGON";
        if "radius" not in  mirror_options:
            distances = np.linalg.norm(self.position, axis = 1)
            furthest_index = np.argmax(distances)
            if prettify_radii:
                radius = self.get_smaller_radii(self.radius)[furthest_index, np.newaxis]
            else:
                radius = self.radius[furthest_index, np.newaxis]
            mirror_options["radius"] = distances[furthest_index] + 1.5*radius;
        if "vertices" not in mirror_options:
            mirror_options["vertices"] = 32;
        if "SUBSURF" not in mirror_options:
            apply_subsurf = True
        else:
            apply_subsurf = mirror_options["SUBSURF"];
            mirror_options.pop("SUBSURF")
        if not hasattr(self, "mirror"):
            bpy.ops.mesh.primitive_circle_add(**(mirror_options));
            self.mirror = bpy.context.active_object;
            self.mirror.rotation_mode = "QUATERNION";
            if apply_subsurf:
                bpy.ops.object.modifier_add(type="SUBSURF");
                bpy.ops.object.modifier_apply();
            #bpy.ops.object.modifier_add(type="SOLIDIFY"); #This modifier has its moments, but it screws up animations with planar molecules
            #self.mirror.modifiers["Solidify"].offset = 0;
            #self.mirror.modifiers["Solidify"].thickness = 0.01;
            #bpy.ops.object.modifier_apply();
        if not bpy.data.materials.get("mirror"):
            m = bpy.data.materials.new(name="mirror");
        m = bpy.data.materials.get("mirror");
        self.mirror.active_material = m;
        m.use_nodes = True;
        if m.node_tree.nodes.get("Principled BSDF"):
            m.node_tree.nodes["Principled BSDF"].inputs[14].default_value = 1.0;
            m.node_tree.nodes["Principled BSDF"].inputs[7].default_value = .1;
            m.node_tree.nodes["Principled BSDF"].inputs[15].default_value = 1;
            m.node_tree.nodes["Principled BSDF"].inputs[0].default_value = [0.52, 0.9, 1, 1];
        m.use_screen_refraction = True;
        m.blend_method = "BLEND"
        bpy.data.scenes["Scene"].eevee.use_ssr = True;
        bpy.data.scenes["Scene"].eevee.use_ssr_refraction = True;
        self.meshes.append(self.mirror);
        #self.hide(self.mirror);

    def make_rotation_axis(self, options, prettify_radii = True):
        distances = np.linalg.norm(self.position, axis = 1)
        if "depth" not in options:
            furthest_index = np.argmax(distances)
            if prettify_radii:
                radius = self.get_smaller_radii(self.radius)[furthest_index, np.newaxis]
            else:
                radius = self.radius[furthest_index, np.newaxis]
            options["depth"] = (distances[furthest_index] + 1.5*radius)*2;
        if "radius" not in options:
            options["radius"] = 0.025*(distances.max()**(1/1.5) if distances.max()**(1/2)>1 else 1);
        bpy.ops.mesh.primitive_cylinder_add(**options);
        self.rotation_axis = bpy.context.active_object;
        self.rotation_axis.rotation_mode = "QUATERNION";
        self.meshes.append(self.rotation_axis)
        # self.hide(self.rotation_axis);

    def hide(self, obj, hide_render = False):
        obj.hide_set(True);
        obj.hide_render = hide_render;

    def unhide(self, obj, unhide_render = True):
        obj.hide_set(False);
        obj.hide_render = unhide_render;


    def rotation_to(self, v2, v1 = [0, 0, 1]):
        """
        Rotation from v1 to v2
        """
        v1, v2 = np.array(v1)/np.linalg.norm(v1), np.array(v2)/np.linalg.norm(v2);
        axis = np.cross(v1, v2);
        angle = np.arccos(v1.dot(v2));
        return mathutils.Quaternion(axis, angle);

    def set_ending(self, tf):
        """
            Sets the ending of the recording at frame tf"""
        bpy.context.scene.frame_end = tf;

    #For the animation secion below, note that the animations follow the following scheme
    ## time ->   0    0   sp   1(sp+tt) 1(sp+tt)+sp 2(sp+tt)...
    ## interval - > 0   sp  tt      sp        tt     ...

    #Animation durations
    ## Rotation and reflection 3*(sp + tt)
    ## Improper_rotation 5*(sp + tt)
    ## identity (sp + tt)
    ## inversion 2*(sp + tt)


    def animate_rotation(self, t0, normal, angle, orbitals = "pz"):
        """
        Performs the appropriate rotations and scalings to make a reflection animation.
        The times can be tweaked through short_pause and transition.
        returns the time at which the animation ends."""
        #Creating the carrier of the molecule
        self.deselect_all();
        parent_mule = self.make_empty();
        parent_mule.rotation_quaternion = self.rotation_to(normal);
        self.set_parent(parent_mule, self.mules[-2]); #Note that self.mules[-1] will contain parent_mule

        tt = self.transition;
        sp = self.short_pause;
        tf = t0 + 3*(tt + sp);

        #animating the cylinder that shows the axis of rotation
        self.rotation_axis.rotation_quaternion = parent_mule.rotation_quaternion.copy();
        self.rotation_axis.keyframe_insert(data_path = "rotation_quaternion", frame = t0 + sp);
        self.rotation_axis.keyframe_insert(data_path = "rotation_quaternion", frame = tf);

        self.add_transition("scale", self.rotation_axis, t0 + sp, tt, [0, 0, 0], [1, 1, 1])
        self.add_transition("scale", self.rotation_axis, tf - tt, tt, [1, 1, 1], [0, 0, 0]);
        
        #Animating the actual rotation of the carrier
        rot_init = parent_mule.rotation_quaternion.copy();
        rot = rot_init.copy()
        rot.rotate(mathutils.Quaternion(normal, angle));
        self.add_transition("rotation_quaternion", parent_mule, t0+2*sp+tt, tt, rot_init, rot);
        self.add_transition("rotation_quaternion", parent_mule, tf-tt, tt, rot, rot_init);

        if orbitals:
            mat = np.array(mathutils.Matrix.Rotation(angle, 3, normal));
            in_place = self.find_static_orbitals(mat)
            scales = np.abs(self.find_final_projections(mat, orb = orbitals))[self.find_static_atoms(mat).astype(bool)]
            meshes_in_place = [self.orbital_meshes[i] for i, stayed in enumerate(in_place) if stayed]
            fading_meshes = [mesh for mesh in self.orbital_meshes if mesh not in meshes_in_place and mesh]
            
            t2 = int(tt/2)
            for i, mesh in enumerate(meshes_in_place):
                #Making them shine
                self.add_material_transition("emission", mesh, t0 + t2, t2, (0, 0, 0, 1), (.2, .2, .2, 1))
                self.add_material_transition("emission", mesh, tf - t2, t2,(.2, .2, .2, 1), (0, 0, 0, 1))

                #Making them SMOL
                self.add_transition("scale", mesh, t0 + t2, t2, [1]*3, [scales[i]]*3)
                self.add_transition("scale", mesh, tf - t2, t2, [scales[i]]*3, [1]*3)
                
            for mesh in fading_meshes:
                #Initial fade out
                self.add_material_transition("alpha", mesh, t0 + t2, t2, 1, .5)
                self.add_material_transition("alpha", mesh, t0 + tt + sp*2, t2, .5, 0)
                self.add_material_transition("alpha", mesh, tf - t2, t2, 0, 1)

        self.set_ending(tf);
        return tf; #Returns the end of this animation

    def add_transition(self, property, obj, t0, transition, init, end):
        self.deselect_all();
        obj.select_set(True);
        self.set_active(obj);
        setattr(obj, property, init);
        obj.keyframe_insert(data_path = property, frame = t0);
        setattr(obj, property, end);
        obj.keyframe_insert(data_path = property, frame = t0+transition);

    def add_material_transition(self, property_name, obj, t0, transition, init, end, slot = "all"):
        property_index = {"base_color": 0, "transmission": 15, "emission": 17, "alpha": 18}[property_name]
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
            

    def animate_reflection(self, t0, normal, orbitals = "pz"):
        """
        Performs the appropriate rotations and scalings to make a reflection animation.
        The times can be tweaked through short_pause and transition.
        returns the time at which the animation ends."""
        #Creating the carrier of the molecule
        self.deselect_all();
        parent_mule = self.make_empty();
        parent_mule.rotation_quaternion = self.rotation_to(normal);
        self.set_parent(parent_mule, self.mules[-2]); #Note that self.mules[-1] will contain parent_mule

        tt = self.transition;
        sp = self.short_pause;
        tf = t0 + 3*(tt + sp);

        #animating the cylinder that shows the axis of rotation
        self.mirror.rotation_quaternion = parent_mule.rotation_quaternion.copy();
        self.mirror.keyframe_insert(data_path = "rotation_quaternion", frame = t0 + sp);
        self.mirror.keyframe_insert(data_path = "rotation_quaternion", frame = tf);

        self.add_transition("scale", self.mirror, t0 + sp, tt, [0, 0, 0], [1, 1, 1])
        self.add_transition("scale", self.mirror, tf - tt, tt, [1, 1, 1], [0, 0, 0]);

        #Animating the actual rotation of the carrier
        scale_init = parent_mule.scale.copy();
        scale_final = np.array(scale_init)*[1, 1, -1];
        self.add_transition("scale", parent_mule, t0+2*sp+tt, tt, scale_init, scale_final);
        self.add_transition("scale", parent_mule, tf-tt, tt, scale_final, scale_init);

        if orbitals:
            mat = np.array(mathutils.Matrix.Scale(-1, 3, normal));
            in_place = self.find_static_orbitals(mat)
            scales = np.abs(self.find_final_projections(mat, orb = orbitals))[self.find_static_atoms(mat).astype(bool)]
            meshes_in_place = [self.orbital_meshes[i] for i, stayed in enumerate(in_place) if stayed]
            fading_meshes = [mesh for mesh in self.orbital_meshes if mesh not in meshes_in_place and mesh]

            t2 = int(tt/2)
            for i, mesh in enumerate(meshes_in_place):
                #Making them shine
                self.add_material_transition("emission", mesh, t0 + t2, t2, (0, 0, 0, 1), (.2, .2, .2, 1))
                self.add_material_transition("emission", mesh, tf - t2, t2,(.2, .2, .2, 1), (0, 0, 0, 1))

                #Making them SMOL
                self.add_transition("scale", mesh, t0 + t2, t2, [1]*3, [scales[i]]*3)
                self.add_transition("scale", mesh, tf - t2, t2, [scales[i]]*3, [1]*3)
                
            for mesh in fading_meshes:
                #Initial fade out
                self.add_material_transition("alpha", mesh, t0 + t2, t2, 1, .5)
                self.add_material_transition("alpha", mesh, t0 + tt + sp*2, t2, .5, 0)
                self.add_material_transition("alpha", mesh, tf - t2, t2, 0, 1)

        self.set_ending(tf);
        return tf; #Returns the end of this animation


    def animate_improper_rotation(self,t0, normal, angle, orbitals = "pz"):
        """
        Performs the appropriate rotations and scalings to make a reflection animation.
        The times can be tweaked through short_pause and transition.
        returns the time at which the animation ends."""
        #Creating the carrier of the molecule
        self.deselect_all();
        parent_mule = self.make_empty();
        parent_mule.rotation_quaternion = self.rotation_to(normal);
        self.set_parent(parent_mule, self.mules[-2]); #Note that self.mules[-1] will contain parent_mule

        tt = self.transition;
        sp = self.short_pause;
        tf = t0 + 5*(tt + sp);

        #animating the cylinder that shows the axis of rotation
        for o in [self.rotation_axis, self.mirror]:
            o.rotation_quaternion = parent_mule.rotation_quaternion.copy();
            o.keyframe_insert(data_path = "rotation_quaternion", frame = t0 + sp);
            o.keyframe_insert(data_path = "rotation_quaternion", frame = tf);

        self.add_transition("scale", self.rotation_axis, t0 + sp, tt, [0, 0, 0], [1, 1, 1])
        self.add_transition("scale", self.rotation_axis, tf - tt, tt, [1, 1, 1], [0, 0, 0]);

        self.add_transition("scale", self.mirror, t0 + sp + 2*(sp+tt), tt, [0, 0, 0], [1, 1, 1])
        self.add_transition("scale", self.mirror, tf - tt, tt, [1, 1, 1], [0, 0, 0]);

        #Animating the actual rotation of the carrier
        rot_init = parent_mule.rotation_quaternion.copy();
        rot = rot_init.copy();
        rot.rotate(mathutils.Quaternion(normal, angle));
        self.add_transition("rotation_quaternion", parent_mule, t0+1*(sp+tt)+sp, tt, rot_init, rot);
        self.add_transition("rotation_quaternion", parent_mule, tf-tt, tt, rot, rot_init);

        #Animating the actual rotation of the carrier
        scale_init = parent_mule.scale.copy();
        scale_final = np.array(scale_init)*[1, 1, -1];
        self.add_transition("scale", parent_mule, t0 + sp + 3*(sp+tt), tt, scale_init, scale_final);
        self.add_transition("scale", parent_mule, tf-tt, tt, scale_final, scale_init);

        if orbitals:
            mat = np.array(mathutils.Matrix.Rotation(angle, 3, normal));
            in_place = self.find_static_orbitals(mat)
            scales = np.abs(self.find_final_projections(mat, orb = orbitals))[self.find_static_atoms(mat).astype(bool)]
            meshes_in_place = [self.orbital_meshes[i] for i, stayed in enumerate(in_place) if stayed]
            fading_meshes = [mesh for mesh in self.orbital_meshes if mesh not in meshes_in_place and mesh]

            t2 = int(tt/2)
            print(meshes_in_place)
            print(scales)
            print(fading_meshes)
            for i, mesh in enumerate(meshes_in_place):
                #Making them shine
                self.add_material_transition("emission", mesh, t0 + t2, t2, (0, 0, 0, 1), (.2, .2, .2, 1))
                self.add_material_transition("emission", mesh, tf - t2, t2,(.2, .2, .2, 1), (0, 0, 0, 1))

                #Making them SMOL
                self.add_transition("scale", mesh, t0 + t2, t2, [1]*3, [scales[i]]*3)
                self.add_transition("scale", mesh, tf - t2, t2, [scales[i]]*3, [1]*3)
                
            for mesh in fading_meshes:
                #Initial fade out
                self.add_material_transition("alpha", mesh, t0 + t2, t2, 1, .5)
                self.add_material_transition("alpha", mesh, t0 + tt + sp*2, t2, .5, 0)
                self.add_material_transition("alpha", mesh, tf - t2, t2, 0, 1)
            fading_meshes_rotation = fading_meshes

        if orbitals:
            mat = np.array(mathutils.Matrix.Scale(-1, 3, normal)@mathutils.Matrix.Rotation(angle, 3, normal));
            in_place = self.find_static_orbitals(mat)
            scales = np.abs(self.find_final_projections(mat, orb = orbitals))[self.find_static_atoms(mat).astype(bool)]
            fading_meshes = [mesh for mesh in self.orbital_meshes if mesh not in meshes_in_place and mesh not in fading_meshes_rotation and mesh]

            t2 = int(tt/2)
            t0 += 3*sp + 2*tt #For the rotation
            for mesh in fading_meshes:
                #Initial fade out
                self.add_material_transition("alpha", mesh, t0 + t2, t2, 1, .5)
                self.add_material_transition("alpha", mesh, t0 + tt + sp*2, t2, .5, 0)
                self.add_material_transition("alpha", mesh, tf - t2, t2, 0, 1)
            t0 -= 3*sp + 2*tt

        self.set_ending(tf);
        return tf; #Returns the end of this animation


    def animate_identity(self, t0, orbitals = "pz"):
        tf = t0 + self.transition + self.short_pause;
        #The step below just hides the mirror in case the molecule being rendered does not have any symmetry operations that use the mirror
        for o in (self.mirror, self.rotation_axis):
            o.scale = mathutils.Vector([0, 0, 0]);
            o.keyframe_insert(data_path = "scale", frame = t0);

        t2 = int(self.transition/3)
        sp = self.short_pause
        meshes = [om for om in self.orbital_meshes if om]; #Some meshes may be set to None
        for i, mesh in enumerate(meshes):
            #Making them shine
            self.add_material_transition("emission", mesh, t0 + sp, t2, (0, 0, 0, 1), (.2, .2, .2, 1))
            self.add_material_transition("emission", mesh, tf - 2*t2, t2,(.2, .2, .2, 1), (0, 0, 0, 1))
        self.set_ending(tf);
        return tf;

    def animate_inversion(self, t0, orbitals = "pz"):
        sp, tt = self.short_pause, self.transition;
        tf = t0 + 2*(sp + tt);

        #Unlike in the other transformations, we do not need to make a new mule. Mule 0 does not have any inclination, so we can just invert it
        mule = self.empty;
        scale_init = mule.scale.copy();
        scale_final = np.array(scale_init)*-1;
        self.add_transition("scale", mule, t0 + sp, tt, scale_init, scale_final);
        self.add_transition("scale", mule, tf - tt, tt, scale_final, scale_init);


        if orbitals:
            mat = -np.eye(3);
            in_place = self.find_static_orbitals(mat)
            scales = np.abs(self.find_final_projections(mat, orb = orbitals))[self.find_static_atoms(mat).astype(bool)]
            meshes_in_place = [self.orbital_meshes[i] for i, stayed in enumerate(in_place) if stayed]
            fading_meshes = [mesh for mesh in self.orbital_meshes if mesh not in meshes_in_place and mesh]

            t2 = int(tt/2)
            for i, mesh in enumerate(meshes_in_place):
                #Making them shine
                self.add_material_transition("emission", mesh, t0 + t2, t2, (0, 0, 0, 1), (.2, .2, .2, 1))
                self.add_material_transition("emission", mesh, tf - t2, t2,(.2, .2, .2, 1), (0, 0, 0, 1))

                #Making them SMOL
                self.add_transition("scale", mesh, t0 + t2, t2, [1]*3, [scales[i]]*3)
                self.add_transition("scale", mesh, tf - t2, t2, [scales[i]]*3, [1]*3)
                
            for mesh in fading_meshes:
                #Initial fade out
                self.add_material_transition("alpha", mesh, t0 + t2, t2, 1, 0)
                self.add_material_transition("alpha", mesh, tf - t2, t2, 0, 1)
        self.set_ending(tf);
        return tf;

    def setup_dir(self, name):
        if not os.path.isdir(name):
            os.makedirs(name);
        return name;

    def angles_from_normals(self, normals):# In case we ever dicide to encode the angles in the form of the length of the normal vectors
        angles = [];
        for n in normals: #here the LIST of normals is referenced, not the normals!
            if len(n)>0:
                angles.append(np.linalg.norm(n[0])); #Note how the actual normal is extracted from the length
            else:
                angles.append(0);
        return angles;

    def animate_properties(self, cc_list, normals, angles = [], target_directory = "Animation_files",t0 = 0, orbitals = "pz"):
        """
            Angles must be supplied as a list. Transformations which do not use angles should have a None on the list OR a 0
            If angles are not supplied, the length of the normals is used instead.
            """
        if len(angles) == 0:
            angles = self.angles_from_normals(normals);
        starts = [t0]; #Starting times of each animation
        starts_directories = []; #Name of each directory corresponding to a "start"
        directory = os.path.join(os.getcwd(),target_directory); #All the animations will be stored in directories inside this directory
        for ci, cc in enumerate(cc_list): #For each conjugacy class we make a directory
            new_dir = os.path.join(directory, cc);
            pattern = re.compile(r"([A-Z]|i|sig)");
            operation = pattern.findall(cc)[0];
            func = {
                "E": self.animate_identity,
                "C": self.animate_rotation,
                "i": self.animate_inversion,
                "S": self.animate_improper_rotation,
                "sig": self.animate_reflection}[operation]; #For each operatio in the conjugacy class we setup the animation and make the directory
            print(operation)
            if operation == "i" or operation == "E":
                d = os.path.join(new_dir,cc);
                starts_directories.append(d);
                t0 = func(t0, orbitals); #Note how the dimensions of e and i are still then same as the other transformations
                starts.append(t0);
            else:
                for ni, n in enumerate(normals[ci]):
                    d = os.path.join(new_dir, cc + "_" + str(ni));
                    d = PointGroup.normal_to_directory(d); #Removes all the primes from the directory name
                    starts_directories.append(d);
                    if operation == "sig":
                        t0 = func(t0, n, orbitals);
                    else:
                        t0 = func(t0, n, angles[ci], orbitals)
                    starts.append(t0);
        return starts, starts_directories; #Now we can just render into starts_directories

    @staticmethod # I also got this function from Ivo
    def write_xyz(names, vertices, outfile):
        # output results
        f = open(outfile, 'w')
        f.write("%i\n\n" % len(names))
        for n, v in zip(names, vertices):
            f.write('%s  %12.6f  %12.6f  %12.6f\n' % (n, *v));
        f.close()

    
    def render_animation(self, times, directs):
        """
        Times are the ending times of each animation (and beggining of the first)
        directs is the list of directories for each time range!
        """
        zfill = len(str(bpy.data.scenes["Scene"].frame_end)); #Ensures correct ordering!!!
        bpy.data.scenes["Scene"].render.image_settings.file_format = "PNG";
        ranges = [list(range(times[i], times[i+1])) for i in range(len(times[:-1]))];
        for ir, r in enumerate(ranges):
           for frame in r:
               self.setup_dir(directs[ir]);
               bpy.context.scene.render.filepath = os.path.join(directs[ir], str(frame).zfill(zfill));
               bpy.data.scenes["Scene"].frame_current = frame;
               bpy.ops.render.render(use_viewport = True, write_still=True);


    def render_image(self, target_directory, name = None, hide = True):
        if hide:
            try:
                self.hide(self.mirror, hide_render = True);
                self.hide(self.rotation_axis, hide_render = True);
            except:
                None
        bpy.data.scenes["Scene"].render.image_settings.file_format = "PNG";
        bpy.context.scene.render.filepath = os.path.join(os.getcwd(), target_directory, self.name if not name else name);
        bpy.data.scenes["Scene"].frame_current = 0;
        bpy.ops.render.render(use_viewport = True, write_still=True);
