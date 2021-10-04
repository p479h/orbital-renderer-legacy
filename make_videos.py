import os;
import numpy as np
from itertools import product
from PIL import Image
import multiprocessing
from add_labels import *
from Molecule import Molecule
"""
    To use this file, go ALL the way to it's end and list the JSON file names that you wish to compile into a clip
    """

#First we define three functions that may be useful when interpolating
def lagrange(xc, yc, x):
    xc, yc, x = [np.array(i) for i in (xc, yc, x)]
    xk = xc[:, np.newaxis]
    xj = np.tile(xc, (len(xc), 1))[~np.eye(len(xc), dtype=bool)].reshape(len(xc), len(xc)-1)
    bottom = xk - xj;
    numerator = x[:, np.newaxis, np.newaxis] - xj;
    fraction = numerator/bottom
    return np.sum(fraction.prod(axis = 2)*yc, axis = 1);


def bezier(start, finish, n = 30):
    t = np.linspace(0, 1, n);
    x = np.array([start, start, finish, finish]);
    P = (1-t)**3*x[0] + 3*(1-t)**2*t*x[1] +3*(1-t)*t**2*x[2] + t**3*x[3];
    return P; #Note that n plays the role of the frames

def bezier2D(start, finish, n = 30):
    start, finish = [np.array(i) for i in (start, finish)]
    t = np.linspace(0, 1, n);
    x = np.hstack( (start.reshape(-1, 1), start.reshape(-1, 1), finish.reshape(-1,1), finish.reshape(-1, 1)) );
    P = (1-t)**3*x[:, 0, np.newaxis] + 3*(1-t)**2*t*x[:, 1, np.newaxis] +3*(1-t)*t**2*x[:, 2, np.newaxis] + t**3*x[:, 3, np.newaxis];
    return P.T; #Note that here we frames on the 0 axis and property on the 1 axis

#Loop over the directories that do not contain animations
def group_from_dir(d, d_index, base, zfill, pg, name = None):
    print(d)
    print("Is currently running");
    animation_path = os.path.join(os.path.abspath(os.path.join(d, "..")), "a_"+os.path.basename(os.path.normpath(d)));
    if not os.path.exists(animation_path):
        os.makedirs(animation_path)

    #Get a list of the place where to find the pictures
    inner_dirs = [os.path.join(d, inner_d) for inner_d in os.listdir(d)];

    #Now we check how we have to set the layout to best accomodate those transformations
    nt = len(inner_dirs) #Number of elements to accomodate

    #Get the ratio of width/height to find the best picture arrangement
    ratio = base.size[0]/base.size[1];

    numbers = np.array([[i, j] if ratio<=1 else [j, i] for i in range(10) for j in range(10)]); #side lengths
    try_nts = numbers.prod(axis = 1);
    pair = numbers[np.argmin(np.abs(try_nts - nt))];
    if pair[0] == 6 and pair[1] == 2: #Some of these require experimenting
        pair[0] = 4;
        pair[1] = 3;

    if (pair[0] == 6 or pair[0] == 8) and pair[1] == 1:
        pair[0] = 3 if pair[0] == 6 else 4;
        pair[1] = 2;
        #Pair gives the number of elements in horizontal against vertical

    #Now we make a grid that can be accessed by index (For the center of the figures
    horiz = np.linspace(0, base.size[0], pair[0] + 2).astype(int);
    vert = np.linspace(0, base.size[1], pair[1] + 2).astype(int);

    #Now we check wether we must change the ratio wrt to height or width of the pic
    dw = int(np.diff(horiz)[0]);
    dh = int(np.diff(vert)[0]);
    wid = min([dw, dh]);

    #Now we have to add a single frame at a time to the animation path
    nframes = len(os.listdir(inner_dirs[0]))

    #We make the label image
    label_latex = latexify_term(extract_class(PointGroup.directory_to_normal(os.path.basename(d))));    
    label_img = latex_to_img(label_latex, None, scale = 100, lw = 2, edgecolor = None, usetex = True);

    #Make the header label
    images = draw_header_images(pg.pg_name, scale = 100, lw = 3);
    joined, pixels = join_header_images(images);
    W, H = base.size;
    h = int(H/10) #10
    w = int(h/joined.size[1] * joined.size[0])
    if w >= W:
        w = int(W*.9)
        h = int(w/joined.size[0] * joined.size[1])
    new_ratio = w/joined.size[0];
    joined = joined.resize((w, h));
    x = int(base.size[0]/2 - w/2);
    y = int(h/2);
    base.paste(joined, (x, y), joined);
    padding = 10;
    pixels = (np.array(pixels)*new_ratio).astype(int)+[x, y+padding, x, y+padding];

    #Adding the name of the molecule
    padding = 20; #It is ok to ovewrite h rn
    h = int(H/12);
    name_image = latex_to_img(name, color = "#f7f0dd", scale = 100, lw=1, edgecolor = "#f7f0dd", usetex = False);
    new_width = int((h/name_image.size[1])*name_image.size[0]);
    name_image = name_image.resize((new_width, h));
    xx, yy = int(W - new_width - padding*.6), H - h - padding;
    base.paste(name_image, (xx, yy), name_image);

    #Adding the logos
    tue_logo = Image.open("iTUe-logo-descriptor-line-scarlet-rgb.png")
    imc_logo = Image.open("imc_sidetext_light.png")

    #Resizing the tue logo
    new_width = int(h/tue_logo.size[1]*tue_logo.size[0])
    tue_logo = tue_logo.resize((new_width, h), Image.ANTIALIAS)

    #Resizing the imc logo
    new_width = int(h/imc_logo.size[1]*imc_logo.size[0])
    imc_logo = imc_logo.resize((new_width, h), Image.ANTIALIAS)

    #getting the anchor points for both logos and pasting them
    spacing = 7; #Pixels between letters
    tuexx = xx - int(tue_logo.size[0]*.95)
    imcxx = tuexx - imc_logo.size[0] - spacing;
    base.paste(tue_logo, (tuexx, yy+int((name_image.size[1]-tue_logo.size[1])/2)), tue_logo)
    base.paste(imc_logo, (imcxx, yy+int((name_image.size[1]-imc_logo.size[1])/2)) , imc_logo)
    
    #Here we do something for the molecules going in:
    duration = 20; #frames
    w, h = label_img.size;
    resize_duration = 20;
    offset = bezier2D([-w*2, H-h+10],[30, H-h+10], duration*2).astype(int)[duration:];
    croppers = bezier2D([wid/2, wid/2, wid/2, wid/2],[0, 0, wid, wid],resize_duration).astype(int);
    pics = [[os.path.join(inner_dir, pic) for pic in os.listdir(inner_dir)] for inner_dir in inner_dirs];

    #Now we interpolate the positions of the pixels that make up the line under the header
    if d_index>0:
        from_previous = bezier2D(pixels[d_index-1], pixels[d_index], duration*2)[duration:].astype(int);
    else:
        from_previous = bezier2D(pixels[d_index], pixels[d_index], duration).astype(int)
    if d_index < len(pixels)-1:
        to_next = bezier2D(pixels[d_index], pixels[d_index+1], duration*2)[:duration].astype(int);
    else:
        to_next = bezier2D(pixels[d_index], pixels[d_index], duration).astype(int)
      
    means = np.vstack((from_previous, to_next)).mean(axis = 0).astype(int);
    for arr in (from_previous, to_next):#The slider trembles a bit as a result of rounding
        for i in [1, 3]: #Y coordinates
            arr[:, i] = means[i];
      
    for i in range(duration):
        img = base.copy();
        img = add_label(img, label_img, top_left = tuple(offset[i]));
        img = draw_underline(img, from_previous[i], width=1)
        for count, inner_dir in enumerate(inner_dirs): #Loop over the symmetry operations
            hi = count%pair[0] + 1;
            vi = count//pair[0] + 1;
            frame = Image.open(pics[count][0]).resize((wid, wid))
            top_left = (horiz[hi]-int(wid/2), vert[vi]-int(wid/2));
            if i < resize_duration:
                #frame = frame.crop(croppers[i]);
                frame = frame.resize((croppers[i][2] - croppers[i][0]+1, croppers[i][3] - croppers[i][1]+1));
                top_left = tuple(top_left + croppers[i, :2])
            img.paste(frame, top_left, frame)
        img.save(os.path.join(animation_path, str(i).zfill(zfill)+".png"), "PNG")
      
    for iframe in range(duration, duration+nframes):
        img = base.copy(); #Make a copy of the background
        img = add_label(img, label_img, top_left = tuple(offset[-1]));
        img = draw_underline(img, from_previous[-1], width=1)
        for count, inner_dir in enumerate(inner_dirs): #Loop over the symmetry operations
            hi = count%pair[0] + 1;
            vi = count//pair[0] + 1;

            frame = Image.open(pics[count][iframe-duration]).resize((wid, wid))
            img.paste(frame, (horiz[hi]-int(wid/2), vert[vi]-int(wid/2)), frame)
        img.save(os.path.join(animation_path, str(iframe).zfill(zfill)+".png"), "PNG")

    offset = bezier2D([30, H-h+10],[30*2, H*2], duration*2).astype(int)[:duration];
    croppers = croppers[::-1, :]; #We wish to play it in reverse
    for ii, i in enumerate(range(iframe+1, iframe+1 + duration)):
        img = base.copy();
        img = add_label(img, label_img, top_left = tuple(offset[ii]));
        img = draw_underline(img, to_next[ii], width=1)
        for count, inner_dir in enumerate(inner_dirs): #Loop over the symmetry operations
            hi = count%pair[0] + 1;
            vi = count//pair[0] + 1;
            frame = Image.open(pics[count][0]).resize((wid, wid))
            top_left = (horiz[hi]-int(wid/2), vert[vi]-int(wid/2))
            if ii >= duration - resize_duration:
                index = ii - (duration - resize_duration)
                #frame = frame.crop(croppers[index]);
                frame = frame.resize((croppers[index][2] - croppers[index][0]+1, croppers[index][3] - croppers[index][1]+1));
                top_left = tuple(top_left + croppers[index, :2])
            img.paste(frame, top_left, frame)
        img.save(os.path.join(animation_path, str(i).zfill(zfill)+".png"), "PNG")

    print(animation_path)
    print("Rendered");


def compile_transitions(filename, directory = "molecule_data"):
    """
        Takes in the json filename of the molecule whose frames are to be animated
        Checks if any frames are missing from the conjugacy class directories.
        If no files are missing, it proceeds to use group_from_dir() to assemble the frames of the final animation"""
    bg = Image.open("template.psd"); #Opening the background
    img = Image.new("RGBA", bg.size, (0, 0, 0, 0)); #Creating img with alpha channel
    img.paste(bg, (0, 0)); #Adding the background to it
    base = img.copy()
    data = Molecule.load_molecule_data(filename, directory);
    pg = PointGroup(data["point_group"]);
    #data["name"] = "Sulflower";
    name = data["name"] if "name" in data else None; #Molecule name to be displayed on the corner of animation.
    target_dir = os.path.join(os.getcwd(),data["target_directory"])

    #Now we check if the number of elements in the each subfile is correct! This confirms that the animation has been fully rendered
    ccs = [os.path.join(target_dir, PointGroup.normal_to_directory(cc)) for cc in pg.conjugacy_classes];
    for ccdir in ccs: #Checks if the number of pictures fo all summetry operatios of the same conjugacy class have the same number of pictures (rendering has not been interupted"
        try:
            inner_dirs = [os.path.join(ccdir, d) for d in os.listdir(ccdir)];
            true_count = len(os.listdir(inner_dirs[0]))
            for inner_dir in inner_dirs:
                number_pics = len(os.listdir(inner_dir))
                if number_pics != true_count or number_pics == 0:
                    with open("errorlog.txt", "a") as f:
                        f.write("\n\n")
                        f.write("Problems in the following directory:\n")
                        f.write(inner_dir + "\n");
                        f.write(f"Should have had {true_count} but had {number_pics}\n")
                        print("\n\n")
                        print("Problems in the following directory:\n")
                        print(inner_dir + "\n");
                        print(f"Should have had {true_count} but had {number_pics}\n")
                        break
    
        except:
            with open("errorlog.txt", "a") as f: #We do this because sometimes when using multiprocessing, the errors and print statements are not displayed on screen
                f.write("\n\n")
                f.write("Missing conjugacy class:\n")
                f.write(ccdir + "\n");
                print("\n\n")
                print("Missing conjugacy class:\n")
                print(ccdir + "\n");
                break


    #This will later be for loops over every
    dirs = [os.path.join(target_dir, d) for d in pg.conjugacy_classes];
    dirs = [PointGroup.normal_to_directory(d) for d in dirs]
    print(pg.conjugacy_classes)
    print("Will be rendered");
    print("They are in the following directories: ");
    for d in dirs:
        print(d);

    #Making a variable to count the number of frames in the final animation to compute zfill, which helps order the frames
    total_frame_count = 0
    for d in [d for d in dirs if os.path.basename(os.path.normpath(d))[:2]!="a_" and os.path.exists(d)]: #a_ is short for animation_
         d = PointGroup.normal_to_directory(d);
         inner_dirs = [os.path.join(d, inner_d) for inner_d in os.listdir(d)];
         total_frame_count += len(os.listdir(inner_dirs[0]));
    zfill = len(str(total_frame_count))
    print("The total frame count is ", total_frame_count);
            
    dirs_to_loop = [d for d in dirs if os.path.basename(os.path.normpath(d))[:2]!="a_" and os.path.exists(d)]; #a_ is short for animation_
    #with multiprocessing.Pool(processes=multiprocessing.cpu_count()-1) as pool:
    r = pool.starmap(group_from_dir, tuple([(d, d_index, base.copy(), zfill, pg, name) for d_index, d in enumerate(dirs_to_loop)]))


def save_batch(target, inner_dir, inner_dir_labels, delete_after = True):
    for j, pic in enumerate(os.listdir(inner_dir)):
        #Image.open(os.path.join(inner_dir, pic)).save( os.path.join(target, inner_dir_labels[j])+".png" )
        os.rename(os.path.join(inner_dir, pic), os.path.join(target, inner_dir_labels[j])+".png" )
        #if delete_after:
        #    os.remove(os.path.join(inner_dir, pic))
    if delete_after:
        os.rmdir(inner_dir)
        

def sort_premiere(filename, directory = "molecule_data"):
    """
        Takes all images generated per conjugacy class and orders them into a single directory"""
    print("MOlecule about to be sorted")
    print(filename)
    data = Molecule.load_molecule_data(filename, directory);
    pg = PointGroup(data["point_group"]);
    target = os.path.join(data["target_directory"],"premiere_proof")
    if not os.path.exists(target):
        os.makedirs(target)
    ccs = [os.path.join(data["target_directory"], "a_"+PointGroup.normal_to_directory(cc)) for cc in pg.conjugacy_classes]
    ccs = [c for c in ccs if os.path.exists(c)]
    counts = [len(os.listdir(inner_dir)) for inner_dir in ccs]
    cumsum = np.r_[[0],np.cumsum(counts)]
    zfill2 = len(str(cumsum.max()));
    labels = [list(range(cumsum[i], cumsum[i+1])) for i in range(len(cumsum)-1)]

    args = [(target, inner_dir, [str(l).zfill(zfill2) for l in labels[i]]) for i, inner_dir in enumerate(ccs)]
    r = pool.starmap(save_batch, args)
            
filenames = ("data_ferrocene.json", )
directory = "molecule_data"
if __name__ == "__main__":
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        for filename in filenames:
            print("COMPILING TRANSITIONS");
            compile_transitions(filename);
            print("SORTING PREMIERE");
            sort_premiere(filename);
