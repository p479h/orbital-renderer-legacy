import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import io
import os;
import re;
import numpy as np
from PIL import Image, ImageChops, ImageDraw
from PointGroup import PointGroup
import sys
import cairosvg
from PyQt5.QtWidgets import QApplication
app = QApplication(sys.argv)
screen = app.screens()[0]
dpi = screen.physicalDotsPerInch() #now we know the dpi
app.quit()


def latexify_term(t):
    """
        t: string conjugacy class that will be parsed to be used with LaTeX
        """
    t = t.replace("sigma", "sig").replace("Sigma", "sig").replace("infty", "inf").replace("Infty", "inf").replace("_prime", "'");
    #Takes a term t and makes it latex friendly (Could be improved I think)
    pattern = r"([0-9]+)?(\(?)([A-Z]|Sig|sig|sigma|Sigma)('*)([a-z0-9]+|Inf|inf|infty|Infty)?(\))?([0-9]+)?('*)";
    pattern = re.compile(pattern);    
    subbed = pattern.sub(r"\1\2\3_{\5}\6^{\4\7\8}", t);
    if subbed.find(")") != -1:
        subbed = pattern.sub(r"\1\2\3^{\4}_{\5}\6^{\7}", t);
    return subbed.replace("^{}", "").replace("_{}", "").replace("sig", r"\sigma ").replace("inf", r"\infty").replace("'", r"\prime ");

def latexify_terms(terms):
    return [latexify_term(t) for t in terms];
     

def latex_to_img(tex, color = None, scale = 30, lw = 1, edgecolor = None, usetex = True):
    """
        tex: latex text that will be made into image
        color: color of the text filling (use settings that matplotlib understands)
        scale: integer which can be increased for higher resolutions
        lw: width of the colored outline (edge)
        edgecolor: color of the edge (use settings that matplotlib understands)
        usetex: boolean that can be set to False to speed up tests
        """
        
    buf = io.BytesIO()
    plt.rc('text', usetex=usetex)
    plt.rc('font', family='serif')
    plt.rcParams['text.latex.preamble'] = [r'\boldmath']
    if not usetex:
        plt.rc('font', family='sans-serif')
    fig = plt.figure();
    ax = fig.add_axes([0, 0, 1, 1]);
    ax.axis('off')
    ax.patch.set_alpha(0);
    fig.patch.set_alpha(0);

    if type(color) == type(None) or type(edgecolor) == type(None):
        color = edgecolor = "#f7f0dd" if "E" in tex else "#2688ce" if ("S" in tex or "i" == tex) else "#cb302d" if "C" in tex else "#f7f0dd" if tex == "|" else "#7d9002";
        
    t = ax.text(0, 0, f'${tex}$' if usetex else "$\\mathrm{"+tex+"}$", size=scale, ha="center", va="center", color=color);
    t.set_path_effects([path_effects.Stroke(linewidth=lw, foreground=edgecolor),])
    r = fig.canvas.get_renderer();
    bb = t.get_window_extent(r)
    bb2 = bb.transformed(fig.transFigure.inverted())
    ax.set(
        xlim = (bb2.x0, bb2.x1),
        ylim = (bb2.y0, bb2.y1));
    bb = t.get_window_extent(r)
    bb3 = bb.transformed(fig.dpi_scale_trans.inverted())
    fig.set_size_inches(bb3.width*1.1, bb3.height*1.1); #We multiply by 1.1 to ensure no tiny bits of the text get cropped out
    plt.savefig(buf, format='png')
    plt.close();
    
    im = Image.open(buf)
    bg = Image.new(im.mode, im.size, (255, 255, 255, 255))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox();
    return im.crop(bbox)

def extract_class(term):
    """
        term: text with transformation coming from point group table
        """
    pattern = re.compile(r"\(?[A-Za-z]+");
    start_index = next(pattern.finditer(term)).start();
    return term[start_index:] #Actual term that indicates the kind of transformation of the conjugacy class
    

def add_label(bg,label_img, w = [], top_left = []): #If we want to add the same labels many times we can stick this in a for loop instead of rerendering everytime.
    """
        This function adds he label_img to the bottom left corner of bg
        bg: pillow image
        label_img: pillow image
        w: int with the width of the label_img when added to bg
        top_left: coordinates of the top left of label_img when added to bg
        """
        
    if len(w) == 0:
        h = int(bg.size[1]/12);
        w = int(h/label_img.size[1]*label_img.size[0]);
        w = (w, h)
    if len(top_left) == 0:
        top_left = (int(w[0]/100), bg.size[1]-w[1]);
    label_img = label_img.resize(w);
    bg.paste(label_img, top_left, label_img);
    return bg;

def draw_header_images(point_group, color = (0, 0, 0, 0), edgecolor = "gray", scale = 50, lw = 1):
    """
        Returns a list with all relevant images
        point_group: string with the name of relevant point group
        color: color of the filling of the text
        edgecolor: color of the edge of the text
        scale: integer which can be increased for higher resolutions
        lw: width of the colored outline (edge)
        """
    pg = PointGroup(point_group);
    header = (latexify_term(pg.pg_name)+" || "+" | ".join([latexify_term(t).replace(" ", "") for t in pg.conjugacy_classes])).split(" ")
    images = []
    for i, h in enumerate(header):
        c = "#f7f0dd" if "E" in h else "#2688ce" if (("S" in h) or ("i" == h)) else "#cb302d" if "C" in h else "#f7f0dd" if (h == "|" or h == "||") else "#7d9002";
        if i == 0: #For the point group name 
            c = "#f7f0dd";
        images.append(latex_to_img(h, c, scale, edgecolor=c, lw = lw))
    return images;

def join_header_images(images):
    """
        Takes in a list of pillow images returned by draw_header_images() and joins them into a single image with all conjugacy classes for a given point group
        returns the header image (bg) and a list of coordinates used to draw the white underline during the animation
        """
    dx = 2;
    dimensions = np.array([[i.width, i.height] for i in images]);
    H = dimensions[:, 1].max();
    W = dimensions[:, 0].sum()+len(dimensions)*dx + dx;
    bg = Image.new("RGBA", (W, int(H*1.2))); #The 1.2 ensures that H can accomodate the small rectangle underneath
    x = 0;
    pixels = []; #Pixels that the little underline must connect
    for i, d in enumerate(dimensions):
        bg.paste(images[i], (x, int((H-d[1])/2)), images[i]);
        if i%2 == 0:
            pixels.append([x, H, x+images[i].width+dx*2, int(H*1.06)])
        x += d[0]+dx;
        #We return the pixels AFTER the point group name
    return (bg, pixels[1:]); #bg is the picture of all the images and pixels are the top left and bottom right coordinates of the rectangles that go underneath

def interpolate_pair(pp1, pp2, n = 60):
    return np.linspace(pp1, pp2, n).astype(int);

def draw_underline(bg, pp, width=1): #Note that this does not modify the original picture
    """
        Draws a underline onto a copy of bg using the coordinates specified by pp and returns this copy
        width: int controling the width of this underline"""
    bg = bg.copy();
    draw = ImageDraw.Draw(bg);
    draw.rounded_rectangle(tuple(pp), fill = "#f7f0dd", width = width, radius = int((pp[3]-pp[1])/2), outline = None)
    return bg;

def svg_to_PIL(svg):
    """
        Converts SVG file with path given as svg onto pillow image"""
    buff = io.BytesIO()
    cairosvg.svg2png(url=svg, write_to=buff, scale = 2)
    im = Image.open(buff)
    return im

if __name__ == "__main__":
    #You can use the function below to take svg logos and make them png
    #svg_to_PIL("japie.svg").save("japie.png")
    pg = PointGroup("D6h")
    images = draw_header_images("D6h")
    image = join_header_images(images)[0]
    image.save("header","png")

