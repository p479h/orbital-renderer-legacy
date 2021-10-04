from PointGroup import PointGroup
import os
import numpy as np


d = "pointgroup_data"
pgs = [o.replace(".txt", "") for o in os.listdir(d)]

tt = 59
sp = 30
tt2 = 20
p = tt + sp

labels = {k:i for i,k in enumerate("E C i S g".split())}; #g is in 'sig'
shifts = [p, p*3, p*2, p*5, p*3]

with open("spans_second.txt", "w") as f:
    for pg in pgs:
        print(); print()
        print(pg)
        pg = PointGroup(pg)
        limits = [];
        frame = 0;
        for cc in pg.conjugacy_classes:
            for label in labels:
                if label in cc.replace("sig", "g"):
                    print(label)
                    start = frame + 1 #When the animation starts
                    duration = shifts[labels[label]] #Length of transitions in the symmetry operation
                    extra_duration = 2*tt2
                    frame += duration + extra_duration
                    limits.append([start, frame]) #Add first and last frame of the current animation (inclusive)

    
        f.write(pg.pg_name)
        f.write("\n")
        f.write(str(pg.conjugacy_classes))
        f.write("\n")
        f.write(str(np.around((np.array(limits)/30), 1).tolist()))
        #f.write(str(np.around((np.array(limits)/60), 1).tolist()))
        f.write("\n\n\n")
        
                
