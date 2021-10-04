import json
import os

dirs = [o for o in os.listdir(os.getcwd()) if o.endswith(".json")]
print(len(dirs))
for d in dirs:
    with open(d, 'r') as f:
        data = json.load(f)

    with open("names.txt", "a") as f:
        f.write(data["name"] +" "+data["point_group"]+"\n")
   
        
