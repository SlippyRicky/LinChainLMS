import os
import glob
import re

script_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.abspath(os.path.join(script_dir, "../figures_generation/ch2_linear"))
for file in glob.glob(os.path.join(scripts_dir, "*.py")):
    if file.endswith("fix_fig_dir.py"): continue
    
    with open(file, 'r') as f:
        content = f.read()
    
    if "fig_dir =" not in content:
        content = content.replace("data_dir =", "fig_dir = os.path.abspath(os.path.join(script_dir, \"..\"))\ndata_dir =")
    
    with open(file, 'w') as f:
        f.write(content)

print("fig_dir fixed in all scripts.")
