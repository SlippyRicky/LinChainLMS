import os
import glob
import re

script_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.abspath(os.path.join(script_dir, "../figures_generation/ch2_linear"))
for file in glob.glob(os.path.join(scripts_dir, "*.py")):
    if file.endswith("fix_data_dir.py"): continue
    
    with open(file, 'r') as f:
        content = f.read()
    
    # Replace data_dir
    content = re.sub(r'fig_dir = os.path.abspath(os.path.join(script_dir, "../output"))
os.makedirs(fig_dir, exist_ok=True)
data_dir = "/Users/emericmellet/Desktop/Emeric/School/UPS/M1 SGM/Stage/ENSTA/LMS/results/data"', 
                     'fig_dir = os.path.abspath(os.path.join(script_dir, "../output"))
os.makedirs(fig_dir, exist_ok=True)
data_dir = os.path.abspath(os.path.join(script_dir, "../data"))', content)
    
    with open(file, 'w') as f:
        f.write(content)

print("data_dir fixed in all scripts.")
