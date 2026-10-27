import os
import glob
import re

script_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.abspath(os.path.join(script_dir, "../figures_generation/ch2_linear"))
for file in glob.glob(os.path.join(scripts_dir, "*.py")):
    if file.endswith("change_fig_dir.py"): continue
    
    with open(file, 'r') as f:
        content = f.read()
    
    # Replace fig_dir
    content = content.replace('fig_dir = os.path.abspath(os.path.join(script_dir, ".."))', 
                              'fig_dir = os.path.abspath(os.path.join(script_dir, "../output"))')
    
    # Create the dir if it doesn't exist
    if "os.makedirs(fig_dir, exist_ok=True)" not in content and "fig_dir =" in content:
        content = content.replace("fig_dir = os.path.abspath(os.path.join(script_dir, \"../output\"))", 
                                  "fig_dir = os.path.abspath(os.path.join(script_dir, \"../output\"))\nos.makedirs(fig_dir, exist_ok=True)")
    
    with open(file, 'w') as f:
        f.write(content)

print("fig_dir updated in all scripts.")
