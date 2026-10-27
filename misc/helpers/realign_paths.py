import os
import glob
import re

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, "../.."))

# Directory mappings
mappings = {
    os.path.join(root_dir, "figures/02_linear_disorder/scripts"): "figures/02_linear_disorder",
}

def realign_paths():
    print("=========================================================")
    print("   AUTOMATIC PATH REALIGNMENT & DIRECTORY CONSOLIDATION  ")
    print("=========================================================")
    
    for folder, target_subpath in mappings.items():
        if not os.path.exists(folder):
            print(f"Skipping non-existent folder: {folder}")
            continue
            
        print(f"\nProcessing scripts in: {os.path.basename(folder)} -> Targeting LaTeX {target_subpath}")
        target_path_expr = 'os.path.abspath(os.path.join(script_dir, ".."))'
        
        for file_path in glob.glob(os.path.join(folder, "*.py")):
            # Skip path fixing scripts themselves
            if os.path.basename(file_path) in ("fix_data_dir.py", "fix_fig_dir.py", "change_fig_dir.py"):
                continue
                
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            original_content = content
            
            # 1. Update fig_dir definition
            pattern_fig = r'fig_dir\s*=\s*os\.path\.abspath\(os\.path\.join\(script_dir,\s*[\'"][^\'"]+[\'"]\)\)'
            if re.search(pattern_fig, content):
                content = re.sub(pattern_fig, f'fig_dir = {target_path_expr}', content)
            else:
                pattern_fig_simple = r'fig_dir\s*=\s*os\.path\.join\(script_dir,\s*[\'"][^\'"]+[\'"]\)'
                if re.search(pattern_fig_simple, content):
                    content = re.sub(pattern_fig_simple, f'fig_dir = {target_path_expr}', content)
            
            # 2. Update figures_dir definition (used in standalone scripts)
            pattern_figs_dir = r'figures_dir\s*=\s*os\.path\.dirname\(script_dir\)'
            if re.search(pattern_figs_dir, content):
                content = re.sub(pattern_figs_dir, 'figures_dir = os.path.abspath(os.path.join(script_dir, ".."))', content)
            
            # 3. Update data_dir definition
            # Ensure it points to code/figures_generation/data/ (which is relative "../../../code/figures_generation/data" from new script loc)
            pattern_data_rel = r'data_dir\s*=\s*os\.path\.abspath\(os\.path\.join\(script_dir,\s*[\'"](?:\.\./)*data[\'"]\)\)'
            content = re.sub(pattern_data_rel, 'data_dir = os.path.abspath(os.path.join(script_dir, "../../../code/figures_generation/data"))', content)
            
            pattern_data_abs = r'data_dir\s*=\s*[\'"]/Users/emericmellet/Desktop/Emeric/School/UPS/M1 SGM/Stage/ENSTA/LMS/results/data[\'"]'
            content = re.sub(pattern_data_abs, 'data_dir = os.path.abspath(os.path.join(script_dir, "../../../code/figures_generation/data"))', content)
            
            # If the file has changed, save it
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"  ✓ Realigned paths in: {os.path.basename(file_path)}")
                
    print("\n=========================================================")
    print("   PATH REALIGNMENT COMPLETE SUCCESSFULLY!               ")
    print("=========================================================")

if __name__ == "__main__":
    realign_paths()
