import os
import re
import sys

# Configuration
HELPERS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(HELPERS_DIR, "../.."))
SCRIPTS_DIR = os.path.join(ROOT_DIR, "figures/02_linear_disorder/scripts")
OUTPUT_DIR = os.path.join(ROOT_DIR, "figures/02_linear_disorder")
LATEX_CONTENT_DIR = os.path.join(ROOT_DIR, "content")

def get_scripts():
    files = [f for f in os.listdir(SCRIPTS_DIR) if f.startswith("report_ch1_") and f.endswith(".py")]
    files.sort()
    return files

def sync():
    scripts = get_scripts()
    print(f"Found {len(scripts)} report scripts.")
    
    # 1. Identify current numbering and map old names to new names
    renames = []
    current_index = 0
    for script_name in scripts:
        match = re.match(r"report_ch1_(\d\d)_(.*)\.py", script_name)
        if match:
            num = int(match.group(1))
            suffix = match.group(2)
            
            if num != current_index:
                old_script_prefix = f"report_ch1_{num:02d}"
                new_script_prefix = f"report_ch1_{current_index:02d}"
                
                # Check for PDF output filename prefixes in script contents (both 'report_' and 'linear_')
                # PDFs are either report_XX or linear_XX
                old_pdf_report_prefix = f"report_{num:02d}"
                new_pdf_report_prefix = f"report_{current_index:02d}"
                
                old_pdf_linear_prefix = f"linear_{num:02d}"
                new_pdf_linear_prefix = f"linear_{current_index:02d}"
                
                renames.append((
                    old_script_prefix, new_script_prefix, 
                    old_pdf_report_prefix, new_pdf_report_prefix,
                    old_pdf_linear_prefix, new_pdf_linear_prefix,
                    suffix
                ))
            current_index += 1

    if not renames:
        print("All figures are already sequentially numbered. Nothing to do.")
    else:
        print(f"Detected {len(renames)} renumbering operations needed.")
        
        # 2. Apply renames to scripts and output PDFs (reversed order to prevent naming conflicts)
        for old_sp, new_sp, old_rp, new_rp, old_lp, new_lp, suffix in reversed(renames):
            old_script = f"{old_sp}_{suffix}.py"
            new_script = f"{new_sp}_{suffix}.py"
            
            old_path = os.path.join(SCRIPTS_DIR, old_script)
            new_path = os.path.join(SCRIPTS_DIR, new_script)
            
            # Update script content
            if os.path.exists(old_path):
                with open(old_path, 'r') as f:
                    content = f.read()
                
                # Replace script prefixes, report PDF prefixes, and linear PDF prefixes
                new_content = content.replace(old_sp, new_sp)
                new_content = new_content.replace(old_rp, new_rp)
                new_content = new_content.replace(old_lp, new_lp)
                
                with open(old_path, 'w') as f:
                    f.write(new_content)
                
                # Rename script file
                os.rename(old_path, new_path)
                print(f"  Renamed script: {old_script} -> {new_script}")
                
            # Rename PDFs (check for both report_ and linear_)
            for old_p, new_p in [(old_rp, new_rp), (old_lp, new_lp)]:
                old_pdf = f"{old_p}_{suffix}.pdf"
                new_pdf = f"{new_p}_{suffix}.pdf"
                old_pdf_path = os.path.join(OUTPUT_DIR, old_pdf)
                new_pdf_path = os.path.join(OUTPUT_DIR, new_pdf)
                
                if os.path.exists(old_pdf_path):
                    os.rename(old_pdf_path, new_pdf_path)
                    print(f"  Renamed PDF: {old_pdf} -> {new_pdf}")

        for root, dirs, files in os.walk(LATEX_CONTENT_DIR):
            for f in files:
                if f.endswith(".tex"):
                    tex_path = os.path.join(root, f)
                    with open(tex_path, 'r') as file:
                        t_content = file.read()
                    
                    original_t = t_content
                    for old_sp, new_sp, old_rp, new_rp, old_lp, new_lp, suffix in renames:
                        t_content = t_content.replace(f"{old_rp}_{suffix}", f"{new_rp}_{suffix}")
                        t_content = t_content.replace(f"{old_lp}_{suffix}", f"{new_lp}_{suffix}")
                    
                    if t_content != original_t:
                        with open(tex_path, 'w') as file:
                            file.write(t_content)
                        print(f"Updated LaTeX file: {f}")

    # 4. Regenerate ch1_plot_all.py's scripts list dynamically from all existing report_ch1_*.py scripts
    ch1_plot_all_path = os.path.join(SCRIPTS_DIR, "ch1_plot_all.py")
    if os.path.exists(ch1_plot_all_path):
        with open(ch1_plot_all_path, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        in_scripts_list = False
        replaced = False
        for line in lines:
            if "scripts = [" in line and not replaced:
                new_lines.append("scripts = [\n")
                new_lines.append("    # Main Report Figures (Chronological)\n")
                # Get the newly renamed list of scripts
                for s in get_scripts():
                    new_lines.append(f"    '{s}',\n")
                in_scripts_list = True
                replaced = True
                continue
            if in_scripts_list:
                if "]" in line:
                    new_lines.append("]\n")
                    in_scripts_list = False
                continue
            new_lines.append(line)
            
        with open(ch1_plot_all_path, 'w') as f:
            f.writelines(new_lines)
        print("Regenerated scripts list in ch1_plot_all.py dynamically!")

if __name__ == "__main__":
    sync()
