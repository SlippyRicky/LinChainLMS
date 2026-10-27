import os
import re

HELPERS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(HELPERS_DIR, "../.."))
SCRIPTS_DIR = os.path.join(ROOT_DIR, "figures/02_linear_disorder/scripts")
LATEX_PATH = os.path.join(ROOT_DIR, "content/chapters/02_linear_disorder.tex")

def fix():
    # 1. Read actual scripts to build a suffix -> index mapping
    script_files = [f for f in os.listdir(SCRIPTS_DIR) if f.startswith("report_ch1_") and f.endswith(".py")]
    
    suffix_map = {}
    for f in script_files:
        match = re.match(r"report_ch1_(\d\d)_(.*)\.py", f)
        if match:
            idx = match.group(1)
            suffix = match.group(2)
            suffix_map[suffix] = idx
            
    print("Built suffix mapping:")
    for suf, idx in sorted(suffix_map.items()):
        print(f"  {suf} -> {idx}")
        
    # 2. Read LaTeX file
    with open(LATEX_PATH, 'r') as file:
        content = file.read()
        
    # 3. Find and replace each figure name
    # We will search for (report|linear)_\d\d_suffix and replace with (report|linear)_idx_suffix
    original_content = content
    for suffix, idx in suffix_map.items():
        # Match report_XX_suffix and linear_XX_suffix
        pattern_report = r"report_\d\d_" + re.escape(suffix)
        pattern_linear = r"linear_\d\d_" + re.escape(suffix)
        
        content = re.sub(pattern_report, f"report_{idx}_{suffix}", content)
        content = re.sub(pattern_linear, f"linear_{idx}_{suffix}", content)
        
    if content != original_content:
        with open(LATEX_PATH, 'w') as file:
            file.write(content)
        print("\n[SUCCESS] Rebuilt 02_linear_disorder.tex with correct figure indices!")
    else:
        print("\nNo changes made to 02_linear_disorder.tex.")

if __name__ == "__main__":
    fix()
