import os
import re
import json

def clean_latex(text):
    # Replace LaTeX commands to plain readable text
    text = text.replace(r"\og", "«")
    text = text.replace(r"\fg{}", "»")
    text = text.replace(r"\fg", "»")
    text = text.replace(r"\protect", "")
    text = text.replace(r"\&", "&")
    # Strip LaTeX math brackets $...$
    text = re.sub(r'\$(.*?)\$', r'\1', text)
    # Strip any backslashes remaining before keywords
    text = re.sub(r'\\([a-zA-Z]+)', r'\1', text)
    # Strip multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    soutenance_dir = os.path.abspath(os.path.join(script_dir, ".."))
    notes_file_path = os.path.join(soutenance_dir, "build", "main.notes")
    
    if not os.path.exists(notes_file_path):
        print(f"Notes file not found at: {notes_file_path}")
        return
        
    print(f"Reading generated notes from: {notes_file_path}")
    
    pdfpc_data = {
        "pdfpcFormat": 2,
        "pages": []
    }
    
    with open(notes_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Parse page_num: note
            match = re.match(r'^(\d+):\s*(.*)$', line)
            if match:
                page_num = int(match.group(1))
                note_content = match.group(2)
                
                cleaned = clean_latex(note_content)
                
                # pdfpc uses 0-based page indexes
                pdfpc_data["pages"].append({
                    "idx": page_num - 1,
                    "note": cleaned
                })
                
    file_content = json.dumps(pdfpc_data, indent=2, ensure_ascii=False)
    
    # Write to main.pdfpc
    main_pdfpc = os.path.join(soutenance_dir, "main.pdfpc")
    with open(main_pdfpc, "w", encoding="utf-8") as f:
        f.write(file_content)
        
    # Write to Soutenance-MELLET.pdfpc
    soutenance_pdfpc = os.path.join(soutenance_dir, "Soutenance-MELLET.pdfpc")
    with open(soutenance_pdfpc, "w", encoding="utf-8") as f:
        f.write(file_content)
        
    print(f"Successfully generated sidecar files:\n  {main_pdfpc}\n  {soutenance_pdfpc}")

if __name__ == "__main__":
    main()
