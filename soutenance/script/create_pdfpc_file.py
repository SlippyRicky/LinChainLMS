# -*- coding: utf-8 -*-
"""
create_pdfpc_file.py
Parses speaker_notes.txt and generates clean JSON-based .pdfpc sidecar files 
(Soutenance-MELLET.pdfpc and main.pdfpc) with speaker notes converted from 
LaTeX to clean plain text.
"""
import os
import re
import json

# The exact order of pages in the compiled PDF presentation
SLIDE_ORDER = [
    "1", "2", "3", "4", "4bis", "5", "6", "7", "8", "9", "10",
    "11", "12", "13", "14", "15", "15bis", "16", "18", "19", "20",
    "21", "22", "23", "24"
]

def clean_latex(text):
    # Replace LaTeX commands to plain readable text
    text = text.replace(r"\og", "«")
    text = text.replace(r"\fg{}", "»")
    text = text.replace(r"\fg", "»")
    text = text.replace(r"\varepsilon", "epsilon")
    text = text.replace(r"\omega", "omega")
    text = text.replace(r"\xi", "xi")
    text = text.replace(r"\gamma", "gamma")
    text = text.replace(r"\tilde", "")
    text = text.replace(r"\propto", "proportional to")
    # Strip LaTeX $ math brackets
    text = re.sub(r'\$(.*?)\$', r'\1', text)
    # Strip any backslashes remaining before keywords
    text = re.sub(r'\\([a-zA-Z]+)', r'\1', text)
    # Strip multiple spaces
    text = re.sub(r' +', ' ', text)
    return text.strip()

def parse_speaker_notes(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Speaker notes file not found at: {filepath}")
        
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Split content by [Slide ...] tags
    pattern = re.compile(r'\[Slide\s+([\w\d]+)\]', re.IGNORECASE)
    parts = pattern.split(content)
    
    slide_notes = {}
    for i in range(1, len(parts), 2):
        slide_id = parts[i].strip()
        note_content = parts[i+1].strip()
        slide_notes[slide_id] = note_content
        
    return slide_notes

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    soutenance_dir = os.path.abspath(os.path.join(script_dir, ".."))
    notes_file_path = os.path.join(script_dir, "speaker_notes.txt")
    
    print(f"Reading speaker notes from: {notes_file_path}")
    slide_notes = parse_speaker_notes(notes_file_path)
    
    # We build the JSON content
    pdfpc_data = {
        "pdfpcFormat": 2,
        "pages": []
    }
    
    for idx, slide_id in enumerate(SLIDE_ORDER):
        note = slide_notes.get(slide_id, "")
        if not note:
            print(f"Warning: No note found for slide '{slide_id}' in speaker_notes.txt!")
            
        cleaned = clean_latex(note)
        # pdfpc uses 0-based page indexes internally for the pages array
        pdfpc_data["pages"].append({
            "idx": idx,
            "label": slide_id,
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
        
    print(f"Generated sidecar files:\n  {main_pdfpc}\n  {soutenance_pdfpc}")

if __name__ == "__main__":
    main()
