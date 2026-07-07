# -*- coding: utf-8 -*-
"""
embed_notes.py
Parses speaker_notes.txt and programmatically embeds speaker notes into main.tex 
using Beamer's \note{...} command.
"""
import os
import re

# The exact order of pages in the compiled PDF presentation
SLIDE_ORDER = [
    "1", "2", "3", "4", "4bis", "5", "6", "7", "8", "9", "10",
    "11", "12", "13", "14", "15", "15bis", "16", "18", "19", "20",
    "21", "22", "23", "24"
]

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

def remove_notes(content):
    while True:
        idx = content.find(r"\note{")
        if idx == -1:
            break
        # Find the matching closing brace
        brace_count = 1
        pos = idx + len(r"\note{")
        while pos < len(content) and brace_count > 0:
            if content[pos] == "{":
                brace_count += 1
            elif content[pos] == "}":
                brace_count -= 1
            pos += 1
        if brace_count == 0:
            end_pos = pos
            if end_pos < len(content) and content[end_pos] == "\n":
                end_pos += 1
            content = content[:idx] + content[end_pos:]
        else:
            break
    return content

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    soutenance_dir = os.path.abspath(os.path.join(script_dir, ".."))
    main_tex_path = os.path.join(soutenance_dir, "main.tex")
    notes_file_path = os.path.join(script_dir, "speaker_notes.txt")
    
    print(f"Reading speaker notes from: {notes_file_path}")
    slide_notes = parse_speaker_notes(notes_file_path)
    
    with open(main_tex_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Remove any existing \note{...} tags to avoid duplication
    content = remove_notes(content)
    
    # Find all active \begin{frame} ... \end{frame} environments
    frame_indices = []
    for match in re.finditer(r'\\begin\{frame\}', content):
        start_idx = match.start()
        # Check if this frame is commented out
        line_start = content.rfind('\n', 0, start_idx) + 1
        line_content = content[line_start:start_idx]
        if '%' in line_content:
            continue
            
        # Find corresponding \end{frame}
        end_match = re.search(r'\\end\{frame\}', content[start_idx:])
        if end_match:
            end_idx = start_idx + end_match.start()
            frame_indices.append((start_idx, end_idx))
            
    if len(frame_indices) != len(SLIDE_ORDER):
        print(f"Warning: Found {len(frame_indices)} active frames, but have {len(SLIDE_ORDER)} slide IDs!")
        
    # We rebuild the content by inserting notes before \end{frame}
    new_content = ""
    last_idx = 0
    for idx, (start_idx, end_idx) in enumerate(frame_indices):
        # Copy up to the frame end
        new_content += content[last_idx:end_idx]
        
        # Get slide id and note
        slide_id = SLIDE_ORDER[idx]
        note_text = slide_notes.get(slide_id, "").strip()
        
        # Clean note text for Beamer \note command:
        # 1. Replace multiple newlines with \par
        note_text = re.sub(r'\n\s*\n', r' \\par ', note_text)
        # 2. Replace single newlines with spaces
        note_text = note_text.replace('\n', ' ')
        # 3. Collapse multiple spaces
        note_text = re.sub(r'\s+', ' ', note_text).strip()
        
        # Insert the Beamer \note{...}
        if note_text:
            new_content += f"\n  \\note{{{note_text}}}\n"
        last_idx = end_idx
        
    new_content += content[last_idx:]
    
    with open(main_tex_path, "w", encoding="utf-8") as f:
        f.write(new_content)
        
    print("Speaker notes embedded into main.tex successfully!")

if __name__ == "__main__":
    main()
