import re

with open('main.tex', 'r') as f:
    content = f.read()

# Extract all \note{...} blocks (handling nested braces)
notes = []
pattern = r'\\note\{'
for m in re.finditer(pattern, content):
    start = m.end()
    depth = 1
    i = start
    while i < len(content) and depth > 0:
        if content[i] == '{':
            depth += 1
        elif content[i] == '}':
            depth -= 1
        i += 1
    note_text = content[start:i-1]
    # Find which slide this belongs to
    preceding = content[max(0, m.start()-500):m.start()]
    slide_match = re.findall(r'% --- Slide (\S+?):', preceding)
    slide_label = slide_match[-1] if slide_match else '?'
    notes.append((slide_label, note_text))

total_words = 0
slide_data = []

for label, text in notes:
    clean = re.sub(r'\\[a-zA-Z]+', '', text)
    clean = re.sub(r'[{}\\$]', '', clean)
    words = len(clean.split())
    total_words += words
    minutes = words / 130.0
    preview = text.replace('\n', ' ')[:60]
    slide_data.append((label, words, minutes, preview))

print(f"{'Slide':<8} {'Words':>6}  {'Time':>8}  Preview")
print("-" * 90)
for label, words, minutes, preview in slide_data:
    flag = " !!!" if minutes > 1.2 else ""
    print(f"{label:<8} {words:>6}  {minutes:>5.1f} min  {preview}{flag}")

print("-" * 90)
print(f"{'TOTAL':<8} {total_words:>6}  {total_words/130:.1f} min")
print()
print(f"At 130 wpm (slow French pace): {total_words/130:.1f} min")
print(f"At 120 wpm (very slow):        {total_words/120:.1f} min")
print(f"At 140 wpm (moderate):         {total_words/140:.1f} min")
print(f"Target: 15.0 min max")
print(f"Budget at 130 wpm: {15*130} words")
print(f"Surplus/deficit: {15*130 - total_words:+d} words")

# Flag slides that need trimming
over = [(l, w, m) for l, w, m, _ in slide_data if m > 1.2]
if over:
    print()
    print("=== SLIDES OVER 1.2 MIN (candidates for trimming) ===")
    for l, w, m in over:
        trim = w - int(1.0 * 130)
        print(f"  Slide {l}: {w} words ({m:.1f} min) -> trim ~{trim} words")
