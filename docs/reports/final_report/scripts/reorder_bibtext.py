#!/usr/bin/python3
import re
import os
from pathlib import Path
from collections import OrderedDict

target_dir = './report/Capstone Report - Manu Hegde/'

# === Step 1: Collect all .tex files in sorted order ===
tex_files = sorted(Path(target_dir).glob('*.tex'))

# === Step 2: Extract citation keys in usage order ===
citation_pattern = re.compile(r'\\cite\{([^}]+)\}')
used_citations = []

for tex_file in tex_files:
    with open(tex_file, 'r', encoding='utf-8') as f:
        content = f.read()
        for match in citation_pattern.findall(content):
            keys = [key.strip() for key in match.split(',')]
            used_citations.extend(keys)

# === Step 3: Deduplicate while preserving order ===
seen = set()
ordered_citations = []
for key in used_citations:
    if key not in seen:
        seen.add(key)
        ordered_citations.append(key)

print("Ordered citations:\n",ordered_citations)

# === Step 4: Read and index all entries in uwthesis.bib ===
bib_file_path = target_dir+'uwthesis.bib'
with open(bib_file_path, 'r', encoding='utf-8') as bib_file:
    bib_content = bib_file.read()

# Match each full BibTeX entry
entry_pattern = re.compile(r'@(\w+)\{([^,]+),.*?\n\}', re.DOTALL)
entries = {}
for match in re.finditer(r'(@\w+\{[^@]*?\n\})', bib_content, re.DOTALL):
    entry = match.group(1)
    key_match = re.match(r'@\w+\{([^,]+),', entry)
    if key_match:
        key = key_match.group(1).strip()
        entries[key] = entry.strip()

# === Step 5: Write sorted bib file ===
cited = 0
uncited = 0
with open(bib_file_path, 'w', encoding='utf-8') as out:
    for key in ordered_citations:
        if key in entries:
            cited+=1
            out.write(entries[key] + '\n\n')
    # Optionally: Add unused entries at the end
    unused = [k for k in entries if k not in ordered_citations]
    for key in unused:
        uncited+=1
        out.write(entries[key] + '\n\n')

print(f"\nDone. Reordered bibliography written to {bib_file_path}. ")
print(f"\nCitation stats - used:{cited}, unused:{uncited} ")
