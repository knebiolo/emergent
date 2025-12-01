from pathlib import Path
path = Path(r'C:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\src\emergent\salmon_abm\sockeye_SoA.py')
text = path.read_text(encoding='utf-8')
dd = text.count('"""')
ds = text.count("'''")
out_lines = []
out_lines.append(f'file: {path}')
out_lines.append(f'length: {len(text)}')
out_lines.append(f'triple double quotes ("""): {dd}')
out_lines.append(f"triple single quotes (' ''): {ds}")
out_lines.append(f'total triple quotes: {dd+ds}')

def show_context(idx, radius=120):
    start = max(0, idx-radius)
    end = min(len(text), idx+radius)
    snippet = text[start:end]
    print('\n--- context around {} ---'.format(idx))
    print(snippet)
    print('--- end context ---\n')

# find all indices
indices_dd = [i for i in range(len(text)) if text.startswith('"""', i)]
indices_ds = [i for i in range(len(text)) if text.startswith("'''", i)]
print('first """ at', indices_dd[0] if indices_dd else None)
print('last """ at', indices_dd[-1] if indices_dd else None)
print("first ''' at", indices_ds[0] if indices_ds else None)
print("last ''' at", indices_ds[-1] if indices_ds else None)

for pos in (indices_dd[:3] + indices_dd[-3:] + indices_ds[:3] + indices_ds[-3:]):
    if pos is not None:
        start = max(0, pos-120)
        end = min(len(text), pos+120)
        out_lines.append('\n--- context around {} ---'.format(pos))
        out_lines.append(text[start:end])
        out_lines.append('--- end context ---\n')

report_path = Path(r'C:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\tools\triple_quote_report.txt')
report_path.write_text('\n'.join(out_lines), encoding='utf-8')
print('Wrote report to', report_path)
