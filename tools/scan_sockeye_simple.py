import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOCKEYE = ROOT / 'src' / 'emergent' / 'salmon_abm' / 'sockeye.py'
OUT_DIR = ROOT / '.ai_journal' / 'session'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_top_level_symbols(path):
    text = path.read_text(encoding='utf-8')
    lines = text.splitlines()
    symbols = []
    pattern_def = re.compile(r'^def\s+(\w+)\s*\(')
    pattern_class = re.compile(r'^class\s+(\w+)\s*[:\(]')
    i = 0
    while i < len(lines):
        line = lines[i]
        m = pattern_def.match(line)
        typ = None
        name = None
        if m:
            typ = 'function'
            name = m.group(1)
        else:
            m2 = pattern_class.match(line)
            if m2:
                typ = 'class'
                name = m2.group(1)

        if name:
            # try to find a docstring in the next few lines
            doc = ''
            j = i + 1
            # skip decorators/blank lines
            while j < len(lines) and lines[j].strip() == '':
                j += 1
            if j < len(lines) and (lines[j].strip().startswith('"""') or lines[j].strip().startswith("'''") ):
                quote = lines[j].strip()[:3]
                doc_lines = []
                # if docstring is single-line
                if lines[j].strip().endswith(quote) and len(lines[j].strip()) > 3:
                    doc = lines[j].strip().strip(quote).strip()
                else:
                    j += 1
                    while j < len(lines) and quote not in lines[j]:
                        doc_lines.append(lines[j])
                        j += 1
                    doc = '\n'.join(doc_lines).strip()
            symbols.append((typ, name, doc))
        i += 1
    return symbols


def count_usage(root, symbol):
    import re
    pattern = re.compile(r'\b' + re.escape(symbol) + r'\b')
    total = 0
    for p in root.rglob('*.py'):
        try:
            txt = p.read_text(encoding='utf-8')
        except Exception:
            continue
        total += len(pattern.findall(txt))
    return total


def main():
    symbols = find_top_level_symbols(SOCKEYE)
    csvp = OUT_DIR / 'sockeye_inventory.csv'
    with open(csvp, 'w', encoding='utf-8') as f:
        f.write('type,name,docstring,usage_count\n')
        for typ, name, doc in symbols:
            usage = count_usage(ROOT, name)
            # escape newlines in doc
            doc_esc = '"' + doc.replace('"', '""').replace('\n', '\\n') + '"'
            f.write(f'{typ},{name},{doc_esc},{usage}\n')

    prior = sorted([(name, count_usage(ROOT, name)) for (_, name, _) in symbols], key=lambda x: -x[1])
    outp = OUT_DIR / 'sockeye_usage_prioritized.txt'
    with open(outp, 'w', encoding='utf-8') as f:
        for name, usage in prior:
            f.write(f'{usage:5d}  {name}\n')

    print('Wrote', csvp, 'and', outp)


if __name__ == '__main__':
    main()
