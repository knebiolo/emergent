import ast
import os
from pathlib import Path
import csv
import re


ROOT = Path(__file__).resolve().parents[1]
SOCKEYE = ROOT / 'src' / 'emergent' / 'salmon_abm' / 'sockeye.py'
OUT_DIR = ROOT / '.ai_journal' / 'session'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_symbols(path):
    src = path.read_text(encoding='utf-8')
    tree = ast.parse(src)
    rows = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            name = node.name
            doc = ast.get_docstring(node) or ''
            rows.append(('function', name, doc.strip()))
        elif isinstance(node, ast.ClassDef):
            name = node.name
            doc = ast.get_docstring(node) or ''
            rows.append(('class', name, doc.strip()))
    return rows


def count_usage(root, symbol):
    pattern = re.compile(r'\b' + re.escape(symbol) + r'\b')
    count = 0
    for p in root.rglob('*.py'):
        try:
            text = p.read_text(encoding='utf-8')
        except Exception:
            continue
        count += len(pattern.findall(text))
    return count


def main():
    try:
        rows = extract_symbols(SOCKEYE)
    except Exception as e:
        LOG = OUT_DIR / 'scan_sockeye_error.txt'
        LOG.write_text(f'Failed to parse {SOCKEYE}: {e}\n')
        print('Parsing failed, wrote', LOG)
        return

    csv_path = OUT_DIR / 'sockeye_inventory.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['type', 'name', 'docstring', 'usage_count'])
        for typ, name, doc in rows:
            try:
                usage = count_usage(ROOT, name)
            except Exception:
                usage = 0
            writer.writerow([typ, name, doc, usage])

    # Produce prioritized list
    prior_list = []
    for typ, name, doc in rows:
        usage = 0
        try:
            usage = count_usage(ROOT, name)
        except Exception:
            usage = 0
        prior_list.append((name, usage))

    prioritized = sorted(prior_list, key=lambda x: -x[1])
    out_prior = OUT_DIR / 'sockeye_usage_prioritized.txt'
    with open(out_prior, 'w', encoding='utf-8') as f:
        for name, usage in prioritized:
            f.write(f'{usage:5d}  {name}\n')

    LOG = OUT_DIR / 'scan_sockeye_ok.txt'
    LOG.write_text(f'Wrote {csv_path} and {out_prior}\nFound {len(rows)} symbols')
    print('Wrote', csv_path, 'and', out_prior)


if __name__ == '__main__':
    main()
