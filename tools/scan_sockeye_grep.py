from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOCKEYE = ROOT / 'src' / 'emergent' / 'salmon_abm' / 'sockeye.py'
OUT_DIR = ROOT / '.ai_journal' / 'session'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def grep_symbols(path):
    text = path.read_text(encoding='utf-8')
    symbols = []
    for m in re.finditer(r'^def\s+(\w+)\s*\(|^class\s+(\w+)\s*[:\(]', text, flags=re.MULTILINE):
        name = m.group(1) or m.group(2)
        typ = 'function' if m.group(1) else 'class'
        symbols.append((typ, name))
    return symbols


def count_usage(root, symbol):
    pat = re.compile(r'\b' + re.escape(symbol) + r'\b')
    c = 0
    for p in root.rglob('*.py'):
        try:
            t = p.read_text(encoding='utf-8')
        except Exception:
            continue
        c += len(pat.findall(t))
    return c


def main():
    syms = grep_symbols(SOCKEYE)
    csvp = OUT_DIR / 'sockeye_inventory.csv'
    with open(csvp, 'w', encoding='utf-8') as f:
        f.write('type,name,usage_count\n')
        for typ, name in syms:
            u = count_usage(ROOT, name)
            f.write(f'{typ},{name},{u}\n')

    prior = sorted([(name, count_usage(ROOT, name)) for (_, name) in syms], key=lambda x: -x[1])
    outp = OUT_DIR / 'sockeye_usage_prioritized.txt'
    with open(outp, 'w', encoding='utf-8') as f:
        for name, usage in prior:
            f.write(f'{usage:5d}  {name}\n')

    print('Wrote', csvp, outp)


if __name__ == '__main__':
    main()
