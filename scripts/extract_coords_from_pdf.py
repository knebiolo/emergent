import re
from pathlib import Path
from PyPDF2 import PdfReader

pdf_path = Path(r"C:\Users\Kevin.Nebiolo\Downloads\DCA24MM031_PreliminaryReport 3.pdf")
if not pdf_path.exists():
    print('PDF not found:', pdf_path)
    raise SystemExit(1)

text = []
reader = PdfReader(str(pdf_path))
for p in reader.pages:
    try:
        t = p.extract_text()
        if t:
            text.append(t)
    except Exception as e:
        print('Page read error', e)

full = '\n'.join(text)
print('Extracted text length:', len(full))

# Look for DMS patterns like 39°12'56"N 76°31'47"W or decimal patterns like 39.2155, -76.5297
dms_re = re.compile(r"(\d{1,2})[°\s]\s*(\d{1,2})[\'\s]\s*(\d{1,2}(?:\.\d+)?)\"?\s*([NS])[,\s]+(\d{1,3})[°\s]\s*(\d{1,2})[\'\s]\s*(\d{1,2}(?:\.\d+)?)\"?\s*([EW])")
dec_re = re.compile(r"([+-]?\d{1,3}\.\d+)[,\s]+([+-]?\d{1,3}\.\d+)")

found = []
for m in dms_re.finditer(full):
    found.append(('dms', m.group(0), m.groups()))
for m in dec_re.finditer(full):
    # filter improbable lat/lon (lon ~ -180..180, lat ~ -90..90)
    lat = float(m.group(1))
    lon = float(m.group(2))
    if -90 <= lat <= 90 and -180 <= lon <= 180:
        found.append(('dec', m.group(0), (lat, lon)))

print('Found coordinates:', found[:20])

# Save found snippets for manual review
out = Path('outputs/dali_scenario/pdf_coords_snippets.txt')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text('\n'.join([f'{t}: {s}' for t, s, _ in found]))
print('Snippets written to', out)
