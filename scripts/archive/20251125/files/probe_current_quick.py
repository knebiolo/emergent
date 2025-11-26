"""Quick probe that lists candidate S3 URLs for a given port and checks existence for today only.
Usage:
    python scripts\probe_current_quick.py --port "New York"
"""

import argparse
from datetime import datetime
from emergent.ship_abm import ofs_loader

import time

parser = argparse.ArgumentParser()
parser.add_argument('--port', '-p', default='Baltimore')
args = parser.parse_args()
port = args.port

print(f"[probe_current_quick] Port: {port}")

from emergent.ship_abm.config import OFS_MODEL_MAP
model = OFS_MODEL_MAP.get(port, 'rtofs')
print(f"[probe_current_quick] Model: {model}")

from datetime import date
urls = ofs_loader.candidate_urls(model, date.today())
print(f"[probe_current_quick] {len(urls)} candidate URLs:\n")
for u in urls:
    print(u)

print('\nChecking existence for first 10 URLs (this will hit S3):')
from emergent.ship_abm.ofs_loader import first_existing_url
print(first_existing_url(urls[:10]))
