import os
import sys
import time
import json
from matplotlib.backends.backend_pdf import PdfPages
import os
import sys
import time
import json
from matplotlib.backends.backend_pdf import PdfPages

# import local tools using package-style relative imports
from . import plot_trajectories
from .schooling_metrics import compute_metrics

# Lightweight wrapper to assemble report

def generate_report(db_path, out_dir=None, title='Simulation Report'):
    if out_dir is None:
        out_dir = os.path.dirname(db_path)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # compute schooling metrics
    summary_path, debug_path = compute_metrics(db_path, out_dir)

    # generate trajectory image
    img_path = os.path.join(out_dir, 'trajectories_report.png')
    plot_trajectories.plot_db(db_path, img_path)

    # assemble PDF
    pdf_path = os.path.join(out_dir, f'report_{int(time.time())}.pdf')
    with PdfPages(pdf_path) as pdf:
        # add trajectory page
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 8))
        img = plt.imread(img_path)
        ax.imshow(img)
        ax.axis('off')
        pdf.savefig(fig)
        plt.close(fig)

        # add summary JSON as text page
        with open(summary_path, 'r') as sf:
            summary = json.load(sf)
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        txt = json.dumps(summary, indent=2)
        ax.text(0.01, 0.99, txt, va='top', family='monospace', fontsize=6)
        pdf.savefig(fig)
        plt.close(fig)

    print('Wrote report to', pdf_path)
    return pdf_path


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python tools/generate_report.py /path/to/sim_db.h5')
        sys.exit(2)
    db = sys.argv[1]
    generate_report(db)
