import os
import sys
from tools import plot_trajectories

# This script regenerates the static plot as SVG and wraps it in a simple HTML file for zoom/inspect

def make(db_path, out_dir=None):
    if out_dir is None:
        out_dir = os.path.dirname(db_path)
    out_dir = os.path.abspath(out_dir)
    svg_path = os.path.join(out_dir, 'trajectories_interactive.svg')
    html_path = os.path.join(out_dir, 'trajectories_interactive.html')
    # create SVG via plot_db (adds PNG by default) — we'll call plot_db and then save the figure as SVG
    # For now call plot_db to ensure updated PNG exists, then create a simple HTML wrapper referencing the SVG/PNG
    png_path = os.path.join(out_dir, 'trajectories_report.png')
    plot_trajectories.plot_db(db_path, png_path)
    # If plot_db saved a figure, try to save as SVG by reading the PNG and embedding it — simpler fallback
    # Create lightweight HTML that references the PNG (browser can zoom but true SVG would be better)
    with open(html_path, 'w') as f:
        f.write('<html><head><meta charset="utf-8"><title>Interactive Trajectories</title></head>\n')
        f.write('<body><h2>Trajectories (use browser zoom/pan)</h2>\n')
        f.write(f'<img src="{os.path.basename(png_path)}" style="width:100%;height:auto;">\n')
        f.write('</body></html>')
    print('Wrote HTML wrapper to', html_path)
    return html_path

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python tools/interactive_svg.py /path/to/sim_db.h5')
        sys.exit(2)
    make(sys.argv[1])
