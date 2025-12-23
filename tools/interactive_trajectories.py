"""Generate an interactive HTML plot of trajectories using Plotly.

Saves: outputs/trajectories_interactive.html
"""
import os
import sys
import h5py
import numpy as np
import plotly.graph_objects as go


def generate(db_path, out_path=None, max_agents=200):
    if out_path is None:
        out_dir = os.path.dirname(db_path)
        out_path = os.path.join(out_dir, 'trajectories_interactive.html')
    with h5py.File(db_path, 'r') as f:
        depth = f['environment/depth'][()]
        x_coords = f['environment/x_coords'][()]
        y_coords = f['environment/y_coords'][()]
        X = f['agent_data/X'][()]
        Y = f['agent_data/Y'][()]

    n_agents = X.shape[0]
    n_steps = X.shape[1]
    # downsample agents if too many (interactive rendering limits)
    agent_idxs = np.arange(n_agents)
    if n_agents > max_agents:
        np.random.seed(0)
        agent_idxs = np.random.choice(n_agents, max_agents, replace=False)

    # create figure
    fig = go.Figure()

    # add depth as heatmap (use origin='lower' mapping via y reversed)
    z = depth
    # compute extents
    x_min, x_max = float(np.nanmin(x_coords)), float(np.nanmax(x_coords))
    y_min, y_max = float(np.nanmin(y_coords)), float(np.nanmax(y_coords))
    # flip y for correct orientation
    fig.add_trace(go.Heatmap(z=np.flipud(z), x=np.linspace(x_min, x_max, z.shape[1]), y=np.linspace(y_min, y_max, z.shape[0]), colorscale='Blues', showscale=False, opacity=0.8))

    # add agent trajectories as scattergl traces
    for ai in agent_idxs:
        xs = X[ai, :]
        ys = Y[ai, :]
        valid = np.isfinite(xs) & np.isfinite(ys)
        if np.count_nonzero(valid) < 2:
            continue
        fig.add_trace(go.Scattergl(x=xs[valid], y=ys[valid], mode='lines', line=dict(color='red', width=1), name=f'agent_{ai}', hoverinfo='none'))
        # start marker
        sidx = np.where(valid)[0][0]
        fig.add_trace(go.Scattergl(x=[xs[sidx]], y=[ys[sidx]], mode='markers', marker=dict(color='red', size=6), showlegend=False, hoverinfo='none'))

    fig.update_layout(width=1200, height=900, template='plotly_white', title='Interactive trajectories over depth', xaxis_title='Easting', yaxis_title='Northing')
    fig.write_html(out_path, include_plotlyjs='cdn')
    print('Wrote interactive plot to', out_path)
    return out_path


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python tools/interactive_trajectories.py /path/to/sim_db.h5')
        sys.exit(2)
    db = sys.argv[1]
    generate(db)
