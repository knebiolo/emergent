import numpy as np
from scipy.ndimage import distance_transform_edt

# Reuse shared helpers from the monolith to preserve semantics and avoid
# duplicate placeholder implementations. These are defined in
# `src/emergent/salmon_abm/sockeye_SoA_OpenGL.py` and provide Affine
# computation, geo <-> pixel transforms, and safe HDF5 flush.
from emergent.salmon_abm.sockeye_SoA_OpenGL import (
    compute_affine_from_hecras,
    geo_to_pixel,
    get_inv_transform,
    safe_flush,
)


def initialize_mental_map(sim):
    mem_data = sim.hdf5.require_group('memory')

    if not hasattr(sim, 'width') or not hasattr(sim, 'height'):
        avoid_height = 256
        avoid_width = 256
    else:
        avoid_height = int(np.round(sim.height / sim.avoid_cell_size)) + 1
        avoid_width = int(np.round(sim.width / sim.avoid_cell_size)) + 1

    for i in range(sim.num_agents):
        name = str(i)
        if name not in mem_data:
            mem_data.create_dataset(name, shape=(avoid_height, avoid_width),
                                    dtype='i2', chunks=(1, min(avoid_width, 4096)))

    if hasattr(sim, 'depth_rast_transform') and sim.depth_rast_transform is not None:
        base_transform = sim.depth_rast_transform
    else:
        base_transform = None
        try:
            if hasattr(sim, '_hecras_maps') and len(sim._hecras_maps) > 0:
                m = next(iter(sim._hecras_maps.values()))
                coords = m.coords
                base_transform = compute_affine_from_hecras(coords, target_cell_size=sim.avoid_cell_size)
                if not hasattr(sim, 'width') or not hasattr(sim, 'height'):
                    xrange = float(coords[:, 0].max() - coords[:, 0].min())
                    yrange = float(coords[:, 1].max() - coords[:, 1].min())
                    sim.width = int(np.ceil(xrange / sim.avoid_cell_size)) + 1
                    sim.height = int(np.ceil(yrange / sim.avoid_cell_size)) + 1
        except Exception:
            base_transform = compute_affine_from_hecras(np.array([[0.0, 0.0]]), target_cell_size=sim.avoid_cell_size)

    sim.mental_map_transform = base_transform

    avoid_height = int(np.round(sim.height / sim.avoid_cell_size)) + 1 if hasattr(sim, 'height') else 256
    avoid_width = int(np.round(sim.width / sim.avoid_cell_size)) + 1 if hasattr(sim, 'width') else 256
    sim.mental_map_accumulator = np.zeros((sim.num_agents, avoid_height, avoid_width), dtype='u1')

    sim.mental_map_flush_interval = getattr(sim, 'mental_map_flush_interval', 50)
    safe_flush(sim.hdf5)


def initialize_refugia_map(sim):
    mem_grp = sim.hdf5.require_group('refugia')
    if not hasattr(sim, 'width') or not hasattr(sim, 'height'):
        refugia_height = 256
        refugia_width = 256
    else:
        refugia_height = int(np.round(sim.height / sim.refugia_cell_size)) + 1
        refugia_width = int(np.round(sim.width / sim.refugia_cell_size)) + 1

    for i in range(sim.num_agents):
        name = str(i)
        if name not in mem_grp:
            mem_grp.create_dataset(name, shape=(refugia_height, refugia_width), dtype='f4',
                                   chunks=(1, min(refugia_width, 4096)), fillvalue=0.0)

    if hasattr(sim, 'depth_rast_transform') and sim.depth_rast_transform is not None:
        base_t = sim.depth_rast_transform
    else:
        try:
            if hasattr(sim, '_hecras_maps') and len(sim._hecras_maps) > 0:
                m = next(iter(sim._hecras_maps.values()))
                coords = m.coords
                base_t = compute_affine_from_hecras(coords, target_cell_size=sim.refugia_cell_size)
            else:
                base_t = compute_affine_from_hecras(np.array([[0.0, 0.0]]), target_cell_size=sim.refugia_cell_size)
        except Exception:
            base_t = compute_affine_from_hecras(np.array([[0.0, 0.0]]), target_cell_size=sim.refugia_cell_size)

    sim.refugia_map_transform = base_t
    sim.refugia_accumulator = np.zeros((sim.num_agents, refugia_height, refugia_width), dtype='f4')
    safe_flush(sim.hdf5)


def update_mental_map(sim, current_timestep):
    if 'mental_map' in sim._pixel_index_cache:
        rows, cols = sim._pixel_index_cache['mental_map']
    else:
        rows, cols = geo_to_pixel(sim.X, sim.Y, sim.mental_map_transform)

    rows = np.clip(np.round(rows).astype(int), 0, sim.mental_map_accumulator.shape[1] - 1)
    cols = np.clip(np.round(cols).astype(int), 0, sim.mental_map_accumulator.shape[2] - 1)

    agents = np.arange(sim.num_agents, dtype=int)
    sim.mental_map_accumulator[agents, rows, cols] = 1


def update_refugia_map(sim, current_velocity):
    transform = getattr(sim, 'refugia_map_transform', sim.mental_map_transform)
    key = 'refugia' if hasattr(sim, 'refugia_map_transform') else 'mental_map'
    if key in sim._pixel_index_cache:
        rows, cols = sim._pixel_index_cache[key]
    else:
        rows, cols = geo_to_pixel(sim.X, sim.Y, transform)

    try:
        sample_ds = next(iter(sim.hdf5['refugia'].values()))
        max_row = sample_ds.shape[0] - 1
        max_col = sample_ds.shape[1] - 1
    except Exception:
        max_row = int(np.round(sim.height / sim.refugia_cell_size))
        max_col = int(np.round(sim.width / sim.refugia_cell_size))

    rows = np.clip(np.round(rows).astype(int), 0, max_row)
    cols = np.clip(np.round(cols).astype(int), 0, max_col)

    agents = np.arange(sim.num_agents, dtype=int)
    cv = np.asarray(current_velocity, dtype=float)
    valid = np.isfinite(cv)
    if np.any(valid):
        ai = agents[valid]
        ri = rows[valid].astype(int)
        ci = cols[valid].astype(int)
        try:
            sim.refugia_accumulator[ai, ri, ci] = cv[valid]
        except Exception:
            for i in ai:
                r = int(rows[i])
                c = int(cols[i])
                try:
                    sim.refugia_accumulator[i, r, c] = float(cv[i])
                except Exception:
                    continue


def initial_swim_speed(sim):
    try:
        vals = sim.batch_sample_environment([sim.vel_x_rast_transform, sim.vel_y_rast_transform], ['vel_x', 'vel_y'])
        sim.x_vel = vals.get('vel_x', np.zeros(sim.num_agents))
        sim.y_vel = vals.get('vel_y', np.zeros(sim.num_agents))
    except Exception:
        sim.x_vel = np.zeros(sim.num_agents)
        sim.y_vel = np.zeros(sim.num_agents)

    water_velocities = np.sqrt(sim.x_vel**2 + sim.y_vel**2)

    ideal_velocities = np.stack((sim.ideal_sog * np.cos(sim.heading),
                                 sim.ideal_sog * np.sin(sim.heading)), axis=-1)

    sim.swim_speed = np.linalg.norm(ideal_velocities - np.array([sim.x_vel, sim.y_vel]).T, axis=-1)


def initial_heading(sim):
    try:
        if 'vel' in sim._pixel_index_cache:
            row, col = sim._pixel_index_cache['vel']
        else:
            row, col = geo_to_pixel(sim.X, sim.Y, sim.vel_dir_rast_transform)

        values = sim.batch_sample_environment([sim.vel_dir_rast_transform], ['vel_dir'])['vel_dir']

        sim.heading = np.where(values < 0,
                               (np.radians(360) + values) - np.radians(180),
                               values - np.radians(180))
    except Exception:
        try:
            vals_deg = np.degrees(np.arctan2(sim.y_vel, sim.x_vel))
        except Exception:
            vals_deg = np.zeros(sim.num_agents, dtype=float)

        sim.heading = np.deg2rad(vals_deg) - np.pi

    sim.max_practical_sog = np.array([sim.sog * np.cos(sim.heading), sim.sog * np.sin(sim.heading)])


# Smoke import check
try:
    print('BEHAVIOR_IMPORT_OK')
except Exception:
    pass
