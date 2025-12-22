import os
import numpy as np
import rasterio


def initialize_hdf5(sim):
    """Initialize an HDF5 database for a simulation-like object.

    This mirrors the monolith's `initialize_hdf5` method but operates on a
    `sim` object passed in. It creates groups/datasets expected by the
    simulation (agent_data, static and time-varying datasets) and sets basic
    attributes.
    """
    try:
        agent_data = sim.hdf5.create_group("agent_data")

        # static datasets
        agent_data.create_dataset("sex", (sim.num_agents,), dtype='f4')
        agent_data.create_dataset("length", (sim.num_agents,), dtype='f4')
        agent_data.create_dataset("ucrit", (sim.num_agents,), dtype='f4')
        agent_data.create_dataset("weight", (sim.num_agents,), dtype='f4')
        agent_data.create_dataset("body_depth", (sim.num_agents,), dtype='f4')
        agent_data.create_dataset("too_shallow", (sim.num_agents,), dtype='f4')
        agent_data.create_dataset("opt_wat_depth", (sim.num_agents,), dtype='f4')

        # time-varying datasets (chunked for per-timestep writes)
        chunk_shape = (sim.num_agents, 1)
        agent_data.create_dataset("X", (sim.num_agents, sim.num_timesteps), dtype='f4', chunks=chunk_shape)
        agent_data.create_dataset("Y", (sim.num_agents, sim.num_timesteps), dtype='f4', chunks=chunk_shape)
        agent_data.create_dataset("Z", (sim.num_agents, sim.num_timesteps), dtype='f4', chunks=chunk_shape)
        agent_data.create_dataset("prev_X", (sim.num_agents, sim.num_timesteps), dtype='f4', chunks=chunk_shape)
        agent_data.create_dataset("prev_Y", (sim.num_agents, sim.num_timesteps), dtype='f4', chunks=chunk_shape)
        agent_data.create_dataset("heading", (sim.num_agents, sim.num_timesteps), dtype='f4', chunks=chunk_shape)
        agent_data.create_dataset("sog", (sim.num_agents, sim.num_timesteps), dtype='f4', chunks=chunk_shape)
        agent_data.create_dataset("ideal_sog", (sim.num_agents, sim.num_timesteps), dtype='f4', chunks=chunk_shape)
        agent_data.create_dataset("swim_speed", (sim.num_agents, sim.num_timesteps), dtype='f4', chunks=chunk_shape)
        agent_data.create_dataset("battery", (sim.num_agents, sim.num_timesteps), dtype='f4', chunks=chunk_shape)
        agent_data.create_dataset("swim_behav", (sim.num_agents, sim.num_timesteps), dtype='f4', chunks=chunk_shape)
        agent_data.create_dataset("swim_mode", (sim.num_agents, sim.num_timesteps), dtype='f4', chunks=chunk_shape)
        agent_data.create_dataset("recover_stopwatch", (sim.num_agents, sim.num_timesteps), dtype='f4', chunks=chunk_shape)
        agent_data.create_dataset("ttfr", (sim.num_agents, sim.num_timesteps), dtype='f4', chunks=chunk_shape)
        agent_data.create_dataset("time_out_of_water", (sim.num_agents, sim.num_timesteps), dtype='f4', chunks=chunk_shape)
        agent_data.create_dataset("drag", (sim.num_agents, sim.num_timesteps), dtype='f4', chunks=chunk_shape)
        agent_data.create_dataset("thrust", (sim.num_agents, sim.num_timesteps), dtype='f4', chunks=chunk_shape)
        agent_data.create_dataset("Hz", (sim.num_agents, sim.num_timesteps), dtype='f4', chunks=chunk_shape)
        agent_data.create_dataset("bout_no", (sim.num_agents, sim.num_timesteps), dtype='f4', chunks=chunk_shape)
        agent_data.create_dataset("dist_per_bout", (sim.num_agents, sim.num_timesteps), dtype='f4', chunks=chunk_shape)
        agent_data.create_dataset("bout_dur", (sim.num_agents, sim.num_timesteps), dtype='f4', chunks=chunk_shape)
        agent_data.create_dataset("time_of_jump", (sim.num_agents, sim.num_timesteps), dtype='f4', chunks=chunk_shape)
        agent_data.create_dataset("kcal", (sim.num_agents, sim.num_timesteps), dtype='f4', chunks=chunk_shape)
    except Exception:
        raise

    # metadata
    sim.hdf5.attrs['simulation_name'] = "%s Sockeye Movement Simulation" % (getattr(sim, 'basin', ''))
    sim.hdf5.attrs['num_agents'] = sim.num_agents
    sim.hdf5.attrs['num_timesteps'] = sim.num_timesteps
    sim.hdf5.attrs['basin'] = getattr(sim, 'basin', '')
    sim.hdf5.attrs['crs'] = getattr(sim, 'crs', '')

    sim.hdf5.flush()


def timestep_flush(sim, timestep):
    """Flush per-timestep buffers for a simulation-like object to HDF5 or deferred logs.

    This function mirrors the monolith's `timestep_flush`, using sim._hdf5_buffers
    and sim._buffer_pos. It supports `defer_hdf` paths using `sim._memmap_writer` or
    `sim._log_writer` if present.
    """
    if getattr(sim, 'pid_tuning', False):
        return

    buf_vals = {
        'X': sim.X.astype('float32'),
        'Y': sim.Y.astype('float32'),
        'Z': getattr(sim, 'z', np.zeros_like(sim.X)).astype('float32'),
        'prev_X': sim.prev_X.astype('float32'),
        'prev_Y': sim.prev_Y.astype('float32'),
        'heading': getattr(sim, 'heading', np.zeros_like(sim.X)).astype('float32'),
        'sog': getattr(sim, 'sog', np.zeros_like(sim.X)).astype('float32'),
        'ideal_sog': getattr(sim, 'ideal_sog', np.zeros_like(sim.X)).astype('float32'),
        'swim_speed': getattr(sim, 'swim_speed', np.zeros_like(sim.X)).astype('float32'),
        'battery': sim.battery.astype('float32'),
        'swim_behav': sim.swim_behav.astype('float32'),
        'swim_mode': sim.swim_mode.astype('float32'),
        'recover_stopwatch': sim.recover_stopwatch.astype('float32'),
        'ttfr': sim.ttfr.astype('float32'),
        'time_out_of_water': sim.time_out_of_water.astype('float32'),
        'drag': np.linalg.norm(sim.drag, axis=-1).astype('float32') if getattr(sim, 'drag', None) is not None else np.zeros(sim.num_agents, dtype='float32'),
        'thrust': np.linalg.norm(sim.thrust, axis=-1).astype('float32') if getattr(sim, 'thrust', None) is not None else np.zeros(sim.num_agents, dtype='float32'),
        'Hz': sim.Hz.astype('float32'),
        'bout_no': sim.bout_no.astype('float32'),
        'dist_per_bout': sim.dist_per_bout.astype('float32'),
        'bout_dur': sim.bout_dur.astype('float32'),
        'kcal': sim.kcal.astype('float32')
    }

    pos = sim._buffer_pos
    for k, v in buf_vals.items():
        if k in sim._hdf5_buffers:
            sim._hdf5_buffers[k][:, pos] = v

    sim._buffer_pos += 1
    if sim._buffer_pos >= sim.flush_interval or timestep == (sim.num_timesteps - 1):
        write_len = sim._buffer_pos
        t_end = timestep
        t_start = t_end - write_len + 1

        if getattr(sim, 'defer_hdf', False):
            # memmap writer initialization
            if getattr(sim, 'defer_log_fmt', 'npz') == 'memmap' and getattr(sim, '_memmap_writer', None) is None:
                try:
                    var_shapes = {k: (sim.num_agents, sim.num_timesteps) for k in sim._hdf5_buffers.keys()}
                    from emergent.io.log_writer_memmap import MemmapLogWriter
                    out_dir = sim._memmap_config.get('out_dir', os.path.join(sim.model_dir, 'logs', 'deferred'))
                    sim._memmap_writer = MemmapLogWriter(out_dir, var_shapes, dtype=np.float32)
                except Exception:
                    sim._memmap_writer = None

            if getattr(sim, '_memmap_writer', None) is not None:
                arrays_2d = {}
                for k in sim._hdf5_buffers.keys():
                    try:
                        arrays_2d[k] = sim._hdf5_buffers[k][:, :write_len].astype('f4')
                    except Exception:
                        arrays_2d[k] = np.zeros((sim.num_agents, write_len), dtype='f4')
                try:
                    sim._memmap_writer.append_block(t_start, arrays_2d)
                except Exception:
                    for offset in range(write_len):
                        t_idx = t_start + offset
                        arrays = {k: sim._hdf5_buffers[k][:, offset].astype('f4') for k in sim._hdf5_buffers.keys()}
                        try:
                            sim._memmap_writer.append(t_idx, arrays)
                        except Exception:
                            pass
            elif getattr(sim, '_log_writer', None) is not None:
                for offset in range(write_len):
                    t_idx = t_start + offset
                    arrays = {k: sim._hdf5_buffers[k][:, offset].astype('f4') for k in sim._hdf5_buffers.keys()}
                    try:
                        sim._log_writer.append(t_idx, arrays)
                    except Exception:
                        pass
        else:
            for k, buf in sim._hdf5_buffers.items():
                ds_name = f'agent_data/{k}'
                if ds_name in sim.hdf5:
                    try:
                        sim.hdf5[ds_name][:, t_start:t_end+1] = buf[:, :write_len]
                    except Exception:
                        for offset in range(write_len):
                            sim.hdf5[ds_name][:, t_start + offset] = buf[:, offset]

        for k in list(sim._hdf5_buffers.keys()):
            sim._hdf5_buffers[k][:] = 0
        sim._buffer_pos = 0

        # Best-effort flush of mental map accumulators
        try:
            mem_grp = sim.hdf5.get('memory')
            if mem_grp is not None:
                for aid in range(sim.num_agents):
                    ds = mem_grp.get(str(aid))
                    if ds is None:
                        continue
                    acc = sim.mental_map_accumulator[aid]
                    if np.any(acc):
                        existing = ds[:, :]
                        np.maximum(existing, acc, out=existing)
                        ds[:, :] = existing
                        acc.fill(0)
        except Exception:
            pass

        # Flush refugia accumulators if present
        try:
            refugia_grp = sim.hdf5.get('refugia')
            if refugia_grp is not None:
                acc_r_list = getattr(sim, 'refugia_accumulator', None)
                if acc_r_list is not None:
                    for aid in range(sim.num_agents):
                        acc_arr = acc_r_list[aid]
                        if not np.any(acc_arr):
                            continue
                        dsr = refugia_grp.get(str(aid))
                        if dsr is None:
                            continue
                        existing_r = dsr[:, :]
                        mask = acc_arr != 0
                        existing_r[mask] = acc_arr[mask]
                        dsr[:, :] = existing_r
                        acc_arr.fill(0)
        except Exception:
            pass

        sim.hdf5.flush()


def enviro_import(sim, data_dir, surface_type):
    """Import an environmental raster into the simulation HDF5 environment group.

    This is adapted from the monolith and writes raster bands into
    `sim.hdf5['environment/<name>']` datasets. It will also create `x_coords`
    and `y_coords` datasets if missing.
    """
    if not data_dir or not os.path.exists(data_dir):
        return

    try:
        src = rasterio.open(data_dir)
    except Exception:
        return

    num_bands = src.count
    width = src.width
    height = src.height
    transform = src.transform
    sim.no_data_value = src.nodatavals[0]

    if 'environment' not in sim.hdf5:
        env_data = sim.hdf5.create_group('environment')
    else:
        env_data = sim.hdf5['environment']

    sim.width = width
    sim.height = height

    # create x_coords/y_coords if missing
    if 'x_coords' not in sim.hdf5:
        rows, cols = src.shape
        chunk_size = 1024
        dset_x = sim.hdf5.create_dataset('x_coords', (height, width), dtype='float32')
        dset_y = sim.hdf5.create_dataset('y_coords', (height, width), dtype='float32')
        for i in range(0, rows, chunk_size):
            row_chunk = slice(i, min(i + chunk_size, rows))
            row_indices, col_indices = np.meshgrid(np.arange(row_chunk.start, row_chunk.stop), np.arange(cols), indexing='ij')
            x_coords, y_coords = transform * (col_indices, row_indices)
            dset_x[row_chunk, :] = x_coords.astype('float32')
            dset_y[row_chunk, :] = y_coords.astype('float32')
        sim.hdf5.flush()

    def _create_or_replace_dataset(group, name, shape, dtype='f4', chunks=None):
        if name in group:
            del group[name]
        return group.create_dataset(name, shape, dtype=dtype, chunks=chunks)

    shape = (num_bands, height, width)
    if surface_type == 'wetted':
        sim.wetted_transform = transform
        arr = src.read(1)
        _create_or_replace_dataset(env_data, "wetted", (height, width), dtype='f4', chunks=(1, min(width, 4096)))
        sim.hdf5['environment/wetted'][:, :] = arr
    elif surface_type == 'velocity x':
        sim.vel_x_rast_transform = transform
        arr = src.read(1)
        _create_or_replace_dataset(env_data, "vel_x", (height, width), dtype='f4', chunks=(1, min(width, 4096)))
        sim.hdf5['environment/vel_x'][:, :] = arr
    elif surface_type == 'velocity y':
        sim.vel_y_rast_transform = transform
        arr = src.read(1)
        _create_or_replace_dataset(env_data, "vel_y", (height, width), dtype='f4', chunks=(1, min(width, 4096)))
        sim.hdf5['environment/vel_y'][:, :] = arr
    elif surface_type == 'depth':
        sim.depth_rast_transform = transform
        arr = src.read(1)
        _create_or_replace_dataset(env_data, "depth", (height, width), dtype='f4', chunks=(1, min(width, 4096)))
        sim.hdf5['environment/depth'][:, :] = arr
    elif surface_type == 'wsel':
        sim.wsel_rast_transform = transform
        arr = src.read(1)
        _create_or_replace_dataset(env_data, "wsel", (height, width), dtype='f4', chunks=(1, min(width, 4096)))
        sim.hdf5['environment/wsel'][:, :] = arr
    elif surface_type == 'elevation':
        sim.elev_rast_transform = transform
        arr = src.read(1)
        _create_or_replace_dataset(env_data, "elevation", (height, width), dtype='f4', chunks=(1, min(width, 4096)))
        sim.hdf5['environment/elevation'][:, :] = arr
    elif surface_type == 'velocity direction':
        sim.vel_dir_rast_transform = transform
        arr = src.read(1)
        _create_or_replace_dataset(env_data, "vel_dir", (height, width), dtype='f4', chunks=(1, min(width, 4096)))
        sim.hdf5['environment/vel_dir'][:, :] = arr
    elif surface_type == 'velocity magnitude':
        sim.vel_mag_rast_transform = transform
        arr = src.read(1)
        _create_or_replace_dataset(env_data, "vel_mag", (height, width), dtype='f4', chunks=(1, min(width, 4096)))
        sim.hdf5['environment/vel_mag'][:, :] = arr

    sim.width = width
    sim.height = height
    sim.hdf5.flush()
    src.close()
