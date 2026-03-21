#!/usr/bin/env python3
"""
Real-time RL training visualizer for behavioral weight optimization.

Three-panel Qt interface:
- Left panel: Behavioral weights and diagnostics
- Center panel: Simulation playback (similar to realtime_viewer)
- Right panel: Training controls (play/pause/stop, generation settings, etc.)

Usage:
    python -m emergent.salmon_abm.rl_training_viewer --model-dir data/salmon_abm --start-polygon data/salmon_abm/start_loc_river_right.shp
    
    # Or programmatically
    from emergent.salmon_abm.rl_training_viewer import RLTrainingViewer
    viewer = RLTrainingViewer()
    viewer.start_training(...)
"""
from __future__ import annotations

import sys
import os
import time
import threading
import argparse
import multiprocessing as mp
import queue as queue_mod
import json
from collections import deque
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
from datetime import datetime

import numpy as np
import h5py

try:
    from PyQt5.QtWidgets import (
        QApplication,
        QMainWindow,
        QWidget,
        QPushButton,
        QSlider,
        QLabel,
        QHBoxLayout,
        QVBoxLayout,
        QGridLayout,
        QGroupBox,
        QSpinBox,
        QDoubleSpinBox,
        QTextEdit,
        QSplitter,
        QOpenGLWidget,
        QProgressBar,
        QLineEdit,
        QFileDialog,
        QCheckBox,
        QComboBox,
        QScrollArea,
        QSizePolicy,
        QAction,
    )
    from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QObject, QThread, QSettings, QEvent
    from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QImage
except ImportError as e:
    raise ImportError(f"PyQt5 is required for RL training visualizer: {e}")

# Import RL training components
try:
    from emergent.salmon_abm.rl_training import BehavioralWeights, RLTrainer
    from emergent.salmon_abm.simulation import simulation
    from emergent.salmon_abm.realtime_viewer import ReplayWidget, load_env_from_h5
except ImportError as e:
    raise ImportError(f"Could not import RL training components: {e}")


def _drain_control_messages(control_queue, paused: bool, stopped: bool) -> Tuple[bool, bool]:
    """Return (paused, stopped) after draining control messages."""
    while True:
        try:
            msg = control_queue.get_nowait()
        except queue_mod.Empty:
            break
        if msg == "stop":
            stopped = True
            paused = False
        elif msg == "pause":
            paused = True
        elif msg == "resume":
            paused = False
    return paused, stopped


def _training_process_main(config: Dict[str, Any], output_queue, control_queue) -> None:
    """Run RL training in a subprocess to avoid UI/GIL contention."""
    import traceback
    import numpy as np
    from emergent.salmon_abm.rl_training import BehavioralWeights, RLTrainer, compute_episode_reward
    from emergent.salmon_abm.simulation import simulation

    try:
        def _emit_nonblocking(msg: Dict[str, Any]) -> None:
            try:
                output_queue.put_nowait(msg)
            except queue_mod.Full:
                # Status updates are best-effort; skip when queue is saturated.
                pass

        def _emit_status(message: str) -> None:
            _emit_nonblocking({"type": "status", "message": str(message)})

        env_files = config.get("env_files") or []
        hecras_plan_path = config.get("hecras_plan_path")
        if hecras_plan_path:
            hecras_plan_path = os.path.abspath(str(hecras_plan_path))
            if not os.path.exists(hecras_plan_path):
                raise FileNotFoundError(f"HECRAS plan not found: {hecras_plan_path}")
        elif not env_files:
            raise ValueError("No environment files provided for training process (and no hecras_plan_path)")

        start_polygons = config.get("start_polygons") or []
        if not start_polygons:
            single_start = config.get("start_polygon")
            if single_start:
                start_polygons = [single_start]
        start_polygons = [str(p) for p in start_polygons if p]
        if not start_polygons:
            raise ValueError("No start polygons provided for training process")
        for sp in start_polygons:
            if not os.path.exists(sp):
                raise FileNotFoundError(f"Start polygon not found: {sp}")

        longitudinal_path = config.get("longitudinal_path")
        if not longitudinal_path or not os.path.exists(longitudinal_path):
            raise FileNotFoundError(f"Longitudinal profile shapefile required but not found: {longitudinal_path}")

        num_episodes = int(config.get("num_episodes", 1))
        num_timesteps = int(config.get("num_timesteps", 50))
        num_agents = int(config.get("num_agents", 200))
        exploration_noise = float(config.get("exploration_noise", 0.1))
        reward_weights = config.get("reward_weights") or None
        body_length = float(config.get("body_length", 0.3))
        dt = float(config.get("dt", 1.0))
        initial_heading_mode = str(config.get("initial_heading_mode", "uniform")).strip().lower()
        initial_sog_mode = str(config.get("initial_sog_mode", "uniform")).strip().lower()
        initial_sog_min = float(config.get("initial_sog_min", 0.1))
        initial_sog_max = float(config.get("initial_sog_max", 1.5))
        if initial_sog_max < initial_sog_min:
            raise ValueError(
                f"initial_sog_max ({initial_sog_max}) must be >= initial_sog_min ({initial_sog_min})"
            )
        hecras_start_index = int(config.get("hecras_start_index", 30))
        hecras_time_mode = str(config.get("hecras_time_mode", "loop")).strip().lower()
        hecras_k = int(config.get("hecras_k", 8))
        hecras_cell_size = config.get("hecras_cell_size")
        if hecras_cell_size is not None:
            hecras_cell_size = float(hecras_cell_size)
        hecras_wetted_threshold = config.get("hecras_wetted_threshold")
        if hecras_wetted_threshold is not None:
            hecras_wetted_threshold = float(hecras_wetted_threshold)

        initial_weights = BehavioralWeights.from_dict(config.get("initial_weights") or {})
        archive_h5_path = config.get("archive_h5_path")

        def _json_ready(d: Dict[str, Any]) -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            for key, value in d.items():
                if isinstance(value, (np.floating, np.integer)):
                    out[str(key)] = float(value)
                elif isinstance(value, np.ndarray):
                    out[str(key)] = value.tolist()
                else:
                    out[str(key)] = value
            return out

        if archive_h5_path:
            archive_h5_path = os.path.abspath(str(archive_h5_path))
            archive_dir = os.path.dirname(archive_h5_path)
            if archive_dir:
                os.makedirs(archive_dir, exist_ok=True)
            # Initialize archive once, then close immediately so UI can read while training runs.
            with h5py.File(archive_h5_path, "w") as archive_h5:
                archive_h5.attrs["created_utc"] = datetime.utcnow().isoformat()
                archive_h5.attrs["model_dir"] = str(config.get("model_dir", ""))
                archive_h5.attrs["num_episodes"] = int(num_episodes)
                archive_h5.attrs["num_timesteps"] = int(num_timesteps)
                archive_h5.attrs["num_agents"] = int(num_agents)
                archive_h5.attrs["exploration_noise"] = float(exploration_noise)
                archive_h5.attrs["reward_weights_json"] = json.dumps(_json_ready(reward_weights or {}))
                archive_h5.attrs["hecras_plan_path"] = str(hecras_plan_path or "")
                archive_h5.attrs["hecras_start_index"] = int(hecras_start_index)
                archive_h5.attrs["hecras_time_mode"] = str(hecras_time_mode)
                archive_h5.attrs["hecras_k"] = int(hecras_k)
                archive_h5.attrs["hecras_cell_size"] = (
                    float(hecras_cell_size) if hecras_cell_size is not None else np.nan
                )
                archive_h5.attrs["hecras_wetted_threshold"] = (
                    float(hecras_wetted_threshold) if hecras_wetted_threshold is not None else np.nan
                )
                archive_h5.create_group("episodes")

        current_start_polygon = {"path": start_polygons[0]}

        def simulation_factory(weights: BehavioralWeights):
            sim = simulation(
                model_dir=config.get("model_dir"),
                model_name=config.get("model_name"),
                crs=config.get("crs"),
                basin=config.get("basin"),
                water_temp=float(config.get("water_temp", 12.0)),
                start_polygon=current_start_polygon["path"],
                env_files=env_files,
                longitudinal_profile=longitudinal_path,
                fish_length=None,
                num_timesteps=num_timesteps,
                num_agents=num_agents,
                use_gpu=False,
                pid_tuning=False,
                db_path=None,
                output_write_mode="none",
                output_write_backend="sync",
                hecras_plan_path=hecras_plan_path,
                hecras_start_index=hecras_start_index,
                hecras_time_mode=hecras_time_mode,
                hecras_k=hecras_k,
                hecras_cell_size=hecras_cell_size,
                hecras_wetted_threshold=hecras_wetted_threshold,
            )

            sim.load_behavioral_weights(weights_dict=weights.to_dict())

            fish_length_m = 0.3
            sensory_range = 5.0
            sim.neighbor_buffer_radius = sensory_range * fish_length_m
            sim.neighbor_buffer_lengths = sensory_range

            # Initial heading/SOG policy is applied in RLTrainer.run_episode() after reset_spatial_state().
            return sim

        trainer = RLTrainer(
            simulation_factory=simulation_factory,
            initial_weights=initial_weights,
            config={
                "exploration_noise": exploration_noise,
                "body_length": body_length,
                "dt": dt,
                "num_timesteps": num_timesteps,
                "reward_weights": reward_weights,
                "initial_heading_mode": initial_heading_mode,
                "initial_sog_mode": initial_sog_mode,
                "initial_sog_min": initial_sog_min,
                "initial_sog_max": initial_sog_max,
            },
        )
        _emit_status(
            f"Training process ready (episodes={num_episodes}, timesteps={num_timesteps}, agents={num_agents})"
        )

        longitudinal_profile = None
        try:
            sim = trainer.simulation_factory(trainer.initial_weights)
            longitudinal_profile = getattr(sim, "longitudinal", None)
            sim.close()
        except Exception as e:
            raise RuntimeError(f"Failed to cache longitudinal profile: {e}") from e

        if longitudinal_profile is None:
            raise ValueError("Longitudinal profile not loaded from simulation")

        current_weights = trainer.initial_weights
        paused = False
        stopped = False

        for episode in range(num_episodes):
            paused, stopped = _drain_control_messages(control_queue, paused, stopped)
            if stopped:
                break
            while paused and not stopped:
                time.sleep(0.1)
                paused, stopped = _drain_control_messages(control_queue, paused, stopped)
            if stopped:
                break

            episode_start_polygon = start_polygons[episode % len(start_polygons)]
            current_start_polygon["path"] = episode_start_polygon

            output_queue.put(
                {
                    "type": "episode_started",
                    "episode": episode,
                    "total": num_episodes,
                    "start_polygon": episode_start_polygon,
                }
            )
            _emit_status(
                f"Episode {episode + 1}/{num_episodes}: simulation running ({num_timesteps} timesteps)"
            )
            progress_interval = max(1, int(num_timesteps // 20))

            def _episode_progress(step_num: int, total_steps: int) -> None:
                _emit_status(
                    f"Episode {episode + 1}/{num_episodes}: timestep {step_num}/{total_steps}"
                )

            positions, headings, velocities, battery, alive, velocity_field = trainer.run_episode(
                current_weights,
                progress_callback=_episode_progress,
                progress_interval=progress_interval,
            )
            _emit_status(f"Episode {episode + 1}/{num_episodes}: simulation complete, scoring")

            def _scoring_progress(message: str) -> None:
                _emit_status(f"Episode {episode + 1}/{num_episodes}: {message}")

            reward, components = compute_episode_reward(
                positions,
                headings,
                velocities,
                alive,
                body_length=trainer.body_length,
                threat_level=current_weights.threat_level,
                behavioral_weights=current_weights.to_dict(),
                battery_history=battery,
                longitudinal_profile=longitudinal_profile,
                velocity_field_history=velocity_field,
                reward_weights=trainer.reward_weights,
                phase_callback=_scoring_progress,
            )

            if reward > trainer.best_reward:
                trainer.best_reward = reward
                trainer.best_weights = current_weights

            trainer.episode_history.append((episode, float(reward)))

            archive_group = None
            if archive_h5_path:
                group_name = f"{episode:06d}"
                archive_group = f"episodes/{group_name}"
                try:
                    # Open/write/close per-episode to avoid file-locking stalls in viewer replay.
                    with h5py.File(archive_h5_path, "a") as archive_h5:
                        episodes_group = archive_h5.require_group("episodes")
                        if group_name in episodes_group:
                            del episodes_group[group_name]
                        ep_group = episodes_group.create_group(group_name)
                        ep_group.attrs["episode"] = int(episode)
                        ep_group.attrs["reward"] = float(reward)
                        ep_group.attrs["start_polygon"] = str(episode_start_polygon)
                        ep_group.attrs["weights_json"] = json.dumps(_json_ready(current_weights.to_dict()))
                        ep_group.attrs["components_json"] = json.dumps(_json_ready(components))
                        ep_group.create_dataset("positions", data=np.asarray(positions, dtype=np.float32))
                        ep_group.create_dataset("headings", data=np.asarray(headings, dtype=np.float32))
                        ep_group.create_dataset("battery", data=np.asarray(battery, dtype=np.float32))
                        ep_group.create_dataset("alive", data=np.asarray(alive, dtype=np.uint8))
                except Exception as archive_exc:
                    archive_group = None
                    _emit_status(
                        f"Episode {episode + 1}/{num_episodes}: archive write failed ({archive_exc})"
                    )

            payload_positions = positions
            payload_headings = headings
            payload_battery = battery
            payload_alive = alive
            if archive_group is not None:
                # Keep IPC messages lightweight; load trajectory lazily from archive in UI process.
                payload_positions = None
                payload_headings = None
                payload_battery = None
                payload_alive = None

            output_queue.put(
                {
                    "type": "episode_computed",
                    "episode": episode,
                    "reward": float(reward),
                    "components": components,
                    "positions": payload_positions,
                    "headings": payload_headings,
                    "battery": payload_battery,
                    "alive": payload_alive,
                    "weights": current_weights.to_dict(),
                    "best_reward": float(trainer.best_reward),
                    "archive_group": archive_group,
                    "archive_h5_path": archive_h5_path,
                }
            )

            current_weights = trainer.best_weights.mutate(mutation_scale=trainer.exploration_noise)
            _emit_status(f"Episode {episode + 1}/{num_episodes}: done")

        if stopped:
            output_queue.put(
                {
                    "type": "training_stopped",
                    "best_weights": trainer.best_weights.to_dict(),
                    "history": trainer.episode_history,
                    "archive_h5_path": archive_h5_path,
                }
            )
        else:
            output_queue.put(
                {
                    "type": "training_completed",
                    "best_weights": trainer.best_weights.to_dict(),
                    "history": trainer.episode_history,
                    "archive_h5_path": archive_h5_path,
                }
            )
            _emit_status("Training complete")
    except Exception as exc:
        msg = f"Training error: {exc}\n{traceback.format_exc()}"
        try:
            output_queue.put({"type": "error", "message": msg})
        finally:
            raise

class SimulationCanvas(QWidget):
    """
    Center panel: Real-time simulation visualization using ReplayWidget.
    
    Wrapper around ReplayWidget that handles live position updates during training.
    """
    
    # Signal emitted when animation completes
    animation_finished = pyqtSignal()
    
    def __init__(
        self,
        parent=None,
        model_dir: Optional[str] = None,
        hecras_plan_path: Optional[str] = None,
        hecras_start_index: int = 0,
        hecras_k: int = 8,
    ):
        super().__init__(parent)

        self.model_dir = model_dir
        self.hecras_plan_path = hecras_plan_path
        self.hecras_start_index = int(hecras_start_index)
        self.hecras_k = int(hecras_k)
        self._hecras_bg_cache_key: Optional[Tuple[str, int, int]] = None
        self._hecras_bg_cache_value: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None

        # Velocity fields are optional (arrow overlays)
        self.vel_x_data = None
        self.vel_y_data = None
        self.vel_transform = None
        self.vel_bbox = None

        dummy_positions = np.zeros((1, 1, 2), dtype=np.float32)
        self.replay_widget = ReplayWidget(
            positions=dummy_positions,
            parent=self,
            pad=1.15,
            point_size=4.0,
        )

        # Connect to replay widget's timer to detect when animation finishes
        self._animation_finished_emitted = False
        self._animation_watchdog = QTimer(self)
        self._animation_watchdog.setInterval(100)
        self._animation_watchdog.timeout.connect(self._poll_animation_complete)
        self.replay_widget.timer.timeout.connect(self._check_animation_complete)

        # Layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.replay_widget)
        self.setLayout(layout)

        # Configure depth background immediately (blue ramp handled by ReplayWidget).
        self.configure_environment(
            model_dir=model_dir,
            hecras_plan_path=hecras_plan_path,
            hecras_start_index=self.hecras_start_index,
            hecras_k=self.hecras_k,
        )

    @staticmethod
    def _coerce_replay_positions(positions: Any) -> np.ndarray:
        """Return a valid (T, N, 2) array for ReplayWidget construction."""
        if positions is None:
            return np.zeros((1, 1, 2), dtype=np.float32)
        arr = np.asarray(positions, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[2] == 2 and arr.shape[0] > 0 and arr.shape[1] > 0:
            return arr
        if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] > 0:
            return arr[np.newaxis, :, :]
        print(
            f"Warning: Invalid replay positions shape {getattr(arr, 'shape', None)}; using dummy positions",
            flush=True,
        )
        return np.zeros((1, 1, 2), dtype=np.float32)

    def _resolve_background_sources(
        self,
        model_dir: Optional[str],
        hecras_plan_path: Optional[str],
        hecras_start_index: int,
        hecras_k: int,
    ) -> Tuple[Optional[str], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Resolve depth background from model rasters or HECRAS HDF environment arrays."""
        env_depth = None
        env_depth_array = None
        env_x_coords = None
        env_y_coords = None
        wetted_mask_array = None

        if model_dir and os.path.exists(model_dir):
            depth_path = os.path.join(model_dir, "depth.tif")
            if os.path.exists(depth_path):
                env_depth = depth_path

        if env_depth is None and hecras_plan_path and os.path.exists(hecras_plan_path):
            try:
                depth, x_coords, y_coords, wetted = load_env_from_h5(hecras_plan_path)
                if depth is not None and x_coords is not None and y_coords is not None:
                    env_depth_array = np.asarray(depth, dtype=float)
                    env_x_coords = np.asarray(x_coords, dtype=float)
                    env_y_coords = np.asarray(y_coords, dtype=float)
                    wetted_mask_array = None if wetted is None else np.asarray(wetted, dtype=bool)
                else:
                    cache_key = (os.path.abspath(hecras_plan_path), int(hecras_start_index), int(hecras_k))
                    if self._hecras_bg_cache_key == cache_key and self._hecras_bg_cache_value is not None:
                        d, x, y, w = self._hecras_bg_cache_value
                        env_depth_array = np.asarray(d, dtype=float)
                        env_x_coords = np.asarray(x, dtype=float)
                        env_y_coords = np.asarray(y, dtype=float)
                        wetted_mask_array = np.asarray(w, dtype=bool)
                    else:
                        derived = self._derive_hecras_depth_background(
                            hecras_plan_path=hecras_plan_path,
                            start_index=hecras_start_index,
                            k=hecras_k,
                        )
                        if derived is not None:
                            env_depth_array, env_x_coords, env_y_coords, wetted_mask_array = derived
                            self._hecras_bg_cache_key = cache_key
                            self._hecras_bg_cache_value = (
                                np.asarray(env_depth_array, dtype=np.float32),
                                np.asarray(env_x_coords, dtype=np.float32),
                                np.asarray(env_y_coords, dtype=np.float32),
                                np.asarray(wetted_mask_array, dtype=bool),
                            )
            except Exception as e:
                print(f"Warning: Could not load HECRAS depth background: {e}")

        return env_depth, env_depth_array, env_x_coords, env_y_coords, wetted_mask_array

    def _derive_hecras_depth_background(
        self,
        hecras_plan_path: str,
        start_index: int,
        k: int,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Build a coarse depth raster from HECRAS plan for viewer background when env/depth is absent."""
        try:
            from emergent.salmon_abm import hecras_io
        except Exception as e:
            print(f"Warning: Could not import hecras_io for depth fallback: {e}")
            return None

        plan = None
        try:
            plan = hecras_io.HecrasPlan(hecras_plan_path, area_name="2D area")
            if plan.time_s is not None and len(plan.time_s) > 0:
                idx = max(0, min(int(start_index), len(plan.time_s) - 1))
            else:
                idx = max(0, int(start_index))

            # Coarser grid for fast interactive background rendering.
            transform, shape = hecras_io.compute_grid_from_coords(plan.coords, target_cell_size=2.0)
            fields = plan.read_fields(idx, ("depth",))
            mapped = plan.map_values_to_grid(fields, shape, transform, k=max(1, int(k)))
            depth = np.asarray(mapped["depth"], dtype=np.float32)

            nrows, ncols = int(shape[0]), int(shape[1])
            a, b, c, d, e, f = transform
            cols = np.arange(ncols, dtype=np.float32)
            rows = np.arange(nrows, dtype=np.float32)
            col_idx, row_idx = np.meshgrid(cols, rows)
            x_coords = a * col_idx + b * row_idx + c
            y_coords = d * col_idx + e * row_idx + f
            wetted = np.isfinite(depth) & (depth > 0.05)
            return depth, x_coords, y_coords, wetted
        except Exception as e:
            print(f"Warning: HECRAS depth fallback rasterization failed: {e}")
            return None
        finally:
            if plan is not None:
                plan.close()

    def _load_velocity_fields(self, model_dir: Optional[str]) -> None:
        self.vel_x_data = None
        self.vel_y_data = None
        self.vel_transform = None
        self.vel_bbox = None
        if not model_dir or not os.path.exists(model_dir):
            return

        vel_x_path = os.path.join(model_dir, "vel_x.tif")
        vel_y_path = os.path.join(model_dir, "vel_y.tif")
        if not (os.path.exists(vel_x_path) and os.path.exists(vel_y_path)):
            return

        try:
            from emergent.salmon_abm import io as _io
            vel_x_arr, vel_x_transform, _ = _io.enviro_import(vel_x_path)
            vel_y_arr, _, _ = _io.enviro_import(vel_y_path)

            self.vel_x_data = np.array(vel_x_arr, dtype=float)
            self.vel_y_data = np.array(vel_y_arr, dtype=float)
            self.vel_transform = vel_x_transform

            h, w = self.vel_x_data.shape
            try:
                a_t, b_t, c_t, d_t, e_t, f_t = vel_x_transform
            except Exception:
                t = vel_x_transform
                a_t, b_t, c_t, d_t, e_t, f_t = (t.a, t.b, t.c, t.d, t.e, t.f)

            xs = [a_t * c + b_t * r + c_t for c in [0, w] for r in [0, h]]
            ys = [d_t * c + e_t * r + f_t for c in [0, w] for r in [0, h]]
            self.vel_bbox = (min(xs), max(xs), min(ys), max(ys))
        except Exception as e:
            print(f"Warning: Could not load velocity fields: {e}")

    def configure_environment(
        self,
        model_dir: Optional[str],
        hecras_plan_path: Optional[str],
        hecras_start_index: int = 0,
        hecras_k: int = 8,
    ) -> None:
        """Refresh replay background (depth) and velocity overlays from current inputs."""
        self.model_dir = model_dir
        self.hecras_plan_path = hecras_plan_path
        self.hecras_start_index = int(hecras_start_index)
        self.hecras_k = int(hecras_k)
        self._load_velocity_fields(model_dir)
        env_depth, env_depth_array, env_x_coords, env_y_coords, wetted_mask_array = self._resolve_background_sources(
            model_dir=model_dir,
            hecras_plan_path=hecras_plan_path,
            hecras_start_index=self.hecras_start_index,
            hecras_k=self.hecras_k,
        )

        old_widget = self.replay_widget
        positions = self._coerce_replay_positions(getattr(old_widget, "positions", None))
        battery_array = getattr(old_widget, "battery_array", None)
        heading_array = getattr(old_widget, "heading_array", None)
        alive_array = getattr(old_widget, "alive_array", None)

        new_widget = ReplayWidget(
            positions=positions,
            parent=self,
            env_depth=env_depth,
            pad=1.15,
            point_size=4.0,
            env_depth_array=env_depth_array,
            env_x_coords=env_x_coords,
            env_y_coords=env_y_coords,
            wetted_mask_array=wetted_mask_array,
            battery_array=battery_array,
            heading_array=heading_array,
        )
        if alive_array is not None:
            new_widget.alive_array = np.asarray(alive_array)

        if self.vel_x_data is not None:
            new_widget._vel_x_data = self.vel_x_data
            new_widget._vel_y_data = self.vel_y_data
            new_widget._vel_bbox = self.vel_bbox
            new_widget._vel_transform = self.vel_transform

        new_widget.timer.timeout.connect(self._check_animation_complete)

        layout = self.layout()
        if layout is None:
            raise RuntimeError("SimulationCanvas has no layout while updating environment")
        layout.replaceWidget(old_widget, new_widget)
        old_widget.setParent(None)
        old_widget.deleteLater()
        self.replay_widget = new_widget
        
    def _check_animation_complete(self):
        """Check if animation reached the end and emit signal."""
        if self.replay_widget.frame >= self.replay_widget.T - 1:
            # Animation reached the end
            self.replay_widget.playing = False
            if self.replay_widget.timer.isActive():
                self.replay_widget.timer.stop()
            self._emit_animation_finished()

    def _poll_animation_complete(self):
        """Fallback watchdog in case timer callbacks miss the last frame."""
        if self._animation_finished_emitted:
            return
        if self.replay_widget.frame >= self.replay_widget.T - 1 and not self.replay_widget.timer.isActive():
            self._emit_animation_finished()

    def _emit_animation_finished(self):
        """Emit animation_finished once per episode."""
        if self._animation_finished_emitted:
            return
        self._animation_finished_emitted = True
        if self._animation_watchdog.isActive():
            self._animation_watchdog.stop()
        self.animation_finished.emit()
    
    def wheelEvent(self, event):
        """Forward wheel events to replay widget for zooming only."""
        # Forward to replay_widget and prevent default scroll behavior
        self.replay_widget.wheelEvent(event)
        event.accept()
        
    def set_positions(self, positions: np.ndarray, headings: Optional[np.ndarray] = None, battery: Optional[np.ndarray] = None, alive: Optional[np.ndarray] = None):
        """
        Update agent positions for visualization.
        
        Args:
            positions: Array of shape (num_agents, 2) or (num_timesteps, num_agents, 2)
            headings: Optional array of headings for oriented rendering
            battery: Optional array of battery levels for color visualization (green=full, red=depleted)
            alive: Optional boolean array indicating which agents are alive (for dead fish coloring)
        """
        print(f"[RL VIEWER DEBUG] set_positions called: positions shape={positions.shape if positions is not None else None}", flush=True)
        if positions is None or positions.size == 0:
            print("[RL VIEWER DEBUG] Positions empty, returning early", flush=True)
            return
            
        # Ensure positions are 3D (T, N, 2)
        if positions.ndim == 2:
            # Single timestep: (N, 2) -> (1, N, 2)
            positions = positions[np.newaxis, :, :]
        
        # Update ReplayWidget with full trajectory
        self.replay_widget.positions = positions
        self.replay_widget.T, self.replay_widget.N, _ = positions.shape
        
        # Update headings if provided
        if headings is not None:
            if headings.ndim == 1:
                headings = headings[np.newaxis, :]
            self.replay_widget.heading_array = headings
        
        # Update battery if provided (for color visualization)
        if battery is not None:
            if battery.ndim == 1:
                battery = battery[np.newaxis, :]
            self.replay_widget.battery_array = battery
        
        # Update alive status if provided (for dead fish coloring)
        if alive is not None:
            if alive.ndim == 1:
                alive = alive[np.newaxis, :]
            self.replay_widget.alive_array = alive
        
        # Update bounds to include all positions in trajectory
        xs = positions[:, :, 0]
        ys = positions[:, :, 1]
        valid = np.isfinite(xs) & np.isfinite(ys)
        if np.any(valid):
            self.replay_widget.xmin = float(np.nanmin(xs[valid]))
            self.replay_widget.xmax = float(np.nanmax(xs[valid]))
            self.replay_widget.ymin = float(np.nanmin(ys[valid]))
            self.replay_widget.ymax = float(np.nanmax(ys[valid]))
        
        # Start from first frame and auto-play the trajectory
        print(f"[RL VIEWER DEBUG] Starting animation: T={self.replay_widget.T}, N={self.replay_widget.N}, frame=0", flush=True)
        self.replay_widget.frame = 0
        self.replay_widget.playing = True
        self.replay_widget.start()  # Start animation
        self.replay_widget.update()
        self._animation_finished_emitted = False
        if not self._animation_watchdog.isActive():
            self._animation_watchdog.start()
        print(f"[RL VIEWER DEBUG] Animation started, playing={self.replay_widget.playing}", flush=True)


class WeightsPanel(QWidget):
    """
    Left panel: Display behavioral weights and training diagnostics.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Training Reporting")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)
        
        # Diagnostics
        self.diagnostics_group = QGroupBox("Training Diagnostics")
        diag_layout = QVBoxLayout()
        
        self.episode_label = QLabel("Episode: 0 / 0")
        self.reward_label = QLabel("Current Reward: 0.00")
        self.best_reward_label = QLabel("Best Reward: 0.00")
        self.improvement_label = QLabel("Improvement: 0.00")
        
        diag_layout.addWidget(self.episode_label)
        diag_layout.addWidget(self.reward_label)
        diag_layout.addWidget(self.best_reward_label)
        diag_layout.addWidget(self.improvement_label)
        
        self.diagnostics_group.setLayout(diag_layout)
        self.diagnostics_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(self.diagnostics_group)
        
        # Reward components
        self.components_group = QGroupBox("Reward Components")
        components_group_layout = QVBoxLayout()
        self.components_scroll = QScrollArea()
        self.components_scroll.setWidgetResizable(True)
        self.components_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.components_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.components_scroll.setMinimumHeight(120)
        self.components_scroll.setMaximumHeight(180)
        self.components_scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        self.components_container = QWidget()
        self.components_layout = QVBoxLayout(self.components_container)
        self.components_layout.setContentsMargins(0, 0, 0, 0)
        self.components_layout.setSpacing(2)
        self.components_scroll.setWidget(self.components_container)

        self.component_labels = {}
        components_group_layout.addWidget(self.components_scroll)
        self.components_group.setLayout(components_group_layout)
        self.components_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        # Training progress figure belongs with the rest of monitoring widgets.
        self.plot_group = QGroupBox("Training Progress")
        plot_layout = QVBoxLayout()
        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            self.figure = Figure(figsize=(5, 3), dpi=80)
            self.canvas = FigureCanvasQTAgg(self.figure)
            self.canvas.setMinimumSize(200, 210)

            try:
                callbacks = self.canvas.callbacks.callbacks.get("scroll_event", {})
                for cid in list(callbacks.keys()):
                    self.canvas.mpl_disconnect(cid)
            except Exception:
                pass

            self.ax = self.figure.add_subplot(111)
            self.ax.set_xlabel("Episode")
            self.ax.set_ylabel("Reward")
            self.ax.set_title("RL Training Progress")
            self.ax.grid(True, alpha=0.3)
            self.figure.tight_layout()
            plot_layout.addWidget(self.canvas)
            self.has_plot = True
        except ImportError:
            self.plot_fallback = QTextEdit()
            self.plot_fallback.setReadOnly(True)
            self.plot_fallback.setMaximumHeight(180)
            self.plot_fallback.setFont(QFont("Courier New", 8))
            self.plot_fallback.setPlaceholderText("Install matplotlib to show training progress figure.")
            plot_layout.addWidget(self.plot_fallback)
            self.has_plot = False
        self.plot_group.setLayout(plot_layout)
        self.plot_group.setMinimumHeight(250)
        self.plot_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(self.plot_group)
        layout.addWidget(self.components_group)

        self.episode_history = []
        self.reward_history = []

        # Bottom-left background status log for long-running simulation visibility
        self.background_log_group = QGroupBox("Background Simulation Log")
        background_log_layout = QVBoxLayout()
        self.background_log_text = QTextEdit()
        self.background_log_text.setReadOnly(True)
        self.background_log_text.setMinimumHeight(140)
        self.background_log_text.setMaximumHeight(160)
        self.background_log_text.setFont(QFont("Courier New", 8))
        background_log_layout.addWidget(self.background_log_text)
        self.background_log_group.setLayout(background_log_layout)
        self.background_log_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(self.background_log_group)

        self.advanced_container = QWidget()
        advanced_layout = QVBoxLayout(self.advanced_container)
        advanced_layout.setContentsMargins(0, 0, 0, 0)

        # Weights display (editable spin boxes)
        self.weights_group = QGroupBox("Current Weights (Editable)")
        self.weights_layout = QVBoxLayout()
        self.weight_spinboxes = {}
        self.weights_group.setLayout(self.weights_layout)
        advanced_layout.addWidget(self.weights_group)

        # Arbitration order display
        self.order_group = QGroupBox("Cue Application Order")
        self.order_layout = QVBoxLayout()
        self.order_labels = {}
        self.order_group.setLayout(self.order_layout)
        advanced_layout.addWidget(self.order_group)

        layout.addWidget(self.advanced_container)
        self.set_advanced_visible(False)
        
        layout.addStretch()
        self.setLayout(layout)

    def set_advanced_visible(self, visible: bool):
        """Show/hide heavy tuning widgets to keep laptop layouts uncluttered."""
        is_visible = bool(visible)
        self.advanced_container.setVisible(is_visible)

    def is_advanced_visible(self) -> bool:
        return bool(self.advanced_container.isVisible())
        
    def update_weights(self, weights: BehavioralWeights):
        """Update displayed weights with editable spin boxes."""
        print(f"[UPDATE_WEIGHTS DEBUG] Called with weights, shallow_weight={weights.shallow_weight}, cohesion_weight={weights.cohesion_weight}", flush=True)
        
        # Clear ALL existing widgets and layouts properly
        # First, delete all child widgets and layouts
        while self.weights_layout.count():
            item = self.weights_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                # Clear nested layout
                while item.layout().count():
                    child = item.layout().takeAt(0)
                    if child.widget():
                        child.widget().deleteLater()
                item.layout().deleteLater()
        
        self.weight_spinboxes.clear()
        
        # Add new weight spin boxes
        weights_dict = weights.to_dict()
        print(f"[UPDATE_WEIGHTS DEBUG] weights_dict has {len(weights_dict)} items", flush=True)
        
        # Weight attributes to display (skip order fields and non-weight params)
        weight_keys = [
            'shallow_weight', 'border_cue_weight', 'avoid_weight', 'collision_weight',
            'alignment_weight', 'cohesion_weight', 'low_speed_weight', 'refugia_weight',
            'rheotaxis_weight', 'wave_drag_weight', 'arbitration_tolerance'
        ]
        
        for name in weight_keys:
            if name not in weights_dict:
                continue
                
            value = weights_dict[name]
            
            # Create horizontal layout for label + spinbox
            row_layout = QHBoxLayout()
            
            # Label
            label_text = name.replace('_weight', '').replace('_', ' ').title()
            label = QLabel(f"{label_text}:")
            label.setMinimumWidth(150)
            row_layout.addWidget(label)
            
            # Spin box
            spinbox = QDoubleSpinBox()
            spinbox.setRange(0, 1000000)
            spinbox.setDecimals(1)
            spinbox.setValue(float(value))
            spinbox.setSingleStep(100.0 if name == 'arbitration_tolerance' else 1000.0)
            spinbox.setObjectName(name)  # Store attribute name
            
            # Connect to update function
            spinbox.valueChanged.connect(lambda v, attr=name: self._on_weight_changed(attr, v))
            
            row_layout.addWidget(spinbox)
            self.weights_layout.addLayout(row_layout)
            self.weight_spinboxes[name] = spinbox
            print(f"[UPDATE_WEIGHTS DEBUG] Added spinbox: {name}={value}", flush=True)
    
    def _on_weight_changed(self, attr_name: str, value: float):
        """Called when a weight spinbox value changes."""
        print(f"[WEIGHT CHANGED] {attr_name} = {value}", flush=True)
        # The parent window will handle applying these changes to current_weights
    
    def get_edited_weights(self, base_weights: BehavioralWeights) -> BehavioralWeights:
        """Get a new BehavioralWeights object with values from the spin boxes."""
        weights_dict = base_weights.to_dict()
        
        # Update with edited values from spin boxes
        for name, spinbox in self.weight_spinboxes.items():
            weights_dict[name] = spinbox.value()
        
        # Create new weights object from updated dict
        from emergent.salmon_abm.rl_training import BehavioralWeights
        return BehavioralWeights.from_dict(weights_dict)
    
    def update_diagnostics(self, episode: int, total_episodes: int, reward: float, 
                          best_reward: float, initial_reward: float):
        """Update training diagnostics."""
        self.episode_label.setText(f"Episode: {episode} / {total_episodes}")
        self.reward_label.setText(f"Current Reward: {reward:.2f}")
        self.best_reward_label.setText(f"Best Reward: {best_reward:.2f}")
        improvement = reward - initial_reward
        self.improvement_label.setText(f"Improvement: {improvement:+.2f}")
    
    def clear_diagnostics(self):
        """Clear all diagnostic displays."""
        self.episode_label.setText("Episode: 0 / 0")
        self.reward_label.setText("Current Reward: 0.00")
        self.best_reward_label.setText("Best Reward: 0.00")
        self.improvement_label.setText("Improvement: +0.00")
        
        # Clear component labels
        for label in self.component_labels.values():
            self.components_layout.removeWidget(label)
            label.deleteLater()
        self.component_labels.clear()
        
    def update_components(self, components: Dict[str, float]):
        """Update reward component breakdown."""
        # Clear existing labels
        for label in self.component_labels.values():
            self.components_layout.removeWidget(label)
            label.deleteLater()
        self.component_labels.clear()

        component_order = [
            "cohesion",
            "alignment",
            "separation",
            "upstream_progress",
            "energy_efficiency",
            "drafting_benefit",
            "boundary_penalty",
            "mortality_penalty",
            "smoothness_penalty",
            "fatigue_penalty",
            "stagnation_penalty",
            "rheotaxis_alignment",
            "min_schooling_penalty",
            "weight_diversity_bonus",
            "total",
        ]

        # Add component labels in stable order so penalties are always visible.
        for name in component_order:
            value = float(components.get(name, 0.0))
            label = QLabel(f"{name}: {value:+.2f}")
            label.setFont(QFont("Courier New", 9))
            self.components_layout.addWidget(label)
            self.component_labels[name] = label

    def append_background_log(self, message: str):
        """Append one line to the background simulation log."""
        self.background_log_text.append(message)
        scrollbar = self.background_log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def clear_background_log(self):
        """Clear the background simulation log."""
        self.background_log_text.clear()

    def update_plot(self, episode: int, reward: float):
        """Update the left-panel training progress plot."""
        if not self.has_plot:
            return

        self.episode_history.append(episode)
        self.reward_history.append(reward)

        self.ax.clear()
        self.ax.plot(self.episode_history, self.reward_history, "b-", linewidth=2, label="Episode Reward")
        if self.reward_history:
            best_idx = int(np.argmax(self.reward_history))
            self.ax.plot(
                self.episode_history[best_idx],
                self.reward_history[best_idx],
                "r*",
                markersize=12,
                label="Best",
            )

        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Reward")
        self.ax.set_title("RL Training Progress")
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw()

    def clear_plot(self):
        """Reset the left-panel training progress plot."""
        self.episode_history = []
        self.reward_history = []
        if not self.has_plot:
            return
        self.ax.clear()
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Reward")
        self.ax.set_title("RL Training Progress")
        self.ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw()
    
    def update_order(self, order_dict: Dict[int, str], weights: Optional['BehavioralWeights'] = None):
        """Update displayed arbitration order with optional remapping.
        
        Args:
            order_dict: Default mapping of position -> cue_name
            weights: Optional BehavioralWeights with order_0-9 fields showing new positions
        """
        # Clear existing labels
        for label in self.order_labels.values():
            self.order_layout.removeWidget(label)
            label.deleteLater()
        self.order_labels.clear()
        
        # Get remapping if weights provided
        if weights is not None:
            # Extract order remapping from weights
            new_positions = [
                weights.order_0, weights.order_1, weights.order_2, weights.order_3, weights.order_4,
                weights.order_5, weights.order_6, weights.order_7, weights.order_8, weights.order_9
            ]
        else:
            new_positions = None
        
        # Add order labels
        for position, cue_name in sorted(order_dict.items()):
            if new_positions is not None:
                new_pos = new_positions[position]
                if new_pos != position:
                    label_text = f"{position}: {cue_name} → {new_pos}"
                else:
                    label_text = f"{position}: {cue_name}"
            else:
                label_text = f"{position}: {cue_name}"
            
            label = QLabel(label_text)
            label.setFont(QFont("Courier New", 9))
            self.order_layout.addWidget(label)
            self.order_labels[position] = label


class ControlPanel(QWidget):
    """
    Right panel: Training controls and settings.
    """
    
    # Signals
    start_training = pyqtSignal()
    pause_training = pyqtSignal()
    stop_training = pyqtSignal()
    randomize_weights = pyqtSignal()
    set_blanket_value = pyqtSignal()  # NEW: Set all weights to uniform value
    randomize_order = pyqtSignal()
    reset_training = pyqtSignal()  # NEW: Reset to initial state
    episode_selected = pyqtSignal(int)  # NEW: User selected episode to replay
    next_episode = pyqtSignal()  # NEW: Show next episode
    prev_episode = pyqtSignal()  # NEW: Show previous episode
    toggle_playback = pyqtSignal()  # Play/pause animation
    restart_animation = pyqtSignal()  # Restart current episode animation
    start_recording = pyqtSignal()  # Start snippet recording
    stop_recording = pyqtSignal()  # Stop snippet recording and save clip
    open_archive = pyqtSignal()  # Open an existing RL archive HDF5
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Training Controls")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)

        # Input configuration
        self.input_group = QGroupBox("Input Files")
        input_layout = QGridLayout()

        input_layout.addWidget(QLabel("Model Dir:"), 0, 0)
        self.model_dir_input = QLineEdit()
        self.model_dir_input.setPlaceholderText("Optional (used for raster mode and defaults)")
        input_layout.addWidget(self.model_dir_input, 0, 1)
        self.btn_model_dir = QPushButton("...")
        self.btn_model_dir.setMaximumWidth(40)
        self.btn_model_dir.clicked.connect(self._browse_model_dir)
        input_layout.addWidget(self.btn_model_dir, 0, 2)

        input_layout.addWidget(QLabel("HECRAS HDF:"), 1, 0)
        self.hecras_plan_input = QLineEdit()
        self.hecras_plan_input.setPlaceholderText("Pick HECRAS plan .hdf/.h5")
        input_layout.addWidget(self.hecras_plan_input, 1, 1)
        self.btn_hecras_plan = QPushButton("...")
        self.btn_hecras_plan.setMaximumWidth(40)
        self.btn_hecras_plan.clicked.connect(self._browse_hecras_plan)
        input_layout.addWidget(self.btn_hecras_plan, 1, 2)

        input_layout.addWidget(QLabel("Start Polygon:"), 2, 0)
        self.start_polygon_input = QLineEdit()
        self.start_polygon_input.setPlaceholderText("Required: starting polygon .shp")
        input_layout.addWidget(self.start_polygon_input, 2, 1)
        self.btn_start_polygon = QPushButton("...")
        self.btn_start_polygon.setMaximumWidth(40)
        self.btn_start_polygon.clicked.connect(self._browse_start_polygon)
        input_layout.addWidget(self.btn_start_polygon, 2, 2)

        input_layout.addWidget(QLabel("Longitudinal:"), 3, 0)
        self.longitudinal_input = QLineEdit()
        self.longitudinal_input.setPlaceholderText("Required for RL reward: longitudinal.shp")
        input_layout.addWidget(self.longitudinal_input, 3, 1)
        self.btn_longitudinal = QPushButton("...")
        self.btn_longitudinal.setMaximumWidth(40)
        self.btn_longitudinal.clicked.connect(self._browse_longitudinal)
        input_layout.addWidget(self.btn_longitudinal, 3, 2)

        self.input_group.setLayout(input_layout)
        layout.addWidget(self.input_group)
        
        # Training control buttons
        train_group = QGroupBox("Training")
        train_layout = QVBoxLayout()
        
        self.btn_start = QPushButton("▶ Start Training")
        self.btn_pause = QPushButton("⏸ Pause Training")
        self.btn_stop = QPushButton("⏹ Stop Training")
        
        self.btn_start.clicked.connect(self.start_training.emit)
        self.btn_pause.clicked.connect(self.pause_training.emit)
        self.btn_stop.clicked.connect(self.stop_training.emit)
        
        train_layout.addWidget(self.btn_start)
        train_layout.addWidget(self.btn_pause)
        train_layout.addWidget(self.btn_stop)
        train_group.setLayout(train_layout)
        layout.addWidget(train_group)
        
        # Animation playback controls
        self.playback_group = QGroupBox("Animation Playback")
        playback_layout = QHBoxLayout()
        
        self.btn_play_pause = QPushButton("⏸ Pause")
        self.btn_play_pause.setToolTip("Play/Pause current episode animation")
        self.btn_play_pause.clicked.connect(self.toggle_playback)
        
        self.btn_restart_anim = QPushButton("↺ Restart")
        self.btn_restart_anim.setToolTip("Restart current episode from beginning")
        self.btn_restart_anim.clicked.connect(self.restart_animation)
        
        playback_layout.addWidget(self.btn_play_pause)
        playback_layout.addWidget(self.btn_restart_anim)
        self.playback_group.setLayout(playback_layout)
        layout.addWidget(self.playback_group)

        # On-the-fly clip recording
        self.recording_group = QGroupBox("Recording")
        recording_layout = QVBoxLayout()

        recording_buttons = QHBoxLayout()
        self.btn_record_start = QPushButton("● Start Recording")
        self.btn_record_start.setToolTip("Capture a clip from the center simulation viewer")
        self.btn_record_start.clicked.connect(self.start_recording.emit)
        recording_buttons.addWidget(self.btn_record_start)

        self.btn_record_stop = QPushButton("■ Stop + Save Clip")
        self.btn_record_stop.setToolTip("Stop capture and save an animated GIF clip")
        self.btn_record_stop.setEnabled(False)
        self.btn_record_stop.clicked.connect(self.stop_recording.emit)
        recording_buttons.addWidget(self.btn_record_stop)
        recording_layout.addLayout(recording_buttons)

        self.recording_status_label = QLabel("Idle")
        recording_layout.addWidget(self.recording_status_label)

        self.recording_group.setLayout(recording_layout)
        layout.addWidget(self.recording_group)
        
        # Episode navigation
        self.nav_group = QGroupBox("Episode Navigation")
        nav_layout = QVBoxLayout()
        
        # Dropdown + buttons row
        nav_controls = QHBoxLayout()
        
        self.btn_prev_episode = QPushButton("◀")
        self.btn_prev_episode.setMaximumWidth(40)
        self.btn_prev_episode.setToolTip("Show previous episode")
        self.btn_prev_episode.clicked.connect(self.prev_episode.emit)
        nav_controls.addWidget(self.btn_prev_episode)
        
        self.episode_combo = QComboBox()
        self.episode_combo.setToolTip("Select episode to replay")
        self.episode_combo.currentIndexChanged.connect(self._on_episode_combo_changed)
        nav_controls.addWidget(self.episode_combo)
        
        self.btn_next_episode = QPushButton("▶")
        self.btn_next_episode.setMaximumWidth(40)
        self.btn_next_episode.setToolTip("Show next episode")
        self.btn_next_episode.clicked.connect(self.next_episode.emit)
        nav_controls.addWidget(self.btn_next_episode)
        
        nav_layout.addLayout(nav_controls)
        self.btn_open_archive = QPushButton("Open Archive...")
        self.btn_open_archive.setToolTip("Open existing RL archive HDF5 for generation replay")
        self.btn_open_archive.clicked.connect(self.open_archive.emit)
        nav_layout.addWidget(self.btn_open_archive)
        self.nav_group.setLayout(nav_layout)
        layout.addWidget(self.nav_group)
        
        # Training parameters
        params_group = QGroupBox("Training Parameters")
        params_layout = QGridLayout()
        
        # Episodes
        params_layout.addWidget(QLabel("Episodes:"), 0, 0)
        self.episodes_spin = QSpinBox()
        self.episodes_spin.setRange(1, 100000)
        self.episodes_spin.setValue(50)
        self.episodes_spin.setToolTip("Number of training episodes to run. Each episode tests one set of behavioral weights.")
        params_layout.addWidget(self.episodes_spin, 0, 1)
        
        # Timesteps per episode
        params_layout.addWidget(QLabel("Timesteps:"), 1, 0)
        self.timesteps_spin = QSpinBox()
        self.timesteps_spin.setRange(10, 100000)
        self.timesteps_spin.setValue(100)
        self.timesteps_spin.setToolTip("Number of simulation timesteps per episode. Large values increase compute and memory.")
        params_layout.addWidget(self.timesteps_spin, 1, 1)
        
        # Number of agents
        params_layout.addWidget(QLabel("Agents:"), 2, 0)
        self.agents_spin = QSpinBox()
        self.agents_spin.setRange(10, 50000)
        self.agents_spin.setValue(200)
        self.agents_spin.setToolTip("Number of fish agents in the simulation. Large values can be slow and memory-heavy.")
        params_layout.addWidget(self.agents_spin, 2, 1)
        
        # Exploration noise
        params_layout.addWidget(QLabel("Exploration:"), 3, 0)
        self.noise_spin = QDoubleSpinBox()
        self.noise_spin.setRange(0.01, 1.0)
        self.noise_spin.setSingleStep(0.01)
        self.noise_spin.setValue(0.1)
        self.noise_spin.setToolTip("Mutation scale for exploring new behavioral weights (0.1 = 10% random variation). Higher values = more exploration, lower = more exploitation of good weights.")
        params_layout.addWidget(self.noise_spin, 3, 1)
        
        # Storage interval (memory optimization)
        params_layout.addWidget(QLabel("Store every Nth:"), 4, 0)
        self.storage_interval_spin = QSpinBox()
        self.storage_interval_spin.setRange(1, 10000)
        self.storage_interval_spin.setValue(10)
        self.storage_interval_spin.setToolTip("Store trajectory data for replay every N episodes (plus first 5 and best). 1=all episodes, 10=every 10th. Saves memory for long training runs.")
        params_layout.addWidget(self.storage_interval_spin, 4, 1)

        # Persist all generations so replay can lazy-load from disk
        self.archive_hdf_check = QCheckBox("Archive episodes to HDF")
        self.archive_hdf_check.setChecked(True)
        self.archive_hdf_check.setToolTip("Write every episode to an HDF archive for full generation replay")
        params_layout.addWidget(self.archive_hdf_check, 5, 0, 1, 2)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Blanket value controls (above randomize for better flow)
        self.blanket_controls = QWidget()
        blanket_layout = QHBoxLayout(self.blanket_controls)
        blanket_layout.setContentsMargins(0, 0, 0, 0)
        self.blanket_value_input = QLineEdit("10.0")
        self.blanket_value_input.setMaximumWidth(60)
        self.blanket_value_input.setToolTip("Value to set for all behavioral weights")
        self.blanket_value_label = QLabel("Blanket value:")
        blanket_layout.addWidget(self.blanket_value_label)
        blanket_layout.addWidget(self.blanket_value_input)
        
        self.btn_set_blanket = QPushButton("📋 Set All Weights")
        self.btn_set_blanket.clicked.connect(self.set_blanket_value.emit)
        self.btn_set_blanket.setToolTip("Set all behavioral weights to the specified blanket value. Only works before training starts.")
        blanket_layout.addWidget(self.btn_set_blanket)
        blanket_layout.addStretch()
        
        layout.addWidget(self.blanket_controls)
        
        # Reward weights section (SEPARATE from behavioral weights - controls scoring, not training)
        self.reward_group = QGroupBox("Reward Weights (Objective Function)")
        reward_layout = QGridLayout()

        def _new_reward_spin(min_v: float, max_v: float, value: float, decimals: int, tooltip: str) -> QDoubleSpinBox:
            spin = QDoubleSpinBox()
            spin.setRange(min_v, max_v)
            spin.setValue(value)
            spin.setDecimals(decimals)
            spin.setToolTip(tooltip)
            spin.setMaximumWidth(110)
            return spin

        row = 0
        reward_layout.addWidget(QLabel("<b>Positive:</b>"), row, 0, 1, 2)
        row += 1

        reward_layout.addWidget(QLabel("Upstream:"), row, 0)
        self.upstream_weight_spin = _new_reward_spin(
            0, 100, 10.0, 2, "Multiplier for meters traveled upstream (PRIMARY)"
        )
        reward_layout.addWidget(self.upstream_weight_spin, row, 1)
        row += 1

        reward_layout.addWidget(QLabel("Cohesion:"), row, 0)
        self.cohesion_reward_spin = _new_reward_spin(
            0, 100, 0.1, 4, "Multiplier for cohesion score sum"
        )
        reward_layout.addWidget(self.cohesion_reward_spin, row, 1)
        row += 1

        reward_layout.addWidget(QLabel("Alignment:"), row, 0)
        self.alignment_reward_spin = _new_reward_spin(
            0, 100, 1.0, 4, "Multiplier for alignment score sum"
        )
        reward_layout.addWidget(self.alignment_reward_spin, row, 1)
        row += 1

        reward_layout.addWidget(QLabel("Energy eff:"), row, 0)
        self.energy_reward_spin = _new_reward_spin(
            0, 100, 2.0, 2, "Multiplier for distance/speed² ratio"
        )
        reward_layout.addWidget(self.energy_reward_spin, row, 1)
        row += 1

        reward_layout.addWidget(QLabel("Drafting:"), row, 0)
        self.drafting_reward_spin = _new_reward_spin(
            0, 100, 20.0, 1, "Multiplier for formation benefits (disabled)"
        )
        reward_layout.addWidget(self.drafting_reward_spin, row, 1)
        row += 1

        reward_layout.addWidget(QLabel("<b>Penalties:</b>"), row, 0, 1, 2)
        row += 1

        reward_layout.addWidget(QLabel("Rheotaxis:"), row, 0)
        self.rheotaxis_penalty_spin = _new_reward_spin(
            -100, 100, -100.0, 2, "Penalty for facing wrong direction"
        )
        reward_layout.addWidget(self.rheotaxis_penalty_spin, row, 1)
        row += 1

        reward_layout.addWidget(QLabel("Separation:"), row, 0)
        self.separation_penalty_spin = _new_reward_spin(
            -100, 100, -0.2, 4, "Penalty for crowding violations"
        )
        reward_layout.addWidget(self.separation_penalty_spin, row, 1)
        row += 1

        reward_layout.addWidget(QLabel("Mortality:"), row, 0)
        self.mortality_penalty_spin = _new_reward_spin(
            -100, 100, -50.0, 1, "Penalty per fish death"
        )
        reward_layout.addWidget(self.mortality_penalty_spin, row, 1)
        row += 1

        reward_layout.addWidget(QLabel("Fatigue:"), row, 0)
        self.fatigue_penalty_spin = _new_reward_spin(
            -100, 100, -0.9, 3, "Penalty for low battery timesteps"
        )
        reward_layout.addWidget(self.fatigue_penalty_spin, row, 1)
        row += 1

        reward_layout.addWidget(QLabel("Stagnation:"), row, 0)
        self.stagnation_penalty_spin = _new_reward_spin(
            -100, 100, -0.9, 3, "Penalty for stationary timesteps"
        )
        reward_layout.addWidget(self.stagnation_penalty_spin, row, 1)
        row += 1

        reward_layout.addWidget(QLabel("<b>Other:</b>"), row, 0, 1, 2)
        row += 1

        reward_layout.addWidget(QLabel("Smoothness:"), row, 0)
        self.smoothness_penalty_spin = _new_reward_spin(
            -100, 100, -0.2, 4, "Penalty for jerky movement (use negative values)"
        )
        reward_layout.addWidget(self.smoothness_penalty_spin, row, 1)
        row += 1

        reward_layout.addWidget(QLabel("Boundary:"), row, 0)
        self.boundary_penalty_spin = _new_reward_spin(
            -100, 100, -10.0, 2, "Penalty for boundary proximity (disabled)"
        )
        reward_layout.addWidget(self.boundary_penalty_spin, row, 1)
        reward_layout.setColumnStretch(0, 1)
        reward_layout.setColumnStretch(1, 0)
        
        self.reward_group.setLayout(reward_layout)
        self.reward_group.setToolTip("These weights define WHAT YOU VALUE (objective function). They are NOT trained by RL.")
        layout.addWidget(self.reward_group)
        
        # Randomize buttons
        self.btn_randomize = QPushButton("🎲 Randomize Weights")
        self.btn_randomize.clicked.connect(self.randomize_weights.emit)
        self.btn_randomize.setToolTip("Generate new random initial weights (100% variation from defaults). Only works before training starts.")
        layout.addWidget(self.btn_randomize)
        
        self.btn_randomize_order = QPushButton("🎪 Randomize Cue Order")
        self.btn_randomize_order.clicked.connect(self.randomize_order.emit)
        self.btn_randomize_order.setToolTip("Shuffle behavioral cue application order. Order stays fixed throughout all episodes. Only works before training starts.")
        layout.addWidget(self.btn_randomize_order)
        
        # Reset button
        self.btn_reset = QPushButton("🔄 Reset Training")
        self.btn_reset.clicked.connect(self.reset_training.emit)
        self.btn_reset.setToolTip("Reset to initial state (clears training history, keeps current weights). Start fresh without restarting the app.")
        layout.addWidget(self.btn_reset)
        
        # Progress
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready")
        progress_layout.addWidget(self.status_label)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        log_group = QGroupBox("Training Log")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        self.log_text.setFont(QFont("Courier New", 8))
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        self.set_replay_controls_visible(False)
        self.set_advanced_controls_visible(False)
        self.refresh_layout_height_hint()
        
        layout.addStretch()
        self.setLayout(layout)

    def set_replay_controls_visible(self, visible: bool):
        is_visible = bool(visible)
        self.playback_group.setVisible(is_visible)
        self.recording_group.setVisible(is_visible)
        self.nav_group.setVisible(is_visible)
        self.refresh_layout_height_hint()

    def is_replay_controls_visible(self) -> bool:
        return bool(self.playback_group.isVisible())

    def set_advanced_controls_visible(self, visible: bool):
        is_visible = bool(visible)
        self.blanket_controls.setVisible(is_visible)
        self.reward_group.setVisible(is_visible)
        self.btn_randomize.setVisible(is_visible)
        self.btn_randomize_order.setVisible(is_visible)
        self.refresh_layout_height_hint()

    def is_advanced_controls_visible(self) -> bool:
        return bool(self.reward_group.isVisible())

    def refresh_layout_height_hint(self):
        """Force a content-based minimum height so scrollbars appear when needed."""
        self.adjustSize()
        self.setMinimumHeight(self.sizeHint().height())
    
    def _on_episode_combo_changed(self, index):
        """Handle episode selection from dropdown."""
        if index >= 0:
            self.episode_selected.emit(index)

    def _browse_model_dir(self):
        start_dir = self.model_dir_input.text().strip() or os.getcwd()
        path = QFileDialog.getExistingDirectory(self, "Select Model Directory", start_dir)
        if path:
            self.model_dir_input.setText(path)
            self._autofill_common_paths(path)

    def _browse_hecras_plan(self):
        start_dir = self.model_dir_input.text().strip() or os.getcwd()
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select HECRAS Plan HDF",
            start_dir,
            "HECRAS/HDF5 (*.hdf *.h5 *.hdf5);;All Files (*)",
        )
        if path:
            self.hecras_plan_input.setText(path)
            plan_dir = os.path.dirname(path)
            if not self.model_dir_input.text().strip():
                self.model_dir_input.setText(plan_dir)
            self._autofill_common_paths(plan_dir)

    def _browse_start_polygon(self):
        start_dir = self.model_dir_input.text().strip() or os.getcwd()
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Start Polygon",
            start_dir,
            "Shapefile (*.shp);;All Files (*)",
        )
        if path:
            self.start_polygon_input.setText(path)

    def _browse_longitudinal(self):
        start_dir = self.model_dir_input.text().strip() or os.getcwd()
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Longitudinal Profile",
            start_dir,
            "Shapefile (*.shp);;All Files (*)",
        )
        if path:
            self.longitudinal_input.setText(path)

    def _autofill_common_paths(self, base_dir: str):
        if not base_dir:
            return
        base = os.path.abspath(base_dir)
        candidates = [
            ("start_polygon_input", os.path.join(base, "start_loc_river_right.shp")),
            ("start_polygon_input", os.path.join(base, "shapes", "start_loc_river_right.shp")),
            ("longitudinal_input", os.path.join(base, "longitudinal.shp")),
            ("longitudinal_input", os.path.join(base, "shapes", "longitudinal.shp")),
        ]
        for attr_name, candidate in candidates:
            if os.path.exists(candidate):
                field = getattr(self, attr_name, None)
                if field is not None and not field.text().strip():
                    field.setText(candidate)

    def set_input_paths(
        self,
        model_dir: Optional[str] = None,
        hecras_plan: Optional[str] = None,
        start_polygon: Optional[str] = None,
        longitudinal_profile: Optional[str] = None,
    ):
        if model_dir:
            self.model_dir_input.setText(str(model_dir))
        if hecras_plan:
            self.hecras_plan_input.setText(str(hecras_plan))
        if start_polygon:
            self.start_polygon_input.setText(str(start_polygon))
        if longitudinal_profile:
            self.longitudinal_input.setText(str(longitudinal_profile))
        if model_dir:
            self._autofill_common_paths(str(model_dir))

    def get_model_dir(self) -> Optional[str]:
        value = self.model_dir_input.text().strip()
        return value or None

    def get_hecras_plan(self) -> Optional[str]:
        value = self.hecras_plan_input.text().strip()
        return value or None

    def get_start_polygon(self) -> Optional[str]:
        value = self.start_polygon_input.text().strip()
        return value or None

    def get_longitudinal_profile(self) -> Optional[str]:
        value = self.longitudinal_input.text().strip()
        return value or None
    
    def add_episode_to_list(self, episode_num: int):
        """Add completed episode to navigation dropdown."""
        for i in range(self.episode_combo.count()):
            data = self.episode_combo.itemData(i)
            if data is not None and int(data) == int(episode_num):
                return
        self.episode_combo.addItem(f"Episode {episode_num + 1}", int(episode_num))
        self.episode_combo.setCurrentIndex(self.episode_combo.count() - 1)

    def clear_episode_list(self):
        """Clear replay episode navigation list."""
        self.episode_combo.clear()

    def archive_enabled(self) -> bool:
        """Return whether HDF archiving is enabled."""
        return bool(self.archive_hdf_check.isChecked())
    
    def append_log(self, message: str):
        """Append message to training log."""
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
        
    def set_progress(self, current: int, total: int):
        """Update progress bar."""
        if total > 0:
            percent = int((current / total) * 100)
            self.progress_bar.setValue(percent)
        else:
            self.progress_bar.setValue(0)

    def set_recording_state(self, recording: bool, frame_count: int = 0):
        """Update recording controls to reflect active/inactive state."""
        self.btn_record_start.setEnabled(not recording)
        self.btn_record_stop.setEnabled(recording)
        if recording:
            self.recording_status_label.setText(f"Recording... {frame_count} frames")
        else:
            self.recording_status_label.setText("Idle")
    
    def set_parameters_enabled(self, enabled: bool):
        """Enable or disable parameter spinboxes."""
        self.input_group.setEnabled(enabled)
        self.episodes_spin.setEnabled(enabled)
        self.timesteps_spin.setEnabled(enabled)
        self.agents_spin.setEnabled(enabled)
        self.noise_spin.setEnabled(enabled)
        self.storage_interval_spin.setEnabled(enabled)
        self.archive_hdf_check.setEnabled(enabled)
        self.btn_randomize.setEnabled(enabled)
        self.btn_randomize_order.setEnabled(enabled)
        self.btn_set_blanket.setEnabled(enabled)


class TrainingWorker(QObject):
    """
    Background worker for running RL training without blocking UI.
    """
    
    # Signals
    episode_started = pyqtSignal(int)  # episode number
    episode_computed = pyqtSignal(int, float, dict, object, object, object, object, object)  # episode, reward, components, positions, headings, battery, alive, weights
    training_completed = pyqtSignal(object, list)  # best_weights, history
    error_occurred = pyqtSignal(str)  # error message
    
    def __init__(self, trainer, num_episodes: int):
        super().__init__()
        self.trainer = trainer
        self.num_episodes = num_episodes
        self.is_stopped = False
        self.is_paused = False
        
        # Synchronization: wait for animation to complete
        self.animation_complete = threading.Event()
        self.animation_complete.set()  # Initially ready
        
        # Cache longitudinal profile (create sim once to get it)
        print("Caching longitudinal profile from simulation...", flush=True)
        self.longitudinal_profile = None
        try:
            sim = self.trainer.simulation_factory(self.trainer.initial_weights)
            self.longitudinal_profile = getattr(sim, 'longitudinal', None)
            print(f"Simulation created, checking longitudinal attribute...", flush=True)
            if self.longitudinal_profile is not None:
                print(f"Longitudinal profile cached: {type(self.longitudinal_profile)}", flush=True)
            else:
                print(f"WARNING: sim.longitudinal is None", flush=True)
            sim.close()
        except Exception as e:
            import traceback
            error_msg = f"ERROR caching longitudinal profile: {e}\n{traceback.format_exc()}"
            print(error_msg, flush=True)
            raise RuntimeError(f"Failed to cache longitudinal profile: {e}") from e
        
        if self.longitudinal_profile is None:
            raise ValueError(
                "Longitudinal profile not loaded from simulation.\n"
                "Check that longitudinal_profile path is correct and shapefile loads properly.\n"
                "Simulation may have failed to import the shapefile."
            )
    

    def run(self):
        """Execute training loop with UI updates."""
        try:
            current_weights = self.trainer.initial_weights
            
            for episode in range(self.num_episodes):
                # Check for stop signal
                if self.is_stopped:
                    break
                    
                # Check for pause signal
                while self.is_paused and not self.is_stopped:
                    time.sleep(0.1)
                    
                if self.is_stopped:
                    break
                
                self.episode_started.emit(episode)
                
                # Run episode - this executes ALL timesteps before returning
                positions, headings, velocities, battery, alive, velocity_field = self.trainer.run_episode(current_weights)
                
                # Log completion with actual timestep count
                actual_timesteps = positions.shape[0]
                print(f"Episode {episode}: Completed {actual_timesteps} timesteps")
                
                # Compute reward (use flow vector integration for upstream progress)
                from emergent.salmon_abm.rl_training import compute_episode_reward
                # Pass current weights to reward function for constraint checking
                reward, components = compute_episode_reward(
                    positions, headings, velocities, alive,
                    body_length=self.trainer.body_length,
                    threat_level=current_weights.threat_level,
                    behavioral_weights=current_weights.to_dict(),
                    battery_history=battery,
                    longitudinal_profile=self.longitudinal_profile,
                    velocity_field_history=velocity_field  # Flow integration for braided channels
                )
                
                # Track best
                if reward > self.trainer.best_reward:
                    self.trainer.best_reward = reward
                    self.trainer.best_weights = current_weights
                    
                # Store history
                self.trainer.episode_history.append((episode, float(reward)))
                
                # Emit episode data immediately - viewer will queue if busy
                # NO WAITING - next episode computes while current animates!
                self.episode_computed.emit(episode, float(reward), components, positions, headings, battery, alive, current_weights)
                
                # Mutate for next episode (compute in parallel with visualization)
                current_weights = self.trainer.best_weights.mutate(
                    mutation_scale=self.trainer.exploration_noise
                )
            
            # Training complete
            self.training_completed.emit(self.trainer.best_weights, self.trainer.episode_history)
            
        except Exception as e:
            import traceback
            self.error_occurred.emit(f"Training error: {str(e)}\n{traceback.format_exc()}")
    
    def pause(self):
        """Pause training."""
        self.is_paused = True
        
    def resume(self):
        """Resume training."""
        self.is_paused = False
        
    def stop(self):
        """Stop training."""
        self.is_stopped = True


class RLTrainingViewer(QMainWindow):
    """
    Main window for RL training visualization.
    
    Three-panel layout:
    - Left: Behavioral weights and diagnostics
    - Center: Simulation visualization
    - Right: Training controls
    """
    
    def __init__(
        self,
        model_dir: Optional[str] = None,
        start_polygon: Optional[str] = None,
        model_name: str = "salmon_abm",
        basin: str = "nuyakuk",
        hecras_plan_path: Optional[str] = None,
        hecras_start_index: int = 30,
        hecras_time_mode: str = "loop",
        hecras_k: int = 8,
        hecras_cell_size: Optional[float] = None,
        hecras_wetted_threshold: float = 0.05,
        longitudinal_profile: Optional[str] = None,
    ):
        super().__init__()
        self.setWindowTitle("RL Training Visualizer - Behavioral Weight Optimization")
        self._default_window_width = 1400
        self._default_window_height = 800
        self.resize(self._default_window_width, self._default_window_height)
        
        # Configuration
        self.model_dir = model_dir
        self.start_polygon = start_polygon
        self.model_name = model_name
        self.basin = basin
        self.hecras_plan_path = hecras_plan_path
        self.hecras_start_index = int(hecras_start_index)
        self.hecras_time_mode = str(hecras_time_mode or "loop").strip().lower()
        self.hecras_k = int(hecras_k)
        self.hecras_cell_size = hecras_cell_size
        self.hecras_wetted_threshold = hecras_wetted_threshold
        self.longitudinal_profile = longitudinal_profile
        self.crs = "EPSG:26905"  # UTM Zone 5N for Alaska
        self.water_temp = 12.0  # °C
        
        # Training state
        self.trainer = None
        self.training_thread = None
        self.training_worker = None
        self.training_process = None
        self.training_queue = None
        self.training_control_queue = None
        self.training_poll_timer = None
        self.training_paused = False
        self.initial_reward = None

        # Clip recording state
        self.recording_active = False
        self.record_fps = 10
        self.max_record_frames = 1800
        self.recorded_frames: List[np.ndarray] = []
        self.record_timer: Optional[QTimer] = None
        self.record_clip_start_ts: Optional[datetime] = None
        self.record_clip_dir = Path("outputs/rl_clips")
        self._record_overflow = False

        # Episode archive + replay index
        self.archive_h5_path: Optional[str] = None
        self.episode_archive_index: Dict[int, str] = {}
        self.episode_cache: Dict[int, Tuple[Any, ...]] = {}
        
        # Episode visualization queue (for parallel computation)
        self.episode_queue = deque()  # Queue of (episode, reward, components, positions, headings, battery, alive, weights, archive_group)
        self.is_animating = False  # Track if viewer is currently animating

        # Top ticker state (stock-ticker style background progress stream)
        self._ticker_queue = deque()
        self._ticker_text = ""
        self._ticker_index = 0
        self._ticker_timer: Optional[QTimer] = None
        
        # Completed episodes storage (for replay)
        self.completed_episodes = []  # List of (episode, reward, components, positions, headings, battery, alive, weights, archive_group)
        self.best_episode_data = None  # Always keep best episode: (episode, reward, components, positions, headings, battery, alive, weights)
        self.best_reward = float('-inf')  # Track best reward seen
        self.best_weights = None
        
        # Current behavioral weights (modified by randomize buttons)
        from emergent.salmon_abm.rl_training import BehavioralWeights
        self.current_weights = BehavioralWeights()
        
        # Episode synchronization
        self.pending_episode_data = None  # Stores (episode, reward, components) waiting for animation
        self._ui_settings = QSettings("emergent", "rl_training_viewer")
        self._compact_layout = True
        self._view_state = {
            "show_weight_editor": False,
            "show_replay_controls": False,
            "show_advanced_tuning": False,
            "compact_layout": True,
        }
        self._window_state_adjusting = False
        
        print(f"RLTrainingViewer init: calling init_ui()...", flush=True)
        try:
            self.init_ui()
            self._create_view_menu()
            self._load_view_preferences()
            QTimer.singleShot(0, self._fit_window_to_screen)
            print(f"RLTrainingViewer init: init_ui() complete", flush=True)
        except Exception as e:
            import traceback
            print(f"ERROR in init_ui: {e}", flush=True)
            traceback.print_exc()
            raise

    def _fit_window_to_screen(self):
        """Clamp window size/position to the active screen so it stays usable in noVNC."""
        app = QApplication.instance()
        if app is None:
            return
        screen = None
        handle = self.windowHandle()
        if handle is not None:
            screen = handle.screen()
        if screen is None:
            screen = app.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        if available.width() <= 0 or available.height() <= 0:
            return

        margin_px = 40
        min_width = 980
        min_height = 620
        max_width = max(640, available.width() - margin_px)
        max_height = max(420, available.height() - margin_px)
        target_width = min(self._default_window_width, max_width)
        target_height = min(self._default_window_height, max_height)
        target_width = max(min_width, target_width)
        target_height = max(min_height, target_height)

        # Guard against edge cases where available geometry is very small.
        target_width = max(640, min(target_width, max_width))
        target_height = max(420, min(target_height, max_height))

        self.resize(target_width, target_height)
        x = available.x() + max(0, (available.width() - target_width) // 2)
        y = available.y() + max(0, (available.height() - target_height) // 2)
        self.move(x, y)

    def _fit_from_maximize(self) -> None:
        """Handle title-bar maximize by fitting to safe noVNC geometry instead."""
        if self._window_state_adjusting:
            return
        self._window_state_adjusting = True
        try:
            if self.isMaximized():
                self.showNormal()
            self._fit_window_to_screen()
        finally:
            self._window_state_adjusting = False

    def changeEvent(self, event):
        if event.type() == QEvent.WindowStateChange:
            if self.isMaximized() and not self._window_state_adjusting:
                QTimer.singleShot(0, self._fit_from_maximize)
        super().changeEvent(event)

    def _append_background_status(self, message: str):
        """Append a timestamped status line to the bottom-left background log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.weights_panel.append_background_log(f"[{timestamp}] {message}")
        self._enqueue_ticker_message(message)

    def _enqueue_ticker_message(self, message: str):
        """Queue a message for ticker scrolling across the top strip."""
        clean = " ".join(str(message).split())
        if not clean:
            return
        self._ticker_queue.append(clean)
        if not self._ticker_text:
            self._start_next_ticker_message()

    def _start_next_ticker_message(self):
        if self._ticker_queue:
            msg = self._ticker_queue.popleft()
        else:
            msg = "RL viewer ready"
        visible_chars = self._ticker_visible_chars()
        self._ticker_text = (" " * visible_chars) + msg + (" " * max(8, visible_chars // 2))
        self._ticker_index = 0

    def _ticker_visible_chars(self) -> int:
        fm = self.ticker_label.fontMetrics()
        char_w = max(1, fm.horizontalAdvance("M"))
        width_px = max(120, self.ticker_label.width())
        return max(16, int(width_px // char_w))

    def _advance_ticker(self):
        if not self._ticker_text:
            self._start_next_ticker_message()
        if not self._ticker_text:
            return
        visible_chars = self._ticker_visible_chars()
        end = self._ticker_index + visible_chars
        if end <= len(self._ticker_text):
            frame = self._ticker_text[self._ticker_index:end]
        else:
            frame = self._ticker_text[self._ticker_index:] + (" " * (end - len(self._ticker_text)))
        self.ticker_label.setText(frame)
        self._ticker_index += 1
        if self._ticker_index >= len(self._ticker_text):
            self._start_next_ticker_message()

    @staticmethod
    def _to_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return bool(default)
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        return bool(default)

    def _sync_action_checked(self, action: Optional[QAction], checked: bool) -> None:
        if action is None:
            return
        if action.isChecked() == checked:
            return
        action.blockSignals(True)
        action.setChecked(checked)
        action.blockSignals(False)

    def _set_weight_editor_visible(self, visible: bool, persist: bool = True) -> None:
        show = bool(visible)
        self.weights_panel.set_advanced_visible(show)
        self._view_state["show_weight_editor"] = show
        self._sync_action_checked(getattr(self, "action_show_weight_editor", None), show)
        if persist:
            self._save_view_preferences()

    def _set_replay_controls_visible(self, visible: bool, persist: bool = True) -> None:
        show = bool(visible)
        self.control_panel.set_replay_controls_visible(show)
        self._view_state["show_replay_controls"] = show
        self._sync_action_checked(getattr(self, "action_show_replay", None), show)
        if persist:
            self._save_view_preferences()

    def _set_advanced_tuning_visible(self, visible: bool, persist: bool = True) -> None:
        show = bool(visible)
        self.control_panel.set_advanced_controls_visible(show)
        self._view_state["show_advanced_tuning"] = show
        self._sync_action_checked(getattr(self, "action_show_tuning", None), show)
        if persist:
            self._save_view_preferences()

    def _set_compact_layout(self, compact: bool, persist: bool = True) -> None:
        self._compact_layout = bool(compact)
        if self._compact_layout:
            self.control_scroll.setMinimumWidth(220)
            self.control_scroll.setMaximumWidth(340)
            self.weights_scroll.setMinimumWidth(200)
            self.weights_scroll.setMaximumWidth(300)
            target_sizes = [210, 1120, 250]
        else:
            self.control_scroll.setMinimumWidth(260)
            self.control_scroll.setMaximumWidth(420)
            self.weights_scroll.setMinimumWidth(220)
            self.weights_scroll.setMaximumWidth(360)
            target_sizes = [260, 980, 320]
        if hasattr(self, "main_splitter") and self.main_splitter is not None:
            self.main_splitter.setSizes(target_sizes)
        self._view_state["compact_layout"] = self._compact_layout
        self._sync_action_checked(getattr(self, "action_compact_layout", None), self._compact_layout)
        if persist:
            self._save_view_preferences()

    def _create_view_menu(self) -> None:
        view_menu = self.menuBar().addMenu("&View")

        self.action_fit_window = QAction("Fit Window to noVNC Desktop", self)
        self.action_fit_window.setShortcut("Ctrl+0")
        self.action_fit_window.triggered.connect(self._fit_window_to_screen)
        view_menu.addAction(self.action_fit_window)

        view_menu.addSeparator()

        self.action_compact_layout = QAction("Compact Layout (Laptop)", self, checkable=True)
        self.action_compact_layout.toggled.connect(lambda checked: self._set_compact_layout(checked, persist=True))
        view_menu.addAction(self.action_compact_layout)

        view_menu.addSeparator()

        self.action_show_weight_editor = QAction("Show Weight Editor", self, checkable=True)
        self.action_show_weight_editor.toggled.connect(
            lambda checked: self._set_weight_editor_visible(checked, persist=True)
        )
        view_menu.addAction(self.action_show_weight_editor)

        self.action_show_replay = QAction("Show Replay/Capture Controls", self, checkable=True)
        self.action_show_replay.toggled.connect(
            lambda checked: self._set_replay_controls_visible(checked, persist=True)
        )
        view_menu.addAction(self.action_show_replay)

        self.action_show_tuning = QAction("Show Advanced Tuning Controls", self, checkable=True)
        self.action_show_tuning.toggled.connect(
            lambda checked: self._set_advanced_tuning_visible(checked, persist=True)
        )
        view_menu.addAction(self.action_show_tuning)

    def _load_view_preferences(self) -> None:
        show_weight_editor = self._to_bool(
            self._ui_settings.value("view/show_weight_editor", self._view_state["show_weight_editor"]),
            self._view_state["show_weight_editor"],
        )
        show_replay_controls = self._to_bool(
            self._ui_settings.value("view/show_replay_controls", self._view_state["show_replay_controls"]),
            self._view_state["show_replay_controls"],
        )
        show_advanced_tuning = self._to_bool(
            self._ui_settings.value("view/show_advanced_tuning", self._view_state["show_advanced_tuning"]),
            self._view_state["show_advanced_tuning"],
        )
        compact_layout = self._to_bool(
            self._ui_settings.value("view/compact_layout", self._view_state["compact_layout"]),
            self._view_state["compact_layout"],
        )

        self._set_compact_layout(compact_layout, persist=False)
        self._set_weight_editor_visible(show_weight_editor, persist=False)
        self._set_replay_controls_visible(show_replay_controls, persist=False)
        self._set_advanced_tuning_visible(show_advanced_tuning, persist=False)

        splitter_raw = self._ui_settings.value("view/splitter_sizes", "")
        if isinstance(splitter_raw, str) and splitter_raw.strip() and hasattr(self, "main_splitter"):
            try:
                splitter_sizes = [int(part) for part in splitter_raw.split(",") if part.strip()]
                if len(splitter_sizes) == 3 and sum(splitter_sizes) > 0:
                    self.main_splitter.setSizes(splitter_sizes)
            except ValueError:
                pass

    def _save_view_preferences(self) -> None:
        self._ui_settings.setValue("view/show_weight_editor", int(self._view_state["show_weight_editor"]))
        self._ui_settings.setValue("view/show_replay_controls", int(self._view_state["show_replay_controls"]))
        self._ui_settings.setValue("view/show_advanced_tuning", int(self._view_state["show_advanced_tuning"]))
        self._ui_settings.setValue("view/compact_layout", int(self._view_state["compact_layout"]))
        if hasattr(self, "main_splitter") and self.main_splitter is not None:
            sizes = self.main_splitter.sizes()
            self._ui_settings.setValue("view/splitter_sizes", ",".join(str(int(s)) for s in sizes))
        self._ui_settings.sync()
        
    def init_ui(self):
        """Initialize UI components."""
        # Create panels
        self.weights_panel = WeightsPanel()
        self.weights_scroll = QScrollArea()
        self.weights_scroll.setWidgetResizable(True)
        self.weights_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.weights_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.weights_scroll.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.weights_scroll.setMinimumWidth(210)
        self.weights_scroll.setMaximumWidth(320)
        self.weights_scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.weights_scroll.setWidget(self.weights_panel)
        self.simulation_canvas = SimulationCanvas(
            model_dir=self.model_dir,
            hecras_plan_path=self.hecras_plan_path,
            hecras_start_index=self.hecras_start_index,
            hecras_k=self.hecras_k,
        )
        self.control_panel = ControlPanel()
        self.control_scroll = QScrollArea()
        self.control_scroll.setWidgetResizable(True)
        self.control_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.control_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.control_scroll.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.control_scroll.setMinimumWidth(240)
        self.control_scroll.setMaximumWidth(380)
        self.control_scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.control_scroll.setWidget(self.control_panel)
        self.control_panel.set_input_paths(
            model_dir=self.model_dir,
            hecras_plan=self.hecras_plan_path,
            start_polygon=self.start_polygon,
            longitudinal_profile=self.longitudinal_profile,
        )
        
        # Connect control signals
        self.control_panel.start_training.connect(self.on_start_training)
        self.control_panel.pause_training.connect(self.on_pause_training)
        self.control_panel.stop_training.connect(self.on_stop_training)
        self.control_panel.randomize_weights.connect(self.on_randomize_weights)
        self.control_panel.set_blanket_value.connect(self.on_set_blanket_value)
        self.control_panel.randomize_order.connect(self.on_randomize_order)
        self.control_panel.reset_training.connect(self.on_reset_training)
        self.control_panel.episode_selected.connect(self.on_episode_selected)
        self.control_panel.next_episode.connect(self.on_next_episode)
        self.control_panel.prev_episode.connect(self.on_prev_episode)
        self.control_panel.toggle_playback.connect(self.on_toggle_playback)
        self.control_panel.restart_animation.connect(self.on_restart_animation)
        self.control_panel.start_recording.connect(self.on_start_recording)
        self.control_panel.stop_recording.connect(self.on_stop_recording)
        self.control_panel.open_archive.connect(self.on_open_archive)
        
        # Connect animation finished signal
        self.simulation_canvas.animation_finished.connect(self.on_animation_finished)
        
        # Create splitter for three panels
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.addWidget(self.weights_scroll)
        self.main_splitter.addWidget(self.simulation_canvas)
        self.main_splitter.addWidget(self.control_scroll)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)
        self.main_splitter.setSizes([220, 1060, 280])

        self.ticker_strip = QWidget()
        self.ticker_strip.setObjectName("tickerStrip")
        self.ticker_strip.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.ticker_strip.setMinimumHeight(26)
        self.ticker_strip.setMaximumHeight(26)
        self.ticker_strip.setStyleSheet(
            "#tickerStrip { background-color: #1f2a30; } "
            "#tickerStrip QLabel { color: #d7f5c8; }"
        )

        ticker_layout = QHBoxLayout(self.ticker_strip)
        ticker_layout.setContentsMargins(8, 0, 8, 0)
        ticker_layout.setSpacing(0)

        self.ticker_label = QLabel("    RL viewer ready    ")
        self.ticker_label.setFont(QFont("Courier New", 9, QFont.Bold))
        self.ticker_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.ticker_label.setMinimumHeight(24)
        self.ticker_label.setMaximumHeight(24)
        self.ticker_label.setWordWrap(False)
        self.ticker_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        ticker_layout.addWidget(self.ticker_label)

        self._ticker_timer = QTimer(self)
        self._ticker_timer.setInterval(120)
        self._ticker_timer.timeout.connect(self._advance_ticker)
        self._ticker_timer.start()

        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(self.ticker_strip)
        main_layout.addWidget(self.main_splitter)
        main_layout.setStretch(0, 0)
        main_layout.setStretch(1, 1)

        self.setCentralWidget(main_widget)
        
        # Display initial weights
        self.weights_panel.update_weights(self.current_weights)
        self._append_background_status("Viewer ready. Press Start Training to begin.")

    def _is_training_active(self) -> bool:
        return self.training_process is not None and self.training_process.is_alive()

    def _start_training_poll(self) -> None:
        if self.training_poll_timer is None:
            self.training_poll_timer = QTimer(self)
            self.training_poll_timer.setInterval(100)
            self.training_poll_timer.timeout.connect(self._poll_training_queue)
        if not self.training_poll_timer.isActive():
            self.training_poll_timer.start()

    def _stop_training_poll(self) -> None:
        if self.training_poll_timer is not None and self.training_poll_timer.isActive():
            self.training_poll_timer.stop()

    def _poll_training_queue(self) -> None:
        if self.training_queue is None:
            return
        handled = 0
        while handled < 20:
            try:
                msg = self.training_queue.get_nowait()
            except queue_mod.Empty:
                break
            self._handle_training_message(msg)
            handled += 1
        if self.training_process is not None and not self.training_process.is_alive():
            if self.training_queue is None or self.training_queue.empty():
                self._cleanup_training_process()

    def _handle_training_message(self, msg: Dict[str, Any]) -> None:
        msg_type = msg.get("type")
        if msg_type == "status":
            message = str(msg.get("message", "")).strip()
            if message:
                self._append_background_status(message)
            return
        if msg_type == "episode_started":
            self.on_episode_started(int(msg.get("episode", 0)))
            return
        if msg_type == "episode_computed":
            weights_dict = msg.get("weights") or {}
            current_weights = BehavioralWeights.from_dict(weights_dict) if weights_dict else None
            reward = float(msg.get("reward", 0.0))
            archive_group = msg.get("archive_group")
            archive_h5_path = msg.get("archive_h5_path")
            if archive_h5_path:
                self.archive_h5_path = str(archive_h5_path)
            episode_idx = int(msg.get("episode", 0))
            if archive_group:
                self.episode_archive_index[episode_idx] = str(archive_group)
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_weights = current_weights
            self.on_episode_computed(
                episode_idx,
                reward,
                msg.get("components", {}),
                msg.get("positions"),
                msg.get("headings"),
                msg.get("battery"),
                msg.get("alive"),
                current_weights,
                archive_group=archive_group,
            )
            return
        if msg_type == "training_completed":
            best_weights = BehavioralWeights.from_dict(msg.get("best_weights") or {})
            self.best_weights = best_weights
            history = msg.get("history", [])
            archive_h5_path = msg.get("archive_h5_path")
            if archive_h5_path:
                self.archive_h5_path = str(archive_h5_path)
            self.on_training_completed(best_weights, history)
            self._cleanup_training_process()
            return
        if msg_type == "training_stopped":
            best_weights = BehavioralWeights.from_dict(msg.get("best_weights") or {})
            self.best_weights = best_weights
            archive_h5_path = msg.get("archive_h5_path")
            if archive_h5_path:
                self.archive_h5_path = str(archive_h5_path)
            self._cleanup_training_process()
            self.control_panel.append_log("Training stopped")
            if self.archive_h5_path:
                self.control_panel.append_log(f"Archive saved: {self.archive_h5_path}")
            self.control_panel.status_label.setText("Stopped")
            self.control_panel.set_parameters_enabled(True)
            return
        if msg_type == "error":
            self.on_error_occurred(msg.get("message", "Unknown training error"))
            self._cleanup_training_process()
            return

    def _cleanup_training_process(self) -> None:
        self._stop_training_poll()
        if self.training_process is not None and self.training_process.is_alive():
            self.training_process.join(timeout=1.0)
        self.training_process = None
        self.training_queue = None
        self.training_control_queue = None
        self.training_paused = False
        self.training_thread = None
        self.training_worker = None

    def _archive_group_for_episode(self, episode_num: int) -> Optional[str]:
        if episode_num in self.episode_archive_index:
            return self.episode_archive_index[episode_num]
        if self.archive_h5_path:
            return f"episodes/{int(episode_num):06d}"
        return None

    def _load_episode_from_archive(self, episode_num: int):
        if not self.archive_h5_path:
            raise RuntimeError("Archive replay requested but archive_h5_path is not set")
        group_name = self._archive_group_for_episode(episode_num)
        if not group_name:
            raise KeyError(f"No archive group registered for episode {episode_num}")

        with h5py.File(self.archive_h5_path, "r") as h5:
            if group_name not in h5:
                raise KeyError(f"Episode group not found in archive: {group_name}")
            grp = h5[group_name]
            positions = np.asarray(grp["positions"], dtype=np.float32)
            headings = np.asarray(grp["headings"], dtype=np.float32)
            battery = np.asarray(grp["battery"], dtype=np.float32)
            alive = np.asarray(grp["alive"], dtype=np.uint8).astype(bool)
            reward = float(grp.attrs.get("reward", 0.0))

            components_json = grp.attrs.get("components_json", "{}")
            if isinstance(components_json, bytes):
                components_json = components_json.decode("utf-8")
            components = json.loads(str(components_json))

            weights_json = grp.attrs.get("weights_json", "{}")
            if isinstance(weights_json, bytes):
                weights_json = weights_json.decode("utf-8")
            weights_dict = json.loads(str(weights_json))
            weights = BehavioralWeights.from_dict(weights_dict) if weights_dict else None

        return (
            int(episode_num),
            reward,
            components,
            positions,
            headings,
            battery,
            alive,
            weights,
            group_name,
        )

    def _get_episode_data(self, episode_num: int):
        if episode_num in self.episode_cache:
            return self.episode_cache[episode_num]

        for ep_data in self.completed_episodes:
            if int(ep_data[0]) == int(episode_num):
                self.episode_cache[int(episode_num)] = ep_data
                return ep_data

        archived = self._load_episode_from_archive(int(episode_num))
        self.episode_cache[int(episode_num)] = archived
        return archived

    def _load_archive_index(self, archive_path: str) -> int:
        archive_abs = os.path.abspath(archive_path)
        if not os.path.exists(archive_abs):
            raise FileNotFoundError(f"Archive not found: {archive_abs}")
        self.episode_archive_index.clear()
        with h5py.File(archive_abs, "r") as h5:
            if "episodes" not in h5:
                raise KeyError(f"Archive missing 'episodes' group: {archive_abs}")
            episode_groups = sorted(h5["episodes"].keys())
            if not episode_groups:
                raise ValueError(f"Archive has no episodes: {archive_abs}")
            episode_numbers = []
            for group_name in episode_groups:
                grp = h5["episodes"][group_name]
                episode_num = int(grp.attrs.get("episode", int(group_name)))
                self.episode_archive_index[episode_num] = f"episodes/{group_name}"
                episode_numbers.append(episode_num)

        self.archive_h5_path = archive_abs
        self.episode_cache.clear()
        self.completed_episodes.clear()
        self.control_panel.clear_episode_list()
        for ep in sorted(episode_numbers):
            self.control_panel.add_episode_to_list(ep)
        return len(episode_numbers)

    def on_open_archive(self):
        if self._is_training_active():
            self.control_panel.append_log("Stop training before opening a different archive")
            return
        start_dir = str(Path("outputs/rl_training").resolve())
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open RL Archive HDF5",
            start_dir,
            "HDF5 Files (*.h5 *.hdf5 *.hdf);;All Files (*)",
        )
        if not path:
            return
        try:
            count = self._load_archive_index(path)
        except Exception as exc:
            self.control_panel.append_log(f"ERROR loading archive: {exc}")
            self.control_panel.status_label.setText("Archive load failed")
            return
        self.control_panel.append_log(f"Loaded archive: {path} ({count} episodes)")
        self.control_panel.status_label.setText(f"Archive loaded ({count} episodes)")
        
    def create_simulation_factory(self, num_agents: int, num_timesteps: int):
        """
        Create simulation factory function for RL training.
        
        Args:
            num_agents: Number of agents per episode
            num_timesteps: Number of timesteps per episode
            
        Returns:
            Callable that takes BehavioralWeights and returns a configured simulation object
        """
        # Find environment files
        env_files = []
        for fname in ['depth.tif', 'vel_x.tif', 'vel_y.tif', 'vel_mag.tif', 'vel_dir.tif']:
            fpath = os.path.join(self.model_dir, fname)
            if os.path.exists(fpath):
                env_files.append(fpath)
        
        if not env_files:
            raise FileNotFoundError(f"No environment files found in {self.model_dir}")
        
        self.control_panel.append_log(f"Found {len(env_files)} environment files")
        
        # Look for longitudinal profile shapefile (REQUIRED for reward function)
        longitudinal_path = os.path.join(self.model_dir, 'longitudinal.shp')
        if not os.path.exists(longitudinal_path):
            raise FileNotFoundError(
                f"Longitudinal profile shapefile required but not found: {longitudinal_path}\n"
                f"The reward function requires this to compute upstream progress accurately."
            )
        self.control_panel.append_log(f"Found longitudinal profile: {longitudinal_path}")
        
        def factory_func(weights: BehavioralWeights):
            """Create and configure simulation with given behavioral weights."""
            # Create simulation with minimal output writes (compute-only)
            sim = simulation(
                model_dir=self.model_dir,
                model_name=self.model_name,
                crs=self.crs,
                basin=self.basin,
                water_temp=self.water_temp,
                start_polygon=self.start_polygon,
                env_files=env_files,
                longitudinal_profile=longitudinal_path,
                fish_length=None,  # Random fish lengths
                num_timesteps=num_timesteps,
                num_agents=num_agents,
                use_gpu=False,
                pid_tuning=False,
                db_path=None,  # temporary DB (auto-cleanup)
                output_write_mode='none',  # skip all writes (compute-only)
                output_write_backend='sync'
            )
            
            # Load behavioral weights into simulation
            sim.load_behavioral_weights(weights_dict=weights.to_dict())
            
            # Enable neighbor sensing for schooling cues
            fish_length_m = 0.3  # Approximate 300mm fish
            sensory_range = 5.0  # Fixed biological constant (5 BL = 1.5m)
            sim.neighbor_buffer_radius = sensory_range * fish_length_m
            sim.neighbor_buffer_lengths = sensory_range
            
            # RANDOMIZE initial conditions for RL exploration
            # Random headings [0, 2π] instead of upstream direction
            sim.heading = np.random.uniform(0, 2*np.pi, sim.num_agents).astype(np.float32)
            # Random initial SOG [0.1, 1.5] m/s instead of ideal_sog
            sim.sog = np.random.uniform(0.1, 1.5, sim.num_agents).astype(np.float32)
            
            return sim
        
        return factory_func
        
    def on_start_training(self):
        """Start training with current parameters."""
        if self._is_training_active():
            self.control_panel.append_log("Training already in progress!")
            return

        model_dir = self.control_panel.get_model_dir() or self.model_dir
        hecras_plan_candidate = self.control_panel.get_hecras_plan() or self.hecras_plan_path
        start_polygon = self.control_panel.get_start_polygon() or self.start_polygon
        longitudinal_profile = self.control_panel.get_longitudinal_profile() or self.longitudinal_profile

        # Persist latest UI-selected paths in viewer state
        self.model_dir = model_dir
        self.hecras_plan_path = hecras_plan_candidate
        self.start_polygon = start_polygon
        self.longitudinal_profile = longitudinal_profile

        # Check configuration
        if not start_polygon:
            self.control_panel.append_log("ERROR: Missing start polygon configuration")
            self.control_panel.append_log("Provide --start-polygon <path>")
            return

        if not os.path.exists(start_polygon):
            self.control_panel.append_log(f"ERROR: Start polygon not found: {start_polygon}")
            return

        hecras_plan = None
        if hecras_plan_candidate:
            hecras_plan = os.path.abspath(str(hecras_plan_candidate))
            if not os.path.exists(hecras_plan):
                self.control_panel.append_log(f"ERROR: HECRAS plan not found: {hecras_plan}")
                return

        if hecras_plan is None:
            if not model_dir:
                self.control_panel.append_log("ERROR: Missing model-dir (required for raster mode)")
                return
            if not os.path.exists(model_dir):
                self.control_panel.append_log(f"ERROR: Model directory not found: {model_dir}")
                return
            
        # Get parameters from control panel
        num_episodes = self.control_panel.episodes_spin.value()
        num_timesteps = self.control_panel.timesteps_spin.value()
        num_agents = self.control_panel.agents_spin.value()
        exploration_noise = self.control_panel.noise_spin.value()
        
        self.control_panel.append_log("=" * 50)
        self.control_panel.append_log(f"Starting RL training:")
        self.control_panel.append_log(f"  Episodes: {num_episodes}")
        self.control_panel.append_log(f"  Timesteps: {num_timesteps}")
        self.control_panel.append_log(f"  Agents: {num_agents}")
        self.control_panel.append_log(f"  Exploration: {exploration_noise}")
        self.control_panel.status_label.setText("Initializing...")
        
        try:
            self.simulation_canvas.configure_environment(
                model_dir=model_dir,
                hecras_plan_path=hecras_plan,
                hecras_start_index=self.hecras_start_index,
                hecras_k=self.hecras_k,
            )

            # Resolve inputs for subprocess
            env_files = []
            if hecras_plan is None:
                for fname in ['depth.tif', 'vel_x.tif', 'vel_y.tif', 'vel_mag.tif', 'vel_dir.tif']:
                    fpath = os.path.join(model_dir, fname)
                    if os.path.exists(fpath):
                        env_files.append(fpath)
                if not env_files:
                    raise FileNotFoundError(f"No environment files found in {model_dir}")
                self.control_panel.append_log(f"Input mode: raster ({len(env_files)} files)")
            else:
                self.control_panel.append_log(
                    f"Input mode: HECRAS direct ({os.path.basename(hecras_plan)}; "
                    f"start_index={self.hecras_start_index}, mode={self.hecras_time_mode})"
                )

            longitudinal_path = None
            if longitudinal_profile:
                longitudinal_path = os.path.abspath(longitudinal_profile)
            elif model_dir:
                candidate_longitudinal = os.path.join(model_dir, 'longitudinal.shp')
                if os.path.exists(candidate_longitudinal):
                    longitudinal_path = candidate_longitudinal
            if not longitudinal_path or not os.path.exists(longitudinal_path):
                raise FileNotFoundError(
                    "Longitudinal profile shapefile required for RL reward function. "
                    "Provide --longitudinal-profile or place longitudinal.shp in --model-dir."
                )
            self.control_panel.append_log(f"Found longitudinal profile: {longitudinal_path}")

            # Get edited weights from the panel (user may have manually edited values)
            initial_weights = self.weights_panel.get_edited_weights(self.current_weights)
            self.current_weights = initial_weights  # Update current_weights with edits

            reward_weights = {
                'cohesion': self.control_panel.cohesion_reward_spin.value(),
                'alignment': self.control_panel.alignment_reward_spin.value(),
                'separation': self.control_panel.separation_penalty_spin.value(),
                'upstream_progress': self.control_panel.upstream_weight_spin.value(),
                'energy_efficiency': self.control_panel.energy_reward_spin.value(),
                'drafting_benefit': self.control_panel.drafting_reward_spin.value(),
                'boundary_penalty': self.control_panel.boundary_penalty_spin.value(),
                'mortality_penalty': self.control_panel.mortality_penalty_spin.value(),
                'smoothness_penalty': self.control_panel.smoothness_penalty_spin.value(),
                'fatigue_penalty': self.control_panel.fatigue_penalty_spin.value(),
                'stagnation_penalty': self.control_panel.stagnation_penalty_spin.value(),
                'rheotaxis_alignment': self.control_panel.rheotaxis_penalty_spin.value(),
            }
            penalty_summary = (
                f"Penalty weights: separation={reward_weights['separation']:+.3f}, "
                f"boundary={reward_weights['boundary_penalty']:+.3f}, "
                f"mortality={reward_weights['mortality_penalty']:+.3f}, "
                f"smoothness={reward_weights['smoothness_penalty']:+.3f}, "
                f"fatigue={reward_weights['fatigue_penalty']:+.3f}, "
                f"stagnation={reward_weights['stagnation_penalty']:+.3f}, "
                f"rheotaxis={reward_weights['rheotaxis_alignment']:+.3f}"
            )
            self.control_panel.append_log(penalty_summary)
            self._append_background_status(penalty_summary)

            archive_h5_path = None
            if self.control_panel.archive_enabled():
                archive_dir = Path("outputs/rl_training")
                archive_dir.mkdir(parents=True, exist_ok=True)
                archive_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                archive_h5_path = str((archive_dir / f"rl_training_archive_{archive_stamp}.h5").resolve())
                self.control_panel.append_log(f"Episode archive: {archive_h5_path}")
            self.archive_h5_path = archive_h5_path

            config = {
                "model_dir": model_dir,
                "model_name": self.model_name,
                "crs": self.crs,
                "basin": self.basin,
                "water_temp": self.water_temp,
                "start_polygon": start_polygon,
                "env_files": env_files,
                "longitudinal_path": longitudinal_path,
                "hecras_plan_path": hecras_plan,
                "hecras_start_index": self.hecras_start_index,
                "hecras_time_mode": self.hecras_time_mode,
                "hecras_k": self.hecras_k,
                "hecras_cell_size": self.hecras_cell_size,
                "hecras_wetted_threshold": self.hecras_wetted_threshold,
                "num_episodes": num_episodes,
                "num_timesteps": num_timesteps,
                "num_agents": num_agents,
                "exploration_noise": exploration_noise,
                "reward_weights": reward_weights,
                "body_length": 0.3,
                "dt": 1.0,
                "initial_weights": initial_weights.to_dict(),
                "archive_h5_path": archive_h5_path,
            }

            self.best_reward = float('-inf')
            self.best_weights = None
            self.initial_reward = None
            self.training_paused = False
            self.episode_queue.clear()
            self.completed_episodes.clear()
            self.episode_cache.clear()
            self.episode_archive_index.clear()
            self.control_panel.clear_episode_list()
            self.best_episode_data = None

            self.control_panel.append_log("Trainer initialized successfully (subprocess)")

            # Display initial weights
            self.weights_panel.update_weights(initial_weights)

            default_order = {
                0: 'shallow',
                1: 'border',
                2: 'avoid',
                3: 'collision',
                4: 'alignment',
                5: 'cohesion',
                6: 'low_speed',
                7: 'refugia',
                8: 'rheotaxis',
                9: 'wave_drag',
            }
            self.weights_panel.update_order(default_order, initial_weights)

            # Disable parameter controls during training
            self.control_panel.set_parameters_enabled(False)

            ctx = mp.get_context("spawn")
            self.training_queue = ctx.Queue(maxsize=2)
            self.training_control_queue = ctx.Queue()
            self.training_process = ctx.Process(
                target=_training_process_main,
                args=(config, self.training_queue, self.training_control_queue),
            )
            self.training_process.start()
            self._start_training_poll()
            self.control_panel.append_log("Training process started")
            self._append_background_status("Training subprocess launched; waiting for simulation progress...")

        except Exception as e:
            import traceback
            self.control_panel.append_log(f"ERROR: Failed to start training")
            self.control_panel.append_log(str(e))
            self.control_panel.append_log(traceback.format_exc())
            self.control_panel.status_label.setText("Error")
    
    def on_set_blanket_value(self):
        """Set all behavioral weights to a uniform blanket value."""
        if self._is_training_active():
            self.control_panel.append_log("Cannot set blanket value during training")
            return
        
        # Get blanket value from input field
        try:
            blanket_value = float(self.control_panel.blanket_value_input.text())
        except ValueError:
            self.control_panel.append_log("ERROR: Invalid blanket value (must be a number)")
            return
        
        # Directly modify current weights - don't create new instance
        # (creating new BehavioralWeights() resets to defaults)
        self.current_weights.shallow_weight = blanket_value
        self.current_weights.border_cue_weight = blanket_value
        self.current_weights.avoid_weight = blanket_value
        self.current_weights.collision_weight = blanket_value
        self.current_weights.alignment_weight = blanket_value
        self.current_weights.cohesion_weight = blanket_value
        self.current_weights.low_speed_weight = blanket_value
        self.current_weights.refugia_weight = blanket_value
        self.current_weights.rheotaxis_weight = blanket_value
        self.current_weights.wave_drag_weight = blanket_value
        
        # Debug: confirm values were set
        print(f"[BLANKET VALUE DEBUG] Set all weights to {blanket_value}", flush=True)
        print(f"[BLANKET VALUE DEBUG] shallow_weight={self.current_weights.shallow_weight}, cohesion_weight={self.current_weights.cohesion_weight}", flush=True)
        
        # Update spinbox values directly (don't rebuild UI)
        weight_keys = [
            'shallow_weight', 'border_cue_weight', 'avoid_weight', 'collision_weight',
            'alignment_weight', 'cohesion_weight', 'low_speed_weight', 'refugia_weight',
            'rheotaxis_weight', 'wave_drag_weight'
        ]
        for key in weight_keys:
            if key in self.weights_panel.weight_spinboxes:
                self.weights_panel.weight_spinboxes[key].setValue(blanket_value)
        
        self.control_panel.append_log(f"📋 Set all weights to blanket value: {blanket_value}")
        self.control_panel.status_label.setText(f"Ready with uniform weights ({blanket_value})")
    
    def on_randomize_weights(self):
        """Regenerate random initial weights."""
        if self._is_training_active():
            self.control_panel.append_log("Cannot randomize during training")
            return
        
        # Preserve current order before randomizing
        old_order = [
            self.current_weights.order_0, self.current_weights.order_1,
            self.current_weights.order_2, self.current_weights.order_3,
            self.current_weights.order_4, self.current_weights.order_5,
            self.current_weights.order_6, self.current_weights.order_7,
            self.current_weights.order_8, self.current_weights.order_9
        ]
        
        # Generate new randomized weights
        from emergent.salmon_abm.rl_training import BehavioralWeights
        base_weights = BehavioralWeights()
        new_weights = base_weights.randomize(scale=1.0)  # 100% randomization
        
        # Restore order (as integers)
        new_weights.order_0 = int(old_order[0])
        new_weights.order_1 = int(old_order[1])
        new_weights.order_2 = int(old_order[2])
        new_weights.order_3 = int(old_order[3])
        new_weights.order_4 = int(old_order[4])
        new_weights.order_5 = int(old_order[5])
        new_weights.order_6 = int(old_order[6])
        new_weights.order_7 = int(old_order[7])
        new_weights.order_8 = int(old_order[8])
        new_weights.order_9 = int(old_order[9])
        new_weights.arbitration_tolerance = float(base_weights.arbitration_tolerance)
        
        self.current_weights = new_weights  # Store as current for training
        
        # Update display
        self.weights_panel.update_weights(new_weights)
        
        # Update order display (need default_order dict)
        default_order = {
            0: 'shallow', 1: 'border', 2: 'avoid', 3: 'collision', 4: 'alignment',
            5: 'cohesion', 6: 'low_speed', 7: 'refugia', 8: 'rheotaxis', 9: 'wave_drag'
        }
        self.weights_panel.update_order(default_order, new_weights)
        
        self.control_panel.append_log(
            f"🎲 Randomized initial weights (100% variation; arbitration_tolerance={new_weights.arbitration_tolerance:.1f})"
        )
        self.control_panel.status_label.setText("Ready with new random weights")
    
    def on_randomize_order(self):
        """Shuffle cue application order."""
        if self._is_training_active():
            self.control_panel.append_log("Cannot randomize order during training")
            return
        
        # Random permutation of [0,1,2,3,4,5,6,7,8,9]
        new_order = np.random.permutation(10)
        
        # Update current_weights with shuffled order (as integers)
        self.current_weights.order_0 = int(new_order[0])
        self.current_weights.order_1 = int(new_order[1])
        self.current_weights.order_2 = int(new_order[2])
        self.current_weights.order_3 = int(new_order[3])
        self.current_weights.order_4 = int(new_order[4])
        self.current_weights.order_5 = int(new_order[5])
        self.current_weights.order_6 = int(new_order[6])
        self.current_weights.order_7 = int(new_order[7])
        self.current_weights.order_8 = int(new_order[8])
        self.current_weights.order_9 = int(new_order[9])
        
        # Default cue names for display
        default_order = {
            0: 'shallow', 1: 'border', 2: 'avoid', 3: 'collision', 4: 'alignment',
            5: 'cohesion', 6: 'low_speed', 7: 'refugia', 8: 'rheotaxis', 9: 'wave_drag'
        }
        
        # Update display with remapping
        self.weights_panel.update_order(default_order, self.current_weights)
        self.control_panel.append_log(f"🎪 Randomized cue order: {new_order}")
        self.control_panel.status_label.setText("Ready with shuffled cue order")
    
    def on_reset_training(self):
        """Reset training to initial state - stop all processes and start fresh."""
        if self.recording_active:
            self.on_stop_recording()

        # Stop training if running
        if self._is_training_active():
            self.control_panel.append_log("Stopping training for reset...")
            if self.training_control_queue is not None:
                self.training_control_queue.put("stop")
            if self.training_process is not None:
                self.training_process.join(timeout=2.0)
                if self.training_process.is_alive():
                    self.training_process.terminate()
            self._cleanup_training_process()
        
        # Clear episode data
        self.episode_queue.clear()
        self.completed_episodes.clear()
        self.episode_cache.clear()
        self.episode_archive_index.clear()
        self.archive_h5_path = None
        self.control_panel.clear_episode_list()
        self.is_animating = False
        self.pending_episode_data = None
        self._current_episode_data = None
        
        # Clear visualization
        if hasattr(self.simulation_canvas, 'replay_widget'):
            dummy_positions = np.zeros((1, 1, 2), dtype=np.float32)
            self.simulation_canvas.replay_widget.positions = dummy_positions
            self.simulation_canvas.replay_widget.T, self.simulation_canvas.replay_widget.N, _ = dummy_positions.shape
            self.simulation_canvas.replay_widget.frame = 0
            self.simulation_canvas.replay_widget.heading_array = None
            self.simulation_canvas.replay_widget.battery_array = None
            self.simulation_canvas.replay_widget.alive_array = None
            self.simulation_canvas.replay_widget.playing = False
            self.simulation_canvas.replay_widget.timer.stop()
            self.simulation_canvas.replay_widget.update()
        
        num_agents = self.control_panel.agents_spin.value()
        num_timesteps = self.control_panel.timesteps_spin.value()
        self.trainer = None
        self.training_thread = None
        self.training_worker = None
        self.initial_reward = None
        self.best_reward = float('-inf')
        self.best_weights = None
        
        # Clear left-panel progress plot
        self.weights_panel.clear_plot()
        
        # Reset UI
        self.control_panel.progress_bar.setValue(0)
        self.control_panel.btn_pause.setText("⏸ Pause Training")
        self.control_panel.btn_play_pause.setText("⏸ Pause")
        self.control_panel.archive_hdf_check.setChecked(True)
        self.control_panel.status_label.setText("Ready (Reset)")
        self.weights_panel.clear_diagnostics()
        self.weights_panel.clear_background_log()
        self._append_background_status("Training reset. Ready for new run.")
        self.control_panel.append_log("=" * 50)
        self.control_panel.append_log("Training reset - all processes stopped, new model created")
        self.control_panel.append_log(f"Ready for new training run ({num_agents} agents, {num_timesteps} timesteps)")
        self.control_panel.append_log("=" * 50)
        
        # Re-enable parameter controls
        self.control_panel.set_parameters_enabled(True)
        
        self.control_panel.append_log("=" * 50)
        self.control_panel.append_log("🔄 Training reset - ready to start fresh")
        self.control_panel.append_log("Current weights preserved - press Randomize for new weights")

    def _ensure_record_timer(self):
        """Create the recording timer if it does not exist."""
        if self.record_timer is not None:
            return
        self.record_timer = QTimer(self)
        self.record_timer.setInterval(max(20, int(1000 / float(self.record_fps))))
        self.record_timer.timeout.connect(self._capture_record_frame)

    def _capture_record_frame(self):
        """Capture one frame from the simulation canvas."""
        if not self.recording_active:
            return

        if len(self.recorded_frames) >= self.max_record_frames:
            if not self._record_overflow:
                self._record_overflow = True
                self.control_panel.append_log(
                    f"[REC] Reached {self.max_record_frames} frame limit; auto-stopping"
                )
            self.on_stop_recording()
            return

        qimg = self.simulation_canvas.replay_widget.grabFramebuffer()
        if qimg.isNull():
            qimg = self.simulation_canvas.grab().toImage()
        if qimg.isNull():
            raise RuntimeError("Failed to capture recording frame (null image)")

        qimg = qimg.convertToFormat(QImage.Format_RGBA8888)
        width = qimg.width()
        height = qimg.height()
        ptr = qimg.bits()
        ptr.setsize(height * width * 4)
        rgba = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 4)).copy()
        self.recorded_frames.append(rgba[:, :, :3])

        if len(self.recorded_frames) % 25 == 0:
            self.control_panel.set_recording_state(True, frame_count=len(self.recorded_frames))

    def _save_recording_clip(self, frames: List[np.ndarray]) -> Path:
        """Save captured frames as an animated GIF and return the output path."""
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required to save recording clips") from exc

        self.record_clip_dir.mkdir(parents=True, exist_ok=True)
        stamp = (self.record_clip_start_ts or datetime.now()).strftime("%Y%m%d_%H%M%S")
        output_path = self.record_clip_dir / f"rl_training_clip_{stamp}.gif"

        pil_frames = [Image.fromarray(frame, mode="RGB") for frame in frames]
        frame_duration_ms = max(20, int(1000 / float(self.record_fps)))
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=frame_duration_ms,
            loop=0,
        )
        return output_path

    def on_start_recording(self):
        """Begin capturing a snippet from the live simulation canvas."""
        if self.recording_active:
            self.control_panel.append_log("[REC] Recording already active")
            return

        self.recorded_frames = []
        self.record_clip_start_ts = datetime.now()
        self._record_overflow = False
        self.recording_active = True
        self.control_panel.set_recording_state(True, frame_count=0)
        self.control_panel.status_label.setText("Recording clip...")
        self.control_panel.append_log("[REC] Started clip recording")

        self._ensure_record_timer()
        if self.record_timer is not None and not self.record_timer.isActive():
            self.record_timer.start()

    def on_stop_recording(self):
        """Stop recording and persist captured frames to disk."""
        if not self.recording_active:
            self.control_panel.append_log("[REC] Recording is not active")
            return

        self.recording_active = False
        if self.record_timer is not None and self.record_timer.isActive():
            self.record_timer.stop()

        frames = self.recorded_frames
        self.recorded_frames = []
        self.control_panel.set_recording_state(False)

        if not frames:
            self.control_panel.status_label.setText("Recording stopped (no frames)")
            self.control_panel.append_log("[REC] Stopped recording with no captured frames")
            return

        output_path = self._save_recording_clip(frames)
        duration_s = len(frames) / float(self.record_fps)
        self.control_panel.status_label.setText(f"Saved clip: {output_path.name}")
        self.control_panel.append_log(
            f"[REC] Saved {len(frames)} frames ({duration_s:.1f}s) to {output_path}"
        )
        
    def on_pause_training(self):
        """Pause/resume training."""
        if not self._is_training_active():
            return
        if self.training_control_queue is None:
            return
        if self.training_paused:
            self.training_control_queue.put("resume")
            self.training_paused = False
            self.control_panel.btn_pause.setText("⏸ Pause Training")
            self.control_panel.append_log("Training resumed")
        else:
            self.training_control_queue.put("pause")
            self.training_paused = True
            self.control_panel.btn_pause.setText("▶ Resume Training")
            self.control_panel.append_log("Training paused")
    
    def on_toggle_playback(self):
        """Toggle play/pause for current episode animation."""
        if hasattr(self.simulation_canvas, 'replay_widget'):
            if self.simulation_canvas.replay_widget.playing:
                self.simulation_canvas.replay_widget.playing = False
                self.simulation_canvas.replay_widget.timer.stop()
                self.control_panel.btn_play_pause.setText("▶ Play")
                self.control_panel.append_log("Animation paused")
            else:
                self.simulation_canvas.replay_widget.playing = True
                self.simulation_canvas.replay_widget.timer.start()
                self.control_panel.btn_play_pause.setText("⏸ Pause")
                self.control_panel.append_log("Animation playing")
    
    def on_restart_animation(self):
        """Restart current episode animation from beginning."""
        if hasattr(self.simulation_canvas, 'replay_widget'):
            self.simulation_canvas.replay_widget.frame = 0
            self.simulation_canvas.replay_widget.playing = True
            self.simulation_canvas.replay_widget.timer.start()
            self.control_panel.btn_play_pause.setText("⏸ Pause")
            self.simulation_canvas.replay_widget.update()
            self.control_panel.append_log("Animation restarted")
                
    def on_stop_training(self):
        """Stop training."""
        if self._is_training_active():
            if self.training_control_queue is not None:
                self.training_control_queue.put("stop")
            self.control_panel.append_log("Stopping training...")
            self.control_panel.status_label.setText("Stopping...")
            self.control_panel.set_parameters_enabled(True)
            
    def on_episode_started(self, episode: int):
        """Handle episode start."""
        total = self.control_panel.episodes_spin.value()
        self.control_panel.set_progress(episode, total)
        self.control_panel.status_label.setText(f"Computing episode {episode + 1}/{total}...")
        
    def on_episode_computed(self, episode: int, reward: float, components: Dict[str, float],
                           positions: Optional[np.ndarray], headings: Optional[np.ndarray],
                           battery: Optional[np.ndarray], alive: Optional[np.ndarray],
                           weights, archive_group: Optional[str] = None):
        """Handle episode computation complete - queue for visualization."""
        # Add episode to queue
        self.episode_queue.append((episode, reward, components, positions, headings, battery, alive, weights, archive_group))
        
        # Update status
        total = self.control_panel.episodes_spin.value()
        queue_len = len(self.episode_queue)
        self.control_panel.status_label.setText(f"Computed episode {episode + 1}/{total} (queue: {queue_len})")
        
        # If not currently animating, start processing queue
        if not self.is_animating:
            self.process_next_queued_episode()
    
    def process_next_queued_episode(self):
        """Process next episode from queue."""
        print(f"[RL VIEWER DEBUG] process_next_queued_episode called, queue length={len(self.episode_queue)}", flush=True)
        if len(self.episode_queue) == 0:
            print("[RL VIEWER DEBUG] Queue empty, returning", flush=True)
            return
        
        # Mark as animating
        self.is_animating = True
        print("[RL VIEWER DEBUG] Marked as animating", flush=True)
        
        # Get next episode from queue
        episode, reward, components, positions, headings, battery, alive, weights, archive_group = self.episode_queue.popleft()

        # If trajectory was archived, load lazily now to avoid expensive IPC deserialization.
        if positions is None or headings is None or battery is None or alive is None:
            self.control_panel.append_log(f"Loading episode {episode + 1} trajectory from archive...")
            try:
                loaded_episode = self._load_episode_from_archive(int(episode))
                (
                    _loaded_ep,
                    loaded_reward,
                    loaded_components,
                    positions,
                    headings,
                    battery,
                    alive,
                    loaded_weights,
                    loaded_archive_group,
                ) = loaded_episode
                reward = float(loaded_reward)
                if not components:
                    components = loaded_components
                if weights is None:
                    weights = loaded_weights
                if archive_group is None:
                    archive_group = loaded_archive_group
            except Exception as exc:
                self.control_panel.append_log(
                    f"ERROR loading archived trajectory for episode {episode + 1}: {exc}"
                )
                self.is_animating = False
                return

        print(f"[RL VIEWER DEBUG] Got episode {episode} from queue, positions shape={positions.shape}", flush=True)
        
        # Store full episode data for later saving to completed_episodes
        self._current_episode_data = (
            episode,
            reward,
            components,
            positions,
            headings,
            battery,
            alive,
            weights,
            archive_group,
        )
        
        # Store for processing after animation
        self.pending_episode_data = (episode, reward, components, weights)
        
        # Update status
        total = self.control_panel.episodes_spin.value()
        queue_len = len(self.episode_queue)
        self.control_panel.status_label.setText(f"Visualizing episode {episode + 1}/{total} (queue: {queue_len})")
        
        # Start visualization - this will trigger animation_finished when done
        print(f"[RL VIEWER DEBUG] Calling set_positions for episode {episode}", flush=True)
        self.simulation_canvas.set_positions(positions, headings, battery, alive)
        
    def on_animation_finished(self):
        """Handle animation playback complete - update UI and process next queued episode."""
        if self.pending_episode_data is None:
            self.is_animating = False
            return
            
        episode, reward, components, current_weights = self.pending_episode_data
        
        # Diagnostic logging: Track weight evolution and warn about collapse
        if self.best_weights is not None:
            weights_dict = self.best_weights.to_dict()
            cohesion_w = weights_dict.get('cohesion', 0)
            alignment_w = weights_dict.get('alignment', 0)
            rheotaxis_w = weights_dict.get('rheotaxis', 0)
            collision_w = weights_dict.get('collision', 0)
            refugia_w = weights_dict.get('refugia', 0)
            
            weight_summary = f"Rheo:{rheotaxis_w:.0f} Coh:{cohesion_w:.0f} Align:{alignment_w:.0f} Coll:{collision_w:.0f} Ref:{refugia_w:.0f}"
            
            # CRITICAL WARNING if schooling weights drop too low
            if cohesion_w < 1000 or alignment_w < 1000 or collision_w < 1000:
                self.control_panel.append_log(
                    f"⚠️  WARNING Ep{episode+1}: Schooling collapse! {weight_summary}"
                )
        else:
            weight_summary = "(weights unavailable)"
        self.pending_episode_data = None
        
        total = self.control_panel.episodes_spin.value()
        
        # Update diagnostics
        if self.initial_reward is None:
            self.initial_reward = reward
            
        best_reward = self.best_reward if self.best_reward != float('-inf') else reward
        self.weights_panel.update_diagnostics(episode + 1, total, reward, best_reward, self.initial_reward)
        self.weights_panel.update_components(components)
        
        # Update weights display with CURRENT episode's weights (not just best)
        if current_weights:
            self.weights_panel.update_weights(current_weights)
            # Also update order display
            default_order = {i: ['shallow', 'border', 'avoid', 'collision', 'refugia', 'rheotaxis', 'low_speed', 'wave_drag', 'cohesion', 'alignment'][i] for i in range(10)}
            self.weights_panel.update_order(default_order, current_weights)
        
        # Update left-panel progress figure
        self.weights_panel.update_plot(episode + 1, reward)
        
        # Log progress (minimal)
        is_best = "✓ BEST" if reward >= best_reward else ""
        self.control_panel.append_log(f"Ep {episode + 1}/{total}: {reward:.2f} {is_best} | {weight_summary}")
        
        # Update status
        self.control_panel.status_label.setText(f"Episode {episode + 1}/{total} complete")
        
        # Store completed episode for replay (get original data from process_next_queued_episode)
        # We need to retrieve the positions/headings/battery/alive that were used
        # Store in completed_episodes list (with memory optimization)
        if hasattr(self, '_current_episode_data'):
            episode_num, reward, components, positions, headings, battery, alive, weights, archive_group = self._current_episode_data
            
            # Get storage interval from control panel
            storage_interval = self.control_panel.storage_interval_spin.value()
            
            # Decide whether to store this episode
            should_store = False
            
            # Always store first 5 episodes (show initial chaos)
            if episode_num < 5:
                should_store = True
            # Store every Nth episode
            elif episode_num % storage_interval == 0:
                should_store = True
            
            if should_store:
                self.completed_episodes.append(self._current_episode_data)
                self.episode_cache[int(episode_num)] = self._current_episode_data
                print(f"[STORAGE] Stored episode {episode_num} for replay (interval={storage_interval})", flush=True)

            if archive_group:
                self.episode_archive_index[int(episode_num)] = str(archive_group)
                self.control_panel.add_episode_to_list(episode_num)
            elif should_store:
                self.control_panel.add_episode_to_list(episode_num)
            
            # Always track best episode separately
            if self.best_episode_data is None or reward >= self.best_reward:
                self.best_reward = reward
                self.best_episode_data = self._current_episode_data
                print(f"[STORAGE] New best episode: {episode_num} with reward {reward:.2f}", flush=True)
            
            delattr(self, '_current_episode_data')
        else:
            # Safety: if _current_episode_data missing, still try to add to list
            # (shouldn't happen, but prevents episode 0 from being lost)
            self.control_panel.append_log(f"Warning: Episode {episode} missing episode data, adding anyway")
            self.control_panel.add_episode_to_list(episode)
        
        # Clear pending data
        self.pending_episode_data = None
        
        # Mark animation as complete
        self.is_animating = False
        
        # Process next episode from queue if available
        if len(self.episode_queue) > 0:
            self.process_next_queued_episode()
        
    def on_training_completed(self, best_weights: BehavioralWeights, history: List[Tuple[int, float]]):
        """Handle training completion."""
        self.control_panel.append_log("=" * 40)
        self.control_panel.append_log("Training completed!")
        self.best_weights = best_weights
        if history:
            self.best_reward = max(r for _, r in history)
            self.control_panel.append_log(f"Best reward: {self.best_reward:.2f}")
        
        if history:
            initial = history[0][1]
            final = history[-1][1]
            improvement = final - initial
            self.control_panel.append_log(f"Initial reward: {initial:.2f}")
            self.control_panel.append_log(f"Final reward: {final:.2f}")
            self.control_panel.append_log(f"Improvement: {improvement:+.2f}")
        if self.archive_h5_path:
            self.control_panel.append_log(f"Archive saved: {self.archive_h5_path}")
        
        self.control_panel.status_label.setText("Training complete")
        self.control_panel.set_progress(100, 100)
        
        # Re-enable parameter controls
        self.control_panel.set_parameters_enabled(True)
        
    def on_error_occurred(self, error_msg: str):
        """Handle training error."""
        self.control_panel.append_log("=" * 40)
        self.control_panel.append_log("ERROR:")
        self.control_panel.append_log(error_msg)
        self.control_panel.status_label.setText("Error occurred")
        
        # Re-enable parameter controls
        self.control_panel.set_parameters_enabled(True)
    
    def closeEvent(self, event):
        """Handle window close - ensure process is stopped properly."""
        if self.recording_active:
            self.on_stop_recording()

        if self._is_training_active():
            print("[VIEWER] Stopping training process before close...", flush=True)
            try:
                if self.training_control_queue is not None:
                    self.training_control_queue.put("stop")
                if self.training_process is not None:
                    self.training_process.join(timeout=3.0)
                    if self.training_process.is_alive():
                        print("[VIEWER] WARNING: Process did not stop in time, terminating", flush=True)
                        self.training_process.terminate()
                        self.training_process.join(timeout=1.0)
                self._cleanup_training_process()
            except Exception as e:
                print(f"[VIEWER] Error stopping process: {e}", flush=True)
        
        # Stop animation timer
        if hasattr(self.simulation_canvas, 'replay_widget'):
            try:
                self.simulation_canvas.replay_widget.playing = False
                self.simulation_canvas.replay_widget.timer.stop()
            except Exception:
                pass
        try:
            self._save_view_preferences()
        except Exception:
            pass
        
        print("[VIEWER] Window closed cleanly", flush=True)
        event.accept()
    
    def on_episode_selected(self, index: int):
        """Replay selected episode from dropdown."""
        if index < 0 or index >= self.control_panel.episode_combo.count():
            return

        episode_num_data = self.control_panel.episode_combo.itemData(index)
        if episode_num_data is None:
            episode_num = int(index)
        else:
            episode_num = int(episode_num_data)

        try:
            episode_data = self._get_episode_data(episode_num)
        except Exception as exc:
            self.control_panel.append_log(f"ERROR loading episode {episode_num + 1}: {exc}")
            return

        episode, reward, components, positions, headings, battery, alive, weights, _archive_group = episode_data
        
        # Update weights and diagnostics
        total = self.control_panel.episodes_spin.value()
        best_reward = self.best_reward if self.best_reward != float('-inf') else reward
        initial_reward = self.initial_reward if self.initial_reward else reward
        
        self.weights_panel.update_diagnostics(episode + 1, total, reward, best_reward, initial_reward)
        self.weights_panel.update_components(components)
        self.weights_panel.update_weights(weights)
        
        default_order = {i: ['shallow', 'border', 'avoid', 'collision', 'refugia', 'rheotaxis', 'low_speed', 'wave_drag', 'cohesion', 'alignment'][i] for i in range(10)}
        self.weights_panel.update_order(default_order, weights)
        
        # Replay visualization
        self.simulation_canvas.set_positions(positions, headings, battery, alive)
        
        self.control_panel.status_label.setText(f"Replaying episode {episode + 1}")
    
    def on_next_episode(self):
        """Show next episode in list."""
        current_index = self.control_panel.episode_combo.currentIndex()
        if current_index < self.control_panel.episode_combo.count() - 1:
            self.control_panel.episode_combo.setCurrentIndex(current_index + 1)
    
    def on_prev_episode(self):
        """Show previous episode in list."""
        current_index = self.control_panel.episode_combo.currentIndex()
        if current_index > 0:
            self.control_panel.episode_combo.setCurrentIndex(current_index - 1)


def main():
    """Run RL training visualizer."""
    print("RL Viewer starting...", flush=True)
    mp.freeze_support()
    try:
        print("Parsing arguments...", flush=True)
        parser = argparse.ArgumentParser(description="RL Training Visualizer for Behavioral Weights")
        parser.add_argument('--model-dir', type=str, required=False, default=None,
                           help='Path to model directory with environment files (depth.tif, vel_x.tif, etc.)')
        parser.add_argument('--start-polygon', type=str, required=False, default=None,
                           help='Path to starting polygon shapefile')
        parser.add_argument('--longitudinal-profile', type=str, required=False, default=None,
                           help='Path to longitudinal profile shapefile (required for RL reward)')
        parser.add_argument('--hecras-plan', type=str, required=False, default=None,
                           help='Path to HECRAS plan HDF for direct mode')
        parser.add_argument('--hecras-start-index', type=int, required=False, default=30,
                           help='Start index into HECRAS time series (default: 30)')
        parser.add_argument('--hecras-time-mode', type=str, required=False, default='loop',
                           help='HECRAS time mode: time, index, loop, clamp, hold (default: loop)')
        parser.add_argument('--hecras-k', type=int, required=False, default=8,
                           help='HECRAS IDW neighbors (default: 8)')
        parser.add_argument('--hecras-cell-size', type=float, required=False, default=None,
                           help='Optional HECRAS grid cell size (m) for t0 raster build')
        parser.add_argument('--hecras-wetted-threshold', type=float, required=False, default=0.05,
                           help='Depth threshold for wetted mask at t0 (m, default: 0.05)')
        parser.add_argument('--model-name', type=str, default='salmon_abm',
                           help='Model name (default: salmon_abm)')
        parser.add_argument('--basin', type=str, default='nuyakuk',
                           help='Basin name (default: nuyakuk)')
        
        args = parser.parse_args()
        print(f"Args parsed: model_dir={args.model_dir}, start_polygon={args.start_polygon}", flush=True)

        # Keep UI scaling predictable in remote Xvfb/noVNC sessions.
        os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "0")
        os.environ.setdefault("QT_SCALE_FACTOR", "1")
        os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "0")
        
        print("Creating QApplication...", flush=True)
        app = QApplication(sys.argv)
        print("Creating RLTrainingViewer...", flush=True)
        viewer = RLTrainingViewer(
            model_dir=args.model_dir,
            start_polygon=args.start_polygon,
            model_name=args.model_name,
            basin=args.basin,
            hecras_plan_path=args.hecras_plan,
            hecras_start_index=args.hecras_start_index,
            hecras_time_mode=args.hecras_time_mode,
            hecras_k=args.hecras_k,
            hecras_cell_size=args.hecras_cell_size,
            hecras_wetted_threshold=args.hecras_wetted_threshold,
            longitudinal_profile=args.longitudinal_profile,
        )
        print("Showing viewer...", flush=True)
        viewer.show()
        viewer.raise_()  # Bring to front
        viewer.activateWindow()  # Activate window
        
        # Show startup guidance when launched without required inputs
        if not args.start_polygon and not args.hecras_plan:
            viewer.control_panel.append_log("=" * 50)
            viewer.control_panel.append_log("Welcome to RL Training Visualizer!")
            viewer.control_panel.append_log("")
            viewer.control_panel.append_log("Set input files in the 'Input Files' panel, then press Start Training.")
            viewer.control_panel.append_log("Minimum required:")
            viewer.control_panel.append_log("  - Start Polygon (.shp)")
            viewer.control_panel.append_log("  - Longitudinal Profile (.shp)")
            viewer.control_panel.append_log("  - HECRAS plan (.hdf/.h5) OR raster model-dir")
            viewer.control_panel.append_log("")
            viewer.control_panel.append_log("Optional: launch with CLI args to pre-fill these fields.")
            viewer.control_panel.append_log("=" * 50)
        
        print("Starting event loop...", flush=True)
        sys.exit(app.exec_())
    except Exception as e:
        import traceback
        print(f"FATAL ERROR: {e}", file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
