"""
data_pipeline.py

Preamble/Module plan for data ingestion and export.

Responsibilities (planned):
- Save episode traces, agent trajectories, and per-episode metrics to HDF5/CSV.
- Provide lightweight APIs for streaming writes that avoid per-timestep file opens.
- Functions:
  - `open_trace_writer(path)`
  - `write_timestep(trace_writer, timestep_data)`
  - `close_trace_writer(trace_writer)`

Notes:
- Prefer writing in append mode and ensure data integrity on crashes (explicit flush helpers).
"""
