import time
import importlib
import sys

# Import the simulation module to trigger module-level precompilation
try:
    import src.emergent.salmon_abm.sockeye_SoA as sock
except Exception as e:
    # try direct import path
    try:
        import emergent.salmon_abm.sockeye_SoA as sock
    except Exception:
        print('Failed to import simulation module:', e)
        sys.exit(1)

print('Imported sockeye_SoA, _HAS_NUMBA =', getattr(sock, '_HAS_NUMBA', False))

# Create a minimal simulation object if available
# The profile harness in tools/profile_timestep_cprofile.py constructs a small simulation.
# We'll attempt to call its main function if present.

try:
    import tools.profile_timestep_cprofile as prof
    print('Using existing profiling harness')
    prof.main()
except Exception as e:
    print('Could not run existing harness:', e)
    sys.exit(1)
