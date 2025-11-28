# Emergent 

Python software using agent based models to solve complex spatial problems.  This software was written alongside my dissertation (Nebiolo 2017).   

**Agent-Based Modeling Framework for Spatial Emergent Phenomena**

## Agent Based Models

Python software using agent-based models to solve complex spatial problems in maritime navigation and aquatic ecology. This software was developed alongside Dr. Kevin Nebiolo's dissertation research (Nebiolo 2017).ABMs are a collection of autonomous, goal directed software objects, capable of interacting with other agents, reacting to their environment, and making decisions that maximize their own well being.



[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)ABMs simulate complex adaptive systems, what we see as random movement is not.  There are no stochastic processes in these models other than setting intial states.  Movement therefore is chaotic, once the simulation starts the agents are making decisions on their own - we cannot predict their behavior, it has to unfold.  The goal of these applications is to produce self organized emergent behavior of interest to managers.  Emergence results from individual interaction governed by simple behavioral rules.  Traffic jams the result of congestion, not construction or an accident, are an example of emergence and are known as kinematic waves in the literature.  

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![Status](https://img.shields.io/badge/status-alpha-orange)](https://github.com/knebiolo/emergent)## Getting Involved

If you are a Python developer and want to help out, please send an email to kevin.nebiolo@kleinschmidtgroup.com.  

---

Otherwise, feel free to post in Discussions.  

## What is Emergent?

**Emergent** is a framework for simulating complex adaptive systems where simple individual behaviors lead to sophisticated group patterns. The software includes two major agent-based models:

### 1. **Ship ABM** - Maritime Navigation with Real-World Forcing 
Simulate vessel traffic in harbors and waterways with realistic environmental conditions:
- **Real-time ocean currents** from NOAA Operational Forecast Systems (OFS)
- **Real-time wind data** from HRRR, ERA5, and NOAA meteorological models
- **4-DOF ship dynamics** using Fossen hydrodynamic model
- **COLREGS collision avoidance** (International Regulations for Preventing Collisions at Sea)
- **PID autopilot** with environmental drift compensation
- **Interactive GUI** for route planning and visualization

**Supported Harbors** (with real NOAA data):
- Baltimore, MD (Chesapeake Bay)
- Galveston, TX (Gulf of Mexico)  **Fully operational with currents + winds**
- New Orleans, LA (Gulf of Mexico)
- San Francisco Bay, CA (Pacific Coast)
- Seattle, WA (Puget Sound)
- Rosario Strait, WA (Salish Sea)
- Los Angeles, CA (Southern California)
- New York, NY (Atlantic Coast)

### 2. **Salmon ABM** - Fish Passage and Migration 
Model fish behavior and movement through river systems and hydroelectric facilities:
- Energetic-based decision making
- Hydraulic forcing from HEC-RAS models
- Schooling behavior
- Turbine passage and survival

---

## Quick Start

### Installation

#### Option 1: pip (recommended)
```bash
# Clone the repository
git clone https://github.com/knebiolo/emergent.git
cd emergent

# Install in development mode
pip install -e .

# Or install from requirements.txt
pip install -r requirements.txt
```

#### Option 2: conda
```bash
# Create environment from environment.yml
conda env create -f environment.yml
conda activate emergent

# Install the package
pip install -e .
```

### System Requirements
- **Python**: 3.10 or higher
- **Operating System**: Windows, macOS, or Linux
- **RAM**: 8GB minimum, 16GB+ recommended for large simulations
- **Internet**: Required for downloading NOAA/HRRR environmental data from AWS S3

---

## Usage

### Ship ABM - Interactive GUI

Launch the ship simulation GUI:

```python
python scripts/run_ship.py
```

Note: The launcher now preloads ENC data by default for faster, consistent visualization. To skip loading ENCs (faster startup but without chart overlays), pass the `--no-enc` flag:

```python
python scripts/run_ship.py --no-enc
```

**Workflow**:
1. GUI window opens with harbor map
2. Click **"Define Route"** button
3. Click waypoints on the map to create a route
4. Click **"Add Ship"** to spawn a vessel
5. Click **"Start Simulation"** to begin
6. Watch ships navigate with real currents and winds!

### Ship ABM - Programmatic

```python
from emergent.ship_abm.ofs_loader import get_current_fn
from emergent.ship_abm.atmospheric import wind_sampler
from emergent.ship_abm.config import SIMULATION_BOUNDS
from datetime import datetime
import numpy as np

# Choose harbor
harbor = "Galveston"
bounds = SIMULATION_BOUNDS[harbor]
bbox = (bounds['minx'], bounds['maxx'], bounds['miny'], bounds['maxy'])

# Load environmental forcing
current_fn = get_current_fn(harbor, datetime.now())
wind_fn = wind_sampler(bbox, datetime.now())

# Sample at a location
lon, lat = -95.0, 29.5
currents = current_fn(np.array([lon]), np.array([lat]), datetime.now())
winds = wind_fn(np.array([lon]), np.array([lat]), datetime.now())

print(f"Current: {currents[0]} m/s")
print(f"Wind: {winds[0]} m/s")
```

---

## Environmental Data Sources

### Ocean Currents (NOAA OFS)
- **CBOFS**: Chesapeake Bay (ROMS, 10s cycles)
- **NGOFS2**: Northern Gulf of Mexico (FVCOM, 500k+ elements) 
- **SFBOFS**: San Francisco Bay (FVCOM, 100k+ elements)
- **SSCOFS**: Salish Sea / Puget Sound (FVCOM, 430k+ elements)
- **WCOFS**: West Coast (ROMS, surface-only)
- **RTOFS**: Global fallback (1/12° resolution)

All data accessed anonymously from AWS S3: `s3://noaa-nos-ofs-pds/`

**Performance**: 5-15s load time, <1ms spatial queries

---

## Features

### Ship ABM
- ✅ **Real ocean currents** from 6 NOAA operational forecast models
- ✅ **Real wind data** from HRRR/ERA5/NOAA
- ✅ **Fast spatial interpolation** with KDTree (O(log n))
- ✅ **4-DOF ship dynamics** (surge, sway, roll, yaw)
- ✅ **PID autopilot** with environmental compensation
- ✅ **COLREGS collision avoidance**
- ✅ **Interactive route planning**
- ✅ **Real-time visualization**
- ✅ **8 US harbors** with real NOAA data
- 🔄 **Caching system** (in progress)
- 🔄 **Time interpolation** (in progress)

---

## Development

### Project Structure
```
emergent/
├── src/emergent/
│   ├── ship_abm/           # Ship navigation ABM
│   │   ├── ofs_loader.py   # Ocean current data loader (876 lines)
│   │   ├── atmospheric.py  # Wind data loader (235 lines)
│   │   ├── ship_model.py   # 4-DOF ship dynamics
│   │   ├── simulation_core.py  # Main simulation loop
│   │   ├── ship_viewer.py  # PyQt5 GUI
│   │   └── config.py       # Harbor configurations
│   └── salmon_abm/         # Fish migration ABM
├── scripts/                # Example scripts
├── setup.py                # Installation script  ✅ NEW
├── requirements.txt        # Dependencies  ✅ NEW
├── environment.yml         # Conda environment  ✅ NEW
├── SHIP_ABM_TODO.md       # Development roadmap  ✅ NEW
└── README.md              # This file  ✅ UPDATED
```

### Running Tests

```bash
# Test all harbors (currents) - ALL 8 WORKING! ✅
python scripts/test_all_harbors.py

# Test all harbors (winds) - 1/8 working, 7 in progress
python scripts/test_all_harbors_wind.py

# Test Galveston simulation (currents + winds working!)
python scripts/test_galveston_simulation.py
```

### Contributing

See **`SHIP_ABM_TODO.md`** for detailed roadmap and priorities.

---

## Known Issues

See `SHIP_ABM_TODO.md` for complete list. Key issues:
1. Wind loading works for NGOFS2 (Galveston, New Orleans) but needs fixes for ROMS models
2. ERA5 requires authentication (falls back to NOAA OFS)
3. Some harbors have limited station coverage (uniform wind fields)

---

## Citation

```bibtex
@phdthesis{nebiolo2017emergent,
  title={Agent-Based Modeling of Complex Spatial Systems},
  author={Nebiolo, Kevin},
  year={2017},
  note={Software: https://github.com/knebiolo/emergent}
}
```

---

## Contact

**Kevin Nebiolo, PhD**  
Senior Scientist, Kleinschmidt Associates  
Email: kevin.nebiolo@kleinschmidtgroup.com

**Getting Involved**:
- Developers: Email above
- Users: Post in GitHub Discussions
- Bugs: GitHub Issues

---

## License

MIT License - see [LICENSE](LICENSE) file

---

## Acknowledgments

- **NOAA**: Operational Forecast Systems and public AWS data
- **NCEP**: HRRR high-resolution weather data
- **Kleinschmidt Associates**: Research support

---

## Recent Updates

### October 2, 2025 - Infrastructure Improvements
- ✅ Created `setup.py` for proper package installation
- ✅ Created `requirements.txt` with all dependencies
- ✅ Created `environment.yml` for conda users
- ✅ Created `SHIP_ABM_TODO.md` with detailed roadmap
- ✅ Improved README with comprehensive documentation
- ✅ Fixed ocean current loading (8/8 harbors working!)
- 🔄 Wind loading in progress (1/8 harbors fully functional)

---

