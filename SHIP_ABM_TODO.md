# Ship ABM - Environmental Forcing TODO List

**Last Updated**: October 2, 2025  
**Project**: Ship ABM with Real-World Ocean Currents & Winds  
**Status**: 🟡 Partially Complete - Ready for Testing

---

## ✅ COMPLETED (October 2, 2025)

### Ocean Currents System - 100% Working ✓
- [x] Fixed 6 critical bugs in `ofs_loader.py`
- [x] Updated file naming convention (`.fields.f###.nc` and `.2ds.f###.nc`)
- [x] Implemented fast KDTree nearest-neighbor interpolation (replaced slow Delaunay)
- [x] Added ROMS curvilinear grid support with staggered U/V grids
- [x] Added FVCOM unstructured grid support with element centers
- [x] Optimized to stop at first successful file (not search all 14 days)
- [x] Fixed RTOFS fallback logic
- [x] Added `u_sur`/`v_sur` variable names for WCOFS
- [x] Added Rosario Strait to config (8th harbor)
- [x] Comprehensive testing: **8/8 harbors working**
  - ✅ Baltimore (CBOFS ROMS): 12.0s, 0.032 m/s spatial variation
  - ✅ Galveston (NGOFS2 FVCOM): 9.3s, 0.044 m/s spatial variation
  - ✅ New Orleans (NGOFS2): 6.9s, 0.044 m/s spatial variation
  - ✅ San Francisco Bay (SFBOFS): 5.3s, 0.141 m/s spatial variation
  - ✅ Rosario Strait (SSCOFS): 5.4s, 0.160 m/s spatial variation
  - ✅ Seattle (SSCOFS): 7.3s, 0.053 m/s spatial variation
  - ✅ Los Angeles (WCOFS → tidal proxy): 54.6s
  - ✅ New York (NYOFS → tidal proxy): 113.7s

---

## 🟡 IN PROGRESS

### Wind System - 12.5% Working (1/8 harbors)
**Status**: Coordinate handling fixed, but additional issues discovered

#### Working Harbors:
- ✅ **Galveston (NGOFS2)**: 30.3s load time
  - Uses 1D station coordinates (x, y, lon, lat)
  - Spatial variation confirmed: 3.0 m/s U range, 1.8 m/s speed range
  - KDTree interpolation working correctly
  - Variables: `uwind_speed`, `vwind_speed`

#### Issues Found:
1. **Baltimore (CBOFS)** - KDTree Index Error
   - Problem: Uses 2D ROMS-style coordinates (`lon_rho`, `lat_rho`)
   - Error: "index 172 is out of bounds for axis 0 with size 61"
   - Cause: KDTree built on 2D grid but indexing mismatched with cropped data
   - Fix needed: Handle 2D coordinates in `build_wind_sampler()` in `atmospheric.py`
   - Variables: `Uwind`, `Vwind` (different naming than NGOFS2!)

2. **ERA5 Data Access** - 403 Forbidden
   - AWS S3 bucket `era5-pds` requires authentication
   - Fallback logic catches this correctly
   - Not critical since NOAA OFS fallback works

3. **HRRR Data Access** - Not tested yet
   - May have date/time issues (files only kept ~14 days)
   - May also require authentication
   - Fallback to ERA5 then NOAA OFS should handle this

4. **Remaining Harbors** - Not yet tested:
   - ❓ Los Angeles (WCOFS)
   - ❓ New Orleans (NGOFS2) - Should work like Galveston
   - ❓ New York (NYOFS)
   - ❓ San Francisco Bay (SFBOFS)
   - ❓ Rosario Strait (SSCOFS)
   - ❓ Seattle (SSCOFS)

---

## 📋 HIGH PRIORITY TODO

### 1. Test Galveston Ship Simulation 🎯 **NEXT STEP**
**Timeline**: Now (October 2, 2025)  
**Files**: `scripts/run_ship.py` or create new test script  
**Goal**: Validate end-to-end simulation with currents + winds

**Test Parameters**:
- Harbor: Galveston (only fully working wind+current harbor)
- Number of ships: 2-5 (small test)
- Duration: 1-2 hours simulated time
- Timestep: 1-10 seconds
- Environmental forcing: Both currents (NGOFS2) + winds (NGOFS2 stations)

**Success Criteria**:
- [ ] Ships spawn correctly
- [ ] Ships respond to current drift
- [ ] Ships respond to wind drift
- [ ] COLREGS collision avoidance works
- [ ] PID controllers compensate for drift
- [ ] No crashes or errors
- [ ] Reasonable computation time (<5 min real time per 1 hour sim time)

**Validation Checks**:
- Check that `current_fn()` returns non-zero values
- Check that `wind_fn()` returns non-zero values (we see 5.078 m/s wind in tests)
- Verify ship tracks show drift compensation
- Monitor console for environmental forcing debug output

---

### 2. Fix Baltimore Wind KDTree Issue
**Timeline**: After Galveston simulation succeeds  
**Priority**: High  
**Estimated Time**: 30-60 minutes

**Problem**: 2D ROMS coordinates cause dimension mismatch in KDTree indexing

**Solution Approach**:
```python
# In atmospheric.py, build_wind_sampler():
# Check if lon/lat are 2D (ROMS grid) vs 1D (stations)
if lon.ndim == 2:
    # Flatten for KDTree but track original shape
    lon_flat = lon.ravel()
    lat_flat = lat.ravel()
    tree = cKDTree(np.column_stack((lon_flat, lat_flat)))
    # Index into flattened arrays
else:
    # Existing 1D station logic
    tree = cKDTree(np.column_stack((lon, lat)))
```

**Files to Modify**:
- `src/emergent/ship_abm/atmospheric.py` lines ~120-180 in `build_wind_sampler()`

**Testing**:
- Run `test_all_harbors_wind.py` after fix
- Baltimore should work
- Galveston should still work

---

### 3. Complete Wind Testing for All Harbors
**Timeline**: After Baltimore fix  
**Priority**: Medium  
**Estimated Time**: 1-2 hours (mostly wait time for S3 downloads)

**Remaining Tests**:
- [ ] Los Angeles (WCOFS) - May use 2D ROMS coords
- [ ] New Orleans (NGOFS2) - Should work like Galveston
- [ ] New York (NYOFS) - May not have station files
- [ ] San Francisco Bay (SFBOFS) - Unknown
- [ ] Rosario Strait (SSCOFS) - Unknown  
- [ ] Seattle (SSCOFS) - Same model as Rosario

**Expected Outcomes**:
- NGOFS2 harbors (New Orleans) should work immediately
- ROMS harbors (LA) need Baltimore fix first
- Some harbors may not have MET/stations files → fallback to uniform wind

---

### 4. Wind Fallback Strategy (if HRRR/ERA5/NOAA fail)
**Timeline**: As needed  
**Priority**: Low-Medium

**Options**:
1. **Uniform Wind Field**: Single wind value for entire domain
   - Simplest fallback
   - No spatial variation
   - Could use NOAA buoy data or weather station

2. **Synthetic Wind Model**: 
   - Generate wind field based on pressure gradients
   - More realistic but complex

3. **Historical Climatology**:
   - Pre-computed average wind fields by month/season
   - Static but better than nothing

**Recommendation**: Start with option 1 (uniform wind) for harbors without station data

---

## 📋 MEDIUM PRIORITY TODO

### 5. Performance Optimization
**Timeline**: After all harbors working  
**Priority**: Medium  
**Estimated Time**: 2-4 hours

**Caching System**:
- [ ] Cache downloaded netCDF datasets to disk
- [ ] Check cache before S3 download
- [ ] Implement cache expiration (24-48 hours for forecasts)
- [ ] Reduce redundant downloads for multi-hour simulations

**Time Interpolation**:
- [ ] Currently using nearest-hour (no interpolation)
- [ ] Implement linear interpolation between forecast hours
- [ ] Smoother transitions for long simulations

**Spatial Interpolation**:
- [ ] Consider bilinear for ROMS structured grids (vs nearest-neighbor)
- [ ] May improve accuracy near coastlines
- [ ] Profile performance impact

**Expected Gains**:
- First run: 30-120s (download + process)
- Subsequent runs: <5s (cache hit)
- Multi-hour sims: 1 download for entire time range

---

### 6. Add More Harbors
**Timeline**: As needed for specific studies  
**Priority**: Low-Medium

**Candidates**:
- [ ] Houston Ship Channel (Texas)
- [ ] Port of Savannah (Georgia)
- [ ] Port of Long Beach (California) - separate from LA
- [ ] Columbia River (Oregon/Washington)
- [ ] Mobile Bay (Alabama)
- [ ] San Diego (California)

**Process**:
1. Add bounds to `config.py` SIMULATION_BOUNDS
2. Identify NOAA OFS model coverage (check CO-OPS website)
3. Add to OFS_MODEL_MAP
4. Test with `test_all_harbors.py` and `test_all_harbors_wind.py`

---

### 7. Visualization Enhancements
**Timeline**: After simulations running smoothly  
**Priority**: Medium  
**Estimated Time**: 4-8 hours

**Vector Field Overlay**:
- [ ] Show current vectors on ship viewer
- [ ] Show wind vectors on ship viewer
- [ ] Color code by magnitude
- [ ] Update in real-time during simulation

**Ship Tracks**:
- [ ] Plot historical track with color gradient (time)
- [ ] Show planned path vs actual path (drift effects)
- [ ] Export tracks to GeoJSON/Shapefile

**Environmental Data Inspector**:
- [ ] Click on map to query current/wind at that location
- [ ] Time series plots of environmental forcing
- [ ] Statistics panel (min/max/mean current speed, wind speed)

---

## 📋 LOW PRIORITY TODO

### 8. Documentation
**Timeline**: Ongoing  
**Priority**: Low

- [ ] Document file naming conventions (NOAA OFS models)
- [ ] API documentation for `get_current_fn()` and `wind_sampler()`
- [ ] Example notebooks for common use cases
- [ ] Troubleshooting guide for S3 access issues

---

### 9. Unit Tests
**Timeline**: After system stabilizes  
**Priority**: Low

- [ ] Test ROMS grid handler
- [ ] Test FVCOM grid handler  
- [ ] Test KDTree interpolation accuracy
- [ ] Test fallback logic (regional → RTOFS → tidal proxy)
- [ ] Test wind coordinate handling (1D vs 2D)

---

### 10. Advanced Features
**Timeline**: Future enhancements  
**Priority**: Low

**Waves**:
- [ ] Add wave height/period/direction from WaveWatch III
- [ ] Impact on ship motion (additional forcing)

**Ice**:
- [ ] Add sea ice concentration for Arctic simulations
- [ ] Ship performance degradation in ice

**Tides**:
- [ ] Replace tidal proxy with NOAA CO-OPS tidal constituents
- [ ] More accurate for harbors without OFS coverage

**Weather**:
- [ ] Add visibility, precipitation
- [ ] Impact on navigation decisions

---

## 🔧 KNOWN ISSUES

### Critical (Blocking):
- None currently - Galveston simulation ready to test!

### High Priority:
1. **Baltimore Wind KDTree Error** (see TODO #2)
2. **ERA5 Access** - 403 Forbidden (non-blocking, fallback works)

### Medium Priority:
3. **WCOFS Surface Files** - Only u_sur/v_sur, no 3D data
4. **NYOFS Availability** - Limited recent data, falls back to tidal proxy
5. **Wind Variable Names** - Not standardized (uwind_speed vs Uwind vs u10)

### Low Priority:
6. **Load Times** - 5-120s depending on model/harbor (acceptable but could cache)
7. **No Spatial Variation** - Some harbors only have 1-2 stations (uniform wind)

---

## 📊 SYSTEM ARCHITECTURE SUMMARY

### Data Flow:
```
1. User starts simulation → config.py (harbor bounds, model mapping)
2. simulation_core.py initializes environment:
   ├─ get_current_fn(port, datetime) → ofs_loader.py
   │  ├─ Try regional model (CBOFS, NGOFS2, etc.)
   │  ├─ Fallback to RTOFS (global)
   │  └─ Fallback to tidal proxy (synthetic)
   │
   └─ wind_sampler(bbox, datetime) → atmospheric.py
      ├─ Try HRRR (3km CONUS)
      ├─ Try ERA5 (31km global)
      └─ Try NOAA OFS MET/stations
      
3. Every timestep (1-10s):
   ├─ current_fn(lons, lats, time) → (u, v) currents
   ├─ wind_fn(lons, lats, time) → (u, v) winds
   └─ ship_model.py applies forces → Fossen dynamics → new position

4. Loop until simulation end
```

### Grid Types Supported:
- **ROMS**: Curvilinear C-grid, staggered U/V, separate KDTrees
- **FVCOM**: Unstructured triangular, element centers (lonc/latc)
- **RTOFS**: Regular lat/lon grid
- **Stations**: 1D or 2D point data

### Interpolation Methods:
- **Spatial**: KDTree nearest-neighbor (O(log n) fast)
- **Temporal**: Nearest hour (no interpolation yet)

---

## 🎯 IMMEDIATE NEXT STEPS

**RIGHT NOW** (Next 30 minutes):
1. ✅ Create this TODO list
2. 🔄 **Run Galveston ship simulation** with currents + winds
3. 📊 Analyze results, validate environmental forcing works

**TODAY** (October 2, 2025):
- Fix Baltimore wind KDTree issue
- Test remaining NGOFS2 harbors (New Orleans)
- Document Galveston simulation results

**THIS WEEK**:
- Complete wind testing for all 8 harbors
- Implement basic caching system
- Run multi-hour simulations

**THIS MONTH**:
- Add visualization enhancements
- Performance profiling and optimization
- Consider adding more harbors

---

## 📈 SUCCESS METRICS

### Phase 1: Environmental Forcing (CURRENT STATUS)
- [x] Currents: 8/8 harbors working (100%)
- [~] Winds: 1/8 harbors fully working (12.5%)
- Target: 8/8 by end of week

### Phase 2: Ship Simulation Integration
- [ ] Single ship test: Galveston (NEXT)
- [ ] Multi-ship test: 5-10 ships
- [ ] Collision avoidance validation
- [ ] Target: By end of today

### Phase 3: Production Ready
- [ ] All 8 harbors with winds working
- [ ] Caching implemented
- [ ] <10s startup time (cached)
- [ ] <1 min/hour simulation time
- Target: By end of week

---

## 🚀 LAUNCH CHECKLIST - Galveston Simulation

Before running first real simulation:
- [x] Ocean currents working (NGOFS2)
- [x] Winds working (NGOFS2 stations)
- [x] Spatial variation confirmed (currents: 0.044 m/s, winds: 3.0 m/s)
- [ ] Test script created
- [ ] Ships spawn in valid water locations
- [ ] Reasonable simulation parameters chosen
- [ ] Output directory prepared
- [ ] Monitoring/logging enabled

**Ready to launch!** 🚢

---

## 📞 CONTACTS & RESOURCES

**NOAA Data Sources**:
- Ocean Currents: `s3://noaa-nos-ofs-pds/`
- HRRR Winds: `s3://noaa-hrrr-pds/`
- ERA5 Winds: `s3://era5-pds/` (requires auth)

**Documentation**:
- NOAA CO-OPS: https://tidesandcurrents.noaa.gov/ofs/
- ROMS: https://www.myroms.org/
- FVCOM: http://fvcom.smast.umassd.edu/

**Key Files**:
- Current loading: `src/emergent/ship_abm/ofs_loader.py` (876 lines)
- Wind loading: `src/emergent/ship_abm/atmospheric.py` (235 lines)
- Configuration: `src/emergent/ship_abm/config.py` (350 lines)
- Ship dynamics: `src/emergent/ship_abm/ship_model.py`
- Simulation core: `src/emergent/ship_abm/simulation_core.py`

---

**Last Updated**: October 2, 2025 - Ready for Galveston simulation test! 🎉
