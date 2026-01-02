# Silent Failure Cleanup Plan

**Goal:** Eliminate `try/except Exception: pass` blocks and make failures loud & debuggable

**Date:** 2026-01-01
**Priority:** HIGH - Currently masking behavioral bugs

---

## Summary

Found **100+ `except Exception:` blocks** in `behavior.py` alone. These silent failures hide:
- Data quality issues (e.g., -9999 nodata values)
- Missing environment data
- Shape mismatches
- NaN propagation
- Configuration errors

**Current Impact:** Just spent hours debugging rheotaxis because velocity nodata values silently passed validation.

---

## Phased Cleanup Strategy

### Phase 1: Critical Path - Behavioral Cues (Priority 1) 🔴
**Target:** Lines where silent failures corrupt agent decision-making

**Files:** `behavior.py`

**Focus Areas:**
1. **Cue calculation methods** (rheotaxis, shallow, border, alignment, cohesion, etc.)
   - Lines 1729-1789: `rheo_cue()` - JUST FIXED nodata issue here
   - Lines 1862-1896: `shallow_cue()` 
   - Lines 1787-1860: `border_cue()`
   - Lines 1982-2070: `cohesion_cue()` and `alignment_cue()`
   
2. **Environment sampling** (`_get_env()`, `sample_environment()`)
   - Failures here propagate to ALL cues
   - Need explicit checks: "Dataset X missing" not silent zeros
   
3. **Arbitration logic** (lines 2410-2870)
   - Lines 2547-2549: Tolerance check during accumulation
   - Lines 2587-2593: swim_behav mode selection
   - Lines 2799-2857: Fallback heading calculation

**Action Items:**
- [ ] Audit ALL cue methods for silent failures
- [ ] Add explicit nodata value checks (like the -9999 fix)
- [ ] Replace `except Exception: pass` with specific error types
- [ ] Add validation: `if not np.isfinite(result).all(): raise ValueError(...)`
- [ ] Log warnings for degraded mode (e.g., "cohesion disabled, no neighbors")

**Expected Timeline:** 2-3 sessions

---

### Phase 2: Data Loading & Validation (Priority 2) 🟡
**Target:** Lines where missing/corrupt environment data causes silent degradation

**Files:** `simulation.py`, `hdf5_io.py`, `behavior.py`

**Focus Areas:**
1. HDF5 environment loading
   - Velocity fields (vel_x, vel_y)
   - Depth rasters
   - Temperature, refuge masks
   
2. Transform validation
   - Affine transforms for raster sampling
   - Coordinate conversions
   
3. Initial state setup
   - Agent positions
   - Initial headings (currently zeros → fallback issues)

**Action Items:**
- [ ] Add dataset existence checks: `if 'environment/vel_x' not in h5: raise ...`
- [ ] Validate raster shapes match transforms
- [ ] Check for nodata coverage: "Warning: 45% of domain is nodata"
- [ ] Initialize headings from velocity field, not zeros

**Expected Timeline:** 1-2 sessions

---

### Phase 3: Utilities & Helpers (Priority 3) 🟢
**Target:** Lower-impact utilities where failures are less critical

**Files:** `utils.py`, `hdf5_io.py`

**Focus Areas:**
1. Coordinate transformations
2. Pixel/geo conversions
3. Distance calculations
4. Cache management

**Action Items:**
- [ ] Keep some try/except for optional features (e.g., psutil, numba)
- [ ] Make core geometry functions fail loud
- [ ] Add input validation with clear errors

**Expected Timeline:** 1 session

---

### Phase 4: Optional Dependencies (Acceptable) ✅
**Target:** Cases where silent fallback IS appropriate

**Examples:**
- `try: import psutil` - optional memory monitoring
- `try: import numba` - optional acceleration
- `try: cache.get(key)` - cache miss is expected

**Pattern:**
```python
try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:  # Specific exception type
    _PSUTIL_AVAILABLE = False
    logging.info("psutil not available, memory monitoring disabled")
```

**Action Items:**
- [ ] Document why each fallback exists
- [ ] Use specific exception types (ImportError, KeyError, etc.)
- [ ] Log at INFO level for degraded functionality

---

## Testing Strategy

After each phase:
1. **Run simulation** with debug logging enabled
2. **Inject bad data** (NaN velocities, missing datasets, wrong shapes)
3. **Verify loud failure** with actionable error messages
4. **Check viewer** - do agents behave correctly?

**Success Criteria:**
- Zero `except Exception: pass` in critical path
- Errors include: file/line, variable values, expected vs actual
- No silent degradation (e.g., "rheotaxis disabled because nodata" should ERROR not WARN)

---

## Immediate Next Steps (Current Session)

1. ✅ Add .ai_journal to .gitignore
2. ✅ Create this cleanup plan
3. **Test current fix:** Run simulation, verify agents swim upstream
4. **If still broken:** Add debug logging to rheo_cue() to see sampled velocities
5. **Start Phase 1:** Pick one cue method (shallow_cue?), remove try/except, add validation

---

## Phase 4: Performance Optimization (Priority 4) 🟢
**Target:** Vectorize remaining Python loops in simulation.py for production-scale runs

**Files:** `simulation.py`

**Focus Areas:**
1. **Neighbor graph construction** (lines ~1160-1195)
   - `tree.query_ball_point()` is fast (cKDTree C implementation)
   - **BOTTLENECK:** Lines 1176-1183 - Python loop converting neighbor lists to CSR format
   - With 5000 agents × 5000 neighbors = 25M iterations
   - **CURRENT FIX:** Disabled `agents_within_buffers` list comprehension (line 1191-1195)
   - Alignment/cohesion use CSR directly, but may need Numba kernel for CSR construction
   
2. **Other Python loops in simulation.py** (from 2025-12-31 session notes)
   - Line 816: Unknown loop (needs profiling)
   - Line 1023: Unknown loop (needs profiling)
   - Line 1516: Unknown loop (needs profiling)

**Action Items:**
- [ ] Profile simulation.py to identify remaining bottlenecks
- [ ] Add Numba JIT kernel for CSR neighbor list flattening (lines 1176-1183)
- [ ] Consider caching neighbor graph for multiple timesteps (already implemented with `neighbor_update_seconds`)
- [ ] Vectorize loops at lines 816, 1023, 1516 (identify first via profiling)
- [ ] Benchmark: Target 15k+ timesteps/second for 5000-agent runs

**Expected Timeline:** 1-2 sessions

**Context:** 
- Yesterday optimized movement.py and fatigue.py with Numba (10-30x speedup)
- simulation.py neighbor building was never optimized - still pure Python loops
- Production runs (5000 agents, 900 steps) now bottlenecked on neighbor graph construction

---

## What I Wish I Was Told

- The -9999 nodata bug would have been caught in 30 seconds if `rheo_cue()` had:
  ```python
  if (np.abs(v) > 9990).any():
      raise ValueError(f"Nodata values in velocity: {v[np.abs(v) > 9990]}")
  ```
- Every hour debugging silent failures saves 10 hours of future pain
- "Fail fast, fail loud, fail with context" should be carved into the codebase

---

## Pattern Recognition

**BAD (current pattern):**
```python
try:
    result = do_something_critical()
except Exception:
    result = np.zeros(...)  # Silent corruption
```

**GOOD (new pattern):**
```python
result = do_something_critical()
if not validate(result):
    raise ValueError(f"do_something_critical() failed validation: {diagnostic_info}")
```

**ACCEPTABLE (optional features):**
```python
try:
    optional_feature()
except SpecificError as e:
    logging.warning(f"Optional feature disabled: {e}")
    _FEATURE_AVAILABLE = False
```

---

## Notes

- This is technical debt from rapid prototyping
- User base = 1, prefer debuggability over robustness
- Future: Add `--strict` mode that crashes on ANY warning
