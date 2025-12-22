# Biological Schooling Metrics for Salmon ABM

## Overview

This document explains the biologically-grounded schooling metrics implemented in the RL training system for the salmon agent-based model. These metrics replace simplistic global averages with local, sensory-based measurements that reflect how real fish perceive and interact with their neighbors.

---

## Key Biological Principles

### 1. Sensory Perception Range: 2 Body Lengths

**Biological basis:** Fish primarily rely on their lateral line system and vision to detect nearby neighbors. Studies show sockeye salmon respond most strongly to neighbors within **2 body lengths** (BL).

**Implementation:**
```python
search_radius = 2.0 * body_length  # Meters
neighbor_indices = tree.query_ball_point(position, r=search_radius)
```

**Reference:** Partridge & Pitcher (1980) showed schooling fish maintain spacing relative to body size, with neighbor detection range ~1.5-3.0 BL.

---

### 2. Dynamic Cohesion Based on Threat Level

**Biological basis:** Fish schools exhibit **threat-responsive behavior**:
- **Relaxed state:** Loose formation (3.0 BL spacing) for efficient foraging
- **Threatened state:** Tight ball (1.5 BL spacing) for predator confusion

**Implementation:**
```python
threat = behavioral_weights.threat_level  # 0.0 = calm, 1.0 = high threat
cohesion_radius_BL = (
    cohesion_radius_relaxed * (1 - threat) + 
    cohesion_radius_threatened * threat
)
```

**Reference:** Magurran & Pitcher (1987) documented shoal compression under predation risk in fish populations.

---

### 3. Drafting Benefits: Energy-Efficient Formations

**Biological basis:** Swimming behind another fish reduces drag due to wake effects:
- **Single draft:** 15% drag reduction when behind one fish
- **V-formation draft:** 25% drag reduction when flanked by two fish

**Implementation:**
```python
# Check if agent is swimming in another's wake
if agent_behind_neighbor and angle_aligned:
    if left_flanker and right_flanker:
        drag_reduction = 0.25  # V-formation
    else:
        drag_reduction = 0.15  # Single draft
```

**Reference:** 
- Weihs (1973): Hydrodynamic analysis of fish schooling showed 15-20% energy savings
- Fish & Lauder (2006): PIV studies confirmed vortex-based thrust augmentation in fish formations

---

## Schooling Quality Metrics

### Cohesion Score (0-1)

**Measures:** Proximity to ideal group spacing

**Calculation:**
```python
# Distance from agent to local centroid of neighbors
centroid = mean(neighbor_positions)
dist_to_centroid = ||position - centroid||

# Ideal distance adjusted by threat level
ideal_dist = 2.0 * BL * (1 - 0.5 * threat)

# Gaussian reward centered at ideal
cohesion_score = exp(-0.5 * ((dist - ideal_dist) / (0.5*BL))^2)
```

**Interpretation:**
- 1.0 = Perfect spacing (at ideal distance from group center)
- 0.5 = Moderate deviation (±0.5 BL from ideal)
- 0.0 = Large deviation (>2 BL from ideal)

---

### Alignment Score (-1 to 1)

**Measures:** Heading similarity with neighbors

**Calculation:**
```python
# Circular mean of neighbor headings
mean_heading = arctan2(mean(sin(neighbor_headings)), 
                       mean(cos(neighbor_headings)))

# Angular difference to my heading
heading_diff = my_heading - mean_heading

# Cosine similarity
alignment_score = cos(heading_diff)
```

**Interpretation:**
- 1.0 = Perfect alignment (same direction as neighbors)
- 0.0 = Perpendicular (90° difference)
- -1.0 = Opposite direction (180° difference)

---

### Separation Penalty (-1 to 0)

**Measures:** Crowding penalty for agents too close

**Calculation:**
```python
nearest_neighbor_dist = min(distances_to_all_agents)

if nearest_neighbor_dist < 1.0 * BL:
    separation_penalty = -(1.0*BL - dist) / BL
else:
    separation_penalty = 0.0
```

**Interpretation:**
- 0.0 = No crowding (>1 BL clearance)
- -0.5 = Moderate crowding (0.5 BL apart)
- -1.0 = Severe crowding (touching)

---

### Overall Schooling Score

**Combines all three metrics:**
```python
overall = cohesion_score + alignment_score + separation_penalty
```

**Range:** -1 to 2
- **Excellent schooling:** ~2.0 (perfect cohesion + alignment, no crowding)
- **Good schooling:** 1.0-1.5
- **Poor schooling:** <0.5
- **Dysfunctional:** <0.0 (isolated, misaligned, or crowded)

---

## Energy Efficiency Metrics

### Drag Reduction from Drafting

**Per-agent calculation:**
```python
drag_reductions = compute_drafting_benefits(positions, headings, velocities, ...)
# Returns (N,) array: [0.0, 0.15, 0.25, 0.0, ...]
```

**Energy accounting:**
```python
base_energy = speed^2  # Drag proportional to velocity squared
adjusted_energy = base_energy * (1.0 - drag_reduction)
efficiency = total_distance / total_energy
```

**Reward component:**
```python
drafting_reward = mean(drag_reductions) * 20.0  # Scale 0.0-0.25 → 0-5
efficiency_reward = energy_efficiency * 2.0
```

---

## Reward Function Breakdown

### Component Weights

| Component | Metric | Weight | Range | Impact |
|-----------|--------|--------|-------|--------|
| **Cohesion** | 0-1 | 10.0 | 0-10 | Core schooling quality |
| **Alignment** | -1 to 1 | 5.0 | 0-10 | Heading coordination |
| **Separation** | -1 to 0 | 5.0 | -5 to 0 | Crowding penalty |
| **Upstream progress** | meters/step | 0.5 | varies | Migration efficiency |
| **Energy efficiency** | m/kcal | 2.0 | varies | Metabolic cost |
| **Drafting benefit** | 0.0-0.25 | 20.0 | 0-5 | Formation quality |
| **Boundary proximity** | count | -5.0 | varies | Safety penalty |
| **Mortality** | count | -50.0 | varies | Strong survival penalty |
| **Smoothness** | Δaccel | -0.2 | varies | Movement efficiency |

### Total Reward Calculation

```python
reward = (
    cohesion_score * 10.0 +
    (alignment_score + 1.0) * 5.0 +  # Shift -1:1 → 0:2, scale to 0:10
    separation_penalty * 5.0 +
    mean_upstream_progress * 0.5 +
    energy_efficiency * 2.0 +
    mean_drafting_benefit * 20.0 +
    agents_near_boundary * -5.0 +
    dead_count * -50.0 +
    accel_smoothness * -0.2
)
```

**Expected ranges:**
- **Perfect schooling episode:** 20-30 (tight formation, efficient migration)
- **Good episode:** 10-20 (coordinated but imperfect)
- **Poor episode:** 0-10 (fragmented, inefficient)
- **Catastrophic episode:** <0 (high mortality, bank collisions)

---

## Biological Validation Criteria

### Realistic Schooling Behavior

✅ **Local interactions only:** Agents only respond to neighbors within 2 BL (sensory range)  
✅ **Dynamic cohesion:** Tight ball under threat, loose formation when relaxed  
✅ **Energy-aware formations:** Reward drafting behind neighbors  
✅ **Separation enforcement:** Prevent unrealistic crowding (<1 BL)  

### Physiological Realism

✅ **Energy proportional to speed²:** Reflects hydrodynamic drag  
✅ **Drafting reduces energy:** 15-25% savings from formation swimming  
✅ **Efficiency metric:** Distance per kcal, not just speed  

### Emergent Behavior Goals

🎯 **Upstream migration:** Reward net forward movement  
🎯 **Schooling instinct:** Balance cohesion/alignment/separation  
🎯 **Predator avoidance:** Boundary safety, threat-responsive cohesion  
🎯 **Energy conservation:** Drafting utilization, smooth swimming  

---

## Tuning Recommendations

### Initial Training (Episodes 1-50)

Focus on basic schooling:
- Increase cohesion/alignment weights (10.0, 5.0)
- Lower upstream pressure (0.1 instead of 0.5)
- Enable drafting to encourage formation learning

### Mid Training (Episodes 51-200)

Balance schooling + migration:
- Standard weights as documented
- Monitor mean schooling score (target: >1.0)
- Monitor drafting utilization (target: >0.10 mean)

### Fine Tuning (Episodes 201+)

Optimize for efficiency:
- Increase energy efficiency weight (3.0 instead of 2.0)
- Increase upstream weight (1.0 instead of 0.5)
- Tune threat_level to environment (0.3-0.5 for moderate predation)

---

## References

1. **Partridge, B.L., & Pitcher, T. (1980).** "The sensory basis of fish schools: Relative roles of lateral line and vision." *Journal of Comparative Physiology A*, 135: 315-325.

2. **Magurran, A.E., & Pitcher, T.J. (1987).** "Provenance, shoal size and the sociobiology of predator-evasion behaviour in minnow shoals." *Proceedings of the Royal Society B*, 229: 439-465.

3. **Weihs, D. (1973).** "Hydromechanics of fish schooling." *Nature*, 241: 290-291.

4. **Fish, F.E., & Lauder, G.V. (2006).** "Passive and active flow control by swimming fishes and mammals." *Annual Review of Fluid Mechanics*, 38: 193-224.

5. **Hemelrijk, C.K., et al. (2015).** "The increased efficiency of fish swimming in a school." *Fish and Fisheries*, 16(3): 511-521.

---

## Implementation Notes

### Computational Efficiency

- **KDTree queries:** O(N log N) for neighbor searches within sensory range
- **Per-agent metrics:** Computed once per timestep, cached in `metrics` dict
- **Vectorized operations:** NumPy broadcasting for energy/drafting calculations

### Debugging Tips

If schooling scores are unexpectedly low:
1. Check `threat_level` (too high → unrealistic tight cohesion)
2. Verify `body_lengths` units (should be meters, not mm)
3. Inspect `cohesion_radius_relaxed/threatened` (default 3.0/1.5 BL)
4. Monitor alignment scores separately (may be sacrificed for upstream progress)

If drafting benefits are zero:
1. Confirm `drafting_enabled = True` in BehavioralWeights
2. Check `drafting_distance` (default 2.0 BL may be too strict)
3. Increase `drafting_angle_tolerance` (default 30° may be too tight)
4. Verify agents are swimming fast enough to generate measurable drag

---

**Last Updated:** 2025-01-27  
**Author:** AI-assisted implementation with biological grounding from fisheries research
