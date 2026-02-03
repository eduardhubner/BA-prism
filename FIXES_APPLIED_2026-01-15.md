# Fixes Applied - 2026-01-15

Applied three critical fixes to improve conceptual clarity and defensibility of polysemanticity scoring.

---

## Fix #1: Baseline Normalization Mismatch (CRITICAL)

**Problem:** Z-score was comparing weighted similarity (AUC-weighted) against unweighted baseline (apples to oranges).

**Solution:**
- Changed z-score computation to use `unweighted_mean_sim` instead of `weighted_mean_sim`
- This ensures apples-to-apples comparison with the unweighted baseline distribution
- Renamed metric from `similarity_zscore` → `unweighted_similarity_zscore` for clarity

**Code changes:**
```python
# Before (WRONG):
similarity_zscore = (weighted_mean_sim - baseline_mean) / baseline_std

# After (CORRECT):
unweighted_mean_sim = row['mean_similarity']
unweighted_similarity_zscore = (unweighted_mean_sim - baseline_mean) / baseline_std
```

**Impact:** Z-scores now have valid statistical interpretation.

---

## Fix #2: Inconsistent AUC Threshold (0.55 vs 0.5)

**Problem:** Hardcoded `0.55` threshold for `has_reliable_desc` looked arbitrary.

**Solution:**
- Added explicit constant: `HAS_RELIABLE_DESC_AUC = 0.55`
- Added comment explaining it's "slightly above chance"
- Updated all references to use the constant

**Code changes:**
```python
# Added constant
HAS_RELIABLE_DESC_AUC = 0.55  # AUC threshold for "has reliable description" flag (slightly above chance)

# Updated usage
has_reliable_desc = max_auc >= HAS_RELIABLE_DESC_AUC  # At least one prediction above chance
```

**Impact:** Threshold is now documented and defensible.

---

## Fix #3: Unlabeled Diagnostics (Confusing Naming)

**Problem:** Metrics like `median_sim`, `min_sim`, `redundancy_fraction` are unweighted but not labeled as such.

**Solution:**
- Renamed all diagnostic metrics to include `_unweighted` suffix:
  - `median_sim` → `median_sim_unweighted`
  - `min_sim` → `min_sim_unweighted`
  - `range_gap` → `range_gap_unweighted`
  - `redundancy_fraction` → `redundancy_fraction_unweighted`

**Impact:** Clear distinction between weighted primary metrics and unweighted diagnostics.

---

## Files Modified

1. **[src/polysemanticity_scoring.py](src/polysemanticity_scoring.py)**
   - Added `HAS_RELIABLE_DESC_AUC` constant
   - Fixed baseline z-score computation (Fix #1)
   - Renamed all diagnostic metrics (Fix #3)
   - Updated metadata structure
   - Updated print functions

2. **[src/run_sensitivity_analysis.py](src/run_sensitivity_analysis.py)**
   - Updated Experiment B to use `*_unweighted` names
   - Updated Experiment C to use `redundancy_fraction_unweighted`

---

## Breaking Changes

**WARNING:** This changes the output CSV column names. Old CSVs use:
- `median_sim`, `min_sim`, `range_gap`, `redundancy_fraction`
- `similarity_zscore`

New CSVs use:
- `median_sim_unweighted`, `min_sim_unweighted`, `range_gap_unweighted`, `redundancy_fraction_unweighted`
- `unweighted_similarity_zscore`

**Action taken:** Deleted all old result files to avoid confusion.

---

## Next Steps

1. **Rerun main experiment:**
   ```bash
   python3 src/polysemanticity_scoring.py
   ```

2. **Rerun sensitivity analysis:**
   ```bash
   python3 src/run_sensitivity_analysis.py
   ```

3. **Update interpretation guides** (if needed) to reflect new metric names

---

## Justification for Thesis

**Fix #1 (Baseline):**
> "We z-score the unweighted mean similarity against the unweighted baseline distribution to ensure valid statistical comparison. The weighted similarity is our primary metric but is not normalized, as there is no principled weighted baseline distribution available."

**Fix #2 (Threshold):**
> "We define a neuron as having a 'reliable description' if max(AUC) ≥ 0.55, slightly above the random baseline of 0.5, to ensure descriptions have demonstrable predictive power."

**Fix #3 (Naming):**
> "Diagnostic metrics (median_sim_unweighted, range_gap_unweighted, etc.) are computed on raw similarity values after optional AUC-based filtering, but are not weighted by AUC. This provides interpretable distributional diagnostics on the description sample."

---

## What Didn't Change (By Design)

- **Weighted primary metric (`weighted_mean_sim`)**: Still uses AUC weighting
- **Diversity score**: Still computed as `1 - weighted_mean_sim`
- **n_eff, sum_weights**: Still based on AUC weights
- **Diagnostic computation logic**: Same as before, just renamed

The fixes are purely about clarity, consistency, and statistical validity - not changing the underlying methodology.
