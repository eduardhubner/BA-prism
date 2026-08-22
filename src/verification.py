"""
The two-stage verification procedure -- single source of truth.

This module owns the method described in the paper's Method section: candidate
descriptions are filtered by AUROC, then redundant survivors are collapsed, and
the resulting profile is summarised. It previously existed as four independent
implementations (case_study_analysis, filtered_evaluation,
weighting_gap_auroc_variance, verification_outcomes_table) which agreed on the
data but had to be changed in lockstep. They now all call in here.

Pipeline (per feature)
----------------------
  Stage 1  AUROC filter
      Drop descriptions with AUROC <= AUC_THRESHOLD (chance level). Running this
      BEFORE the collapse matters: a low-AUROC description must not be able to
      bridge two otherwise-distinct survivors in the similarity graph. On the
      239-neuron dataset both orders happen to give identical survivor sets, but
      the ordering is what the method's argument relies on.

  Stage 2  Redundancy collapse
      Among the survivors of stage 1, connect pairs with cosine similarity >
      SIM_THRESHOLD and take connected components. Each component is one concept;
      the highest-AUROC description represents it, the rest are "collapsed".
      Connected components (rather than greedy pairwise merging) make this
      deterministic -- no merge-order dependence, no seed.

Metric conventions
------------------
  sigma-tilde (weighted_mean_sim) is 1.0 at n_s == 1 by convention and undefined
  at n_s == 0. Restricting analyses to n_s >= 2 is a *conditioning* choice -- that
  is where a spread is informative -- not the domain of the metric.

  evidence_quality Q = 2*mean(survivor AUROC) - 1 is THE reported quality metric
  (paper, Method section). Defined for n_s >= 1, undefined at n_s = 0.

  confidence_C is LEGACY -- superseded by Q and not reported in the paper. It
  averages over ALL candidates, so it is mechanically diluted by adding candidates
  (0.515 -> 0.472 at k=10), which is precisely the defect Q avoids by averaging
  over survivors only. Still computed because the k=10 ablation's dilution result
  refers to it. Do not report it as a result.

Terminology: "AUROC" and "AUC" are the same metric. Input columns keep the legacy
`auc_` spelling (see RESULTS_INDEX.md).
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np

SIM_THRESHOLD = 0.70
AUC_THRESHOLD = 0.50
N_DESC = 5

# decision strings (used by the case-study MDS plots)
D_REMOVED = "removed (low AUC)"
D_COLLAPSED = "collapsed"
D_SURVIVED = "survived"


@dataclass(frozen=True)
class FilterResult:
    """Outcome of the two-stage procedure for one feature. Index-based."""

    filtered: tuple           # removed by the AUROC gate
    collapsed: tuple          # redundant; folded into a representative
    surviving: tuple          # representatives -- the characterisation
    decisions: tuple          # one string per description, positionally

    @property
    def reliable(self):
        """Descriptions that passed the AUROC gate (= collapsed + surviving)."""
        return tuple(sorted(self.collapsed + self.surviving))

    @property
    def eligible(self):
        """Descriptions eligible for redundancy collapse.

        A feature left with a single reliable description has nothing to collapse
        against, so that lone survivor is NOT eligible. This is the denominator of
        the paper's redundancy rate -- 468 descriptions, not the 503 that passed
        the filter and not all 1195.
        """
        rel = self.reliable
        return rel if len(rel) >= 2 else ()

    @property
    def n_surviving(self):
        return len(self.surviving)


def sim_matrix_from_row(row, n_desc=N_DESC, sep=""):
    """Build the symmetric similarity matrix from a dataset row.

    `sep=""` reads k=5 columns (`sim_12`); `sep="_"` reads k=10 columns (`sim_1_2`).
    """
    sim = np.eye(n_desc)
    for i, j in combinations(range(1, n_desc + 1), 2):
        v = row[f"sim_{i}{sep}{j}"]
        sim[i - 1, j - 1] = sim[j - 1, i - 1] = v
    return sim


def filter_and_collapse(aucs, sim_matrix, *, sim_threshold=SIM_THRESHOLD,
                        auc_threshold=AUC_THRESHOLD):
    """Run both stages. Returns a FilterResult. Deterministic."""
    aucs = np.asarray(aucs, dtype=float)
    n = len(aucs)
    indices = list(range(n))

    # Stage 1 -- AUROC filter, before the collapse.
    kept = [i for i in indices if aucs[i] > auc_threshold]

    # Stage 2 -- connected components among the kept descriptions only.
    adj = {i: set() for i in kept}
    for i, j in combinations(kept, 2):
        if sim_matrix[i, j] > sim_threshold:
            adj[i].add(j)
            adj[j].add(i)

    visited, components = set(), []
    for start in kept:
        if start not in visited:
            comp, queue = [], [start]
            while queue:
                node = queue.pop()
                if node not in visited:
                    visited.add(node)
                    comp.append(node)
                    queue.extend(adj[node] - visited)
            components.append(comp)

    # Highest-AUROC description represents its component; the rest collapse.
    representatives = {max(comp, key=lambda i: aucs[i]) for comp in components}

    decisions = tuple(
        D_REMOVED if aucs[i] <= auc_threshold
        else (D_SURVIVED if i in representatives else D_COLLAPSED)
        for i in indices
    )
    return FilterResult(
        filtered=tuple(i for i in indices if aucs[i] <= auc_threshold),
        collapsed=tuple(i for i in kept if i not in representatives),
        surviving=tuple(sorted(representatives)),
        decisions=decisions,
    )


def compute_metrics(aucs, sim_matrix, result):
    """Superset of every metric the callers need, for one feature.

    Callers select and rename the fields they write, so output schemas are
    unaffected by living here.
    """
    aucs = np.asarray(aucs, dtype=float)
    n = len(aucs)
    surviving = list(result.surviving)
    n_s = len(surviving)

    prism_score = float(np.mean([sim_matrix[i, j] for i, j in combinations(range(n), 2)]))
    confidence_C = float(np.mean([max(a - 0.5, 0) / 0.5 for a in aucs]))
    mean_auc_pre = float(np.mean(aucs))

    base = dict(prism_score=prism_score, confidence_C=confidence_C,
                mean_auc_pre=mean_auc_pre, n_surviving=n_s)

    if n_s == 0:
        return dict(base, mean_auc_post=np.nan, effective_n=np.nan,
                    mean_sim_post=np.nan, weighted_mean_sim=np.nan,
                    evidence_quality=np.nan, uncharacterized=True)

    surv_aucs = [aucs[i] for i in surviving]
    mean_auc_post = float(np.mean(surv_aucs))
    effective_n = float(np.sum([(a - 0.5) / 0.5 for a in surv_aucs]))
    evidence_quality = float(2 * mean_auc_post - 1)

    if n_s == 1:
        # convention: a single survivor is maximally self-consistent
        mean_sim_post = 1.0
        weighted_mean_sim = 1.0
    else:
        pairs = list(combinations(surviving, 2))
        mean_sim_post = float(np.mean([sim_matrix[i, j] for i, j in pairs]))
        num = den = 0.0
        for i, j in pairs:
            w = ((aucs[i] - 0.5) / 0.5) * ((aucs[j] - 0.5) / 0.5)
            num += w * sim_matrix[i, j]
            den += w
        weighted_mean_sim = float(num / den) if den > 0 else np.nan

    return dict(base, mean_auc_post=mean_auc_post, effective_n=effective_n,
                mean_sim_post=mean_sim_post, weighted_mean_sim=weighted_mean_sim,
                evidence_quality=evidence_quality, uncharacterized=False)


def profile_row(row, n_desc=N_DESC, sep="", **thresholds):
    """Convenience: row -> (FilterResult, metrics). Used by the table/profile scripts."""
    aucs = np.array([row[f"auc_{i}"] for i in range(1, n_desc + 1)], dtype=float)
    sim = sim_matrix_from_row(row, n_desc=n_desc, sep=sep)
    res = filter_and_collapse(aucs, sim, **thresholds)
    return res, compute_metrics(aucs, sim, res)


def collapse_then_filter(aucs, sim_matrix, *, sim_threshold=SIM_THRESHOLD,
                         auc_threshold=AUC_THRESHOLD):
    """NON-CANONICAL stage order: collapse over ALL descriptions, then AUROC-filter.

    Provided only to reproduce two analyses that were written this way
    (k10_ablation_analysis, human_annotation_correlation). Prefer
    `filter_and_collapse`, which is the order the paper's Method section
    describes and which the main pipeline uses.

    The two orders coincide at k=5 on this dataset (verified, 0/239 neurons
    differ) but NOT at k=10, where a low-AUROC description can bridge two
    otherwise-distinct survivors before it is filtered: they differ on 2/60
    neurons, yielding 2 fewer survivors (mean n_s 5.85 vs 5.88).
    """
    aucs = np.asarray(aucs, dtype=float)
    n = len(aucs)
    indices = list(range(n))

    adj = {i: set() for i in indices}
    for i, j in combinations(indices, 2):
        if sim_matrix[i, j] > sim_threshold:
            adj[i].add(j)
            adj[j].add(i)

    visited, components = set(), []
    for start in indices:
        if start not in visited:
            comp, queue = [], [start]
            while queue:
                node = queue.pop()
                if node not in visited:
                    visited.add(node)
                    comp.append(node)
                    queue.extend(adj[node] - visited)
            components.append(comp)

    reps = {max(comp, key=lambda i: aucs[i]) for comp in components}
    surviving = tuple(sorted(i for i in reps if aucs[i] > auc_threshold))
    surv_set = set(surviving)
    decisions = tuple(
        D_SURVIVED if i in surv_set
        else (D_REMOVED if i in reps else D_COLLAPSED)
        for i in indices
    )
    return FilterResult(
        filtered=tuple(i for i in reps if i not in surv_set),
        collapsed=tuple(i for i in indices if i not in reps),
        surviving=surviving,
        decisions=decisions,
    )


def collapse_components(indices, sim_matrix, aucs, sim_threshold=SIM_THRESHOLD):
    """Connected-components collapse over `indices`; returns the representative
    set (highest AUROC per component). No AUROC filtering is applied here.

    Shared building block. `filter_and_collapse` passes only AUROC survivors;
    the non-canonical collapse-first analyses pass all indices.
    """
    aucs = np.asarray(aucs, dtype=float)
    indices = list(indices)
    adj = {i: set() for i in indices}
    for i, j in combinations(indices, 2):
        if sim_matrix[i, j] > sim_threshold:
            adj[i].add(j)
            adj[j].add(i)
    visited, reps = set(), set()
    for start in indices:
        if start not in visited:
            comp, queue = [], [start]
            while queue:
                node = queue.pop()
                if node not in visited:
                    visited.add(node)
                    comp.append(node)
                    queue.extend(adj[node] - visited)
            reps.add(max(comp, key=lambda i: aucs[i]))
    return reps
