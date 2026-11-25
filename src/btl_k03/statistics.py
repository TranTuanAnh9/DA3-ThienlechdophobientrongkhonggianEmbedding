"""Paired session-cluster comparisons shared by experiment reports."""
import numpy as np


def clustered_delta(control, treatment, metric, group, iterations, rng):
    keys = ["source_row", "session_id", "target", "group"]
    aligned = control.merge(treatment, on=keys, suffixes=("_control", "_treatment"), validate="one_to_one")
    if len(aligned) != len(control) or len(aligned) != len(treatment):
        raise ValueError("Cannot compare different evaluation queries")
    if group != "all_warm":
        aligned = aligned[aligned.group.eq(group)]
    if aligned.empty:
        return {"delta": None, "ci_low": None, "ci_high": None, "n_queries": 0, "n_sessions": 0}
    aligned = aligned.assign(difference=aligned[f"{metric}_treatment"] - aligned[f"{metric}_control"])
    clusters = aligned.groupby("session_id").difference.agg(["sum", "count"])
    sums, counts = clusters["sum"].to_numpy(), clusters["count"].to_numpy()
    bootstrap = []
    for _ in range(iterations):
        ids = rng.integers(0, len(clusters), size=len(clusters))
        bootstrap.append(sums[ids].sum() / counts[ids].sum())
    return {"delta": float(sums.sum() / counts.sum()), "ci_low": float(np.quantile(bootstrap, .025)),
            "ci_high": float(np.quantile(bootstrap, .975)), "n_queries": len(aligned), "n_sessions": len(clusters)}
