"""
Measurements on labelled images: pure functions over per-image results, so they can be tested without any model.

A *result row* is what ``evaluation.runner`` records for one image (see ``RESULT_FIELDS`` there); a *label row* is a
line of ``labels.csv`` (``file, set, expectation, age, gender, race``). The expectation is ``human`` (one real human
face, with an age bin and a gender) or ``not_human`` (animals, birds, cartoons, paintings and sketches: nothing there
may be reported as a real human face).

FairFace labels age as a bin, never as a number. Wherever a number is derived from a bin, the assumption is named.
"""

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

Row = dict[str, Any]

# FairFace age bins as half-open ranges [low, high) in years; the last bin has no upper end.
AGE_BINS: dict[str, tuple[float, float | None]] = {
    "0-2": (0, 3), "3-9": (3, 10), "10-19": (10, 20), "20-29": (20, 30), "30-39": (30, 40),
    "40-49": (40, 50), "50-59": (50, 60), "60-69": (60, 70), "more than 70": (70, None),
}  # fmt: skip
# The age error is reported per band of ages. A band groups FairFace bins; the upper end is the age (exclusive) below
# which the band applies at run time. A band with fewer faces than MIN_FACES_PER_BAND borrows the pooled spread.
AGE_BANDS: tuple[tuple[tuple[str, ...], float], ...] = (
    (("0-2", "3-9"), 10.0),
    (("10-19",), 20.0),
    (("20-29",), 30.0),
    (("30-39",), 40.0),
    (("40-49",), 50.0),
    (("50-59",), 60.0),
    (("60-69",), 200.0),
)
MIN_FACES_PER_BAND = 50
ADULT_AGE = 18
MINOR_BINS = ("0-2", "3-9")  # entirely under 18; the 10-19 bin straddles 18 and cannot be scored either way
ADULT_BINS = ("30-39", "40-49", "50-59", "60-69", "more than 70")


def ratio(part: float, whole: float, digits: int = 4) -> float | None:
    return round(part / whole, digits) if whole else None


def main_face(row: Row) -> Row | None:
    """The face a single-portrait label refers to: the largest verified human face."""
    faces = row.get("humans") or []
    return max(faces, key=lambda face: face["area"]) if faces else None


def bin_residual(estimate: float, low: float, high: float | None) -> float:
    """Distance in years from the estimate to the labelled bin; zero inside it (a lower bound on the real error)."""
    if estimate < low:
        return low - estimate
    if high is not None and estimate >= high:
        return estimate - high
    return 0.0


def expected_error_in_bin(estimate: float, low: float, high: float) -> float:
    """
    Expected absolute error if the true age were uniform within the bin (an assumption: FairFace gives no exact age).
    """
    if low <= estimate < high:
        return ((estimate - low) ** 2 + (high - estimate) ** 2) / (2 * (high - low))
    return abs(estimate - (low + high) / 2)


def human_set(rows: Iterable[Row], labels: dict[str, Row]) -> list[tuple[Row, Row]]:
    return [(row, labels[row["file"]]) for row in rows if labels.get(row["file"], {}).get("expectation") == "human"]


def not_human_set(rows: Iterable[Row], labels: dict[str, Row]) -> list[tuple[Row, Row]]:
    return [(row, labels[row["file"]]) for row in rows if labels.get(row["file"], {}).get("expectation") == "not_human"]


# --------------------------------------------------------------------------------------
# Real human faces: detection, verification, gender, age
# --------------------------------------------------------------------------------------


def detection_report(pairs: list[tuple[Row, Row]]) -> dict[str, Any]:
    """How many portraits were reported as exactly one real human face, and what happened to the rest."""
    scored = [row for row, _ in pairs if not row.get("error")]
    counts = [len(row["humans"]) for row in scored]
    return {
        "images": len(pairs),
        "errors": len(pairs) - len(scored),
        # FairFace crops come from group photos: neighbours can be in frame, so "exactly one" is not a ground truth.
        "at_least_one_human_rate": ratio(sum(c >= 1 for c in counts), len(scored)),
        "exactly_one_human_rate": ratio(sum(c == 1 for c in counts), len(scored)),
        "no_human_rate": ratio(sum(c == 0 for c in counts), len(scored)),
        "several_humans_rate": ratio(sum(c > 1 for c in counts), len(scored)),
        "reported_a_non_human_face_rate": ratio(sum(row["others"] > 0 for row in scored), len(scored)),
        "status_counts": dict(sorted(_count(row["status"] for row in scored).items())),
    }


def _count(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[value] += 1
    return counts


def gender_report(pairs: list[tuple[Row, Row]]) -> dict[str, Any]:
    """Gender accuracy among the faces where a gender was given, the share left Unsure, and the same per race."""
    groups: dict[str, list[tuple[Row, Row]]] = defaultdict(list)
    for row, label in pairs:
        face = main_face(row)
        if face is not None:
            groups["ALL"].append((face, label))
            groups[label["race"]].append((face, label))

    def summarise(items: list[tuple[Row, Row]]) -> dict[str, Any]:
        answered = [(face, label) for face, label in items if face["gender"] in ("Male", "Female")]
        correct = sum(face["gender"] == label["gender"] for face, label in answered)
        return {
            "faces": len(items),
            "accuracy_when_answered": ratio(correct, len(answered)),
            "unsure_rate": ratio(len(items) - len(answered), len(items)),
        }

    return {group: summarise(items) for group, items in sorted(groups.items())}


def age_report(pairs: list[tuple[Row, Row]]) -> dict[str, Any]:
    """
    Age against the labelled bins: the share inside the bin, the error to the bin (a lower bound), the expected
    error if ages were uniform within the bin, and the bias and spread of the estimate around each bin's centre.
    """
    by_bin: dict[str, list[float]] = defaultdict(list)
    inside = answered = 0
    lower_bound_errors: list[float] = []
    expected_errors: list[float] = []
    unsure = total = 0
    for row, label in pairs:
        face = main_face(row)
        if face is None or label["age"] not in AGE_BINS:
            continue
        total += 1
        if face["age_status"] != "Estimated":
            unsure += 1
            continue
        low, high = AGE_BINS[label["age"]]
        estimate = face["estimated_age"]
        answered += 1
        residual = bin_residual(estimate, low, high)
        inside += residual == 0
        lower_bound_errors.append(residual)
        if high is not None:
            expected_errors.append(expected_error_in_bin(estimate, low, high))
            by_bin[label["age"]].append(estimate)

    return {
        "faces": total,
        "unsure_rate": ratio(unsure, total),
        "inside_labelled_bin_rate": ratio(inside, answered),
        "mean_error_to_bin_years_lower_bound": round(statistics.fmean(lower_bound_errors), 2) if answered else None,
        "mean_error_years_if_ages_uniform_in_bin": round(statistics.fmean(expected_errors), 2)
        if expected_errors
        else None,
        "per_bin": {name: _bin_spread(name, estimates) for name, estimates in by_bin.items()},
        "error_by_age_band": error_by_band(by_bin),
    }


def _bin_spread(name: str, estimates: list[float]) -> dict[str, Any]:
    """Bias of the estimates around the bin centre and their spread with the bin's own width taken out."""
    low, high = AGE_BINS[name]
    assert high is not None
    centre, width = (low + high) / 2, high - low
    differences = [estimate - centre for estimate in estimates]
    variance = statistics.pvariance(differences) if len(differences) > 1 else 0.0
    return {
        "faces": len(estimates),
        "bias_years": round(statistics.fmean(differences), 2),
        "model_spread_years": round(math.sqrt(max(0.0, variance - width**2 / 12)), 2),
    }


def _differences(by_bin: dict[str, list[float]], names: Iterable[str]) -> tuple[list[float], float]:
    """Estimate minus bin centre for the given bins, and the mean variance a uniform age inside each bin explains."""
    differences: list[float] = []
    width_variance = 0.0
    for name in names:
        low, high = AGE_BINS[name]
        assert high is not None
        estimates = by_bin.get(name, [])
        differences += [estimate - (low + high) / 2 for estimate in estimates]
        width_variance += len(estimates) * (high - low) ** 2 / 12
    return differences, (width_variance / len(differences) if differences else 0.0)


def error_by_band(by_bin: dict[str, list[float]]) -> list[dict[str, Any]]:
    """
    Bias and spread of the age estimate for each band of ages, with the width of the labelled bin taken out (this
    assumes ages are spread evenly inside a bin). A band with too few faces uses the spread of all faces pooled.
    """
    every_bin = [name for names, _ in AGE_BANDS for name in names]
    pooled, pooled_width = _differences(by_bin, every_bin)
    pooled_spread = math.sqrt(max(0.0, statistics.pvariance(pooled) - pooled_width)) if len(pooled) > 1 else 0.0
    table = []
    for names, upper in AGE_BANDS:
        differences, width = _differences(by_bin, names)
        enough = len(differences) >= MIN_FACES_PER_BAND
        spread = math.sqrt(max(0.0, statistics.pvariance(differences) - width)) if enough else pooled_spread
        table.append(
            {
                "below_age": upper,
                "faces": len(differences),
                "bias_years": round(statistics.fmean(differences), 2) if differences else None,
                "spread_years": round(spread, 2),
                "source": "band" if enough else "pooled",
            }
        )
    return table


def minor_report(pairs: list[tuple[Row, Row]], threshold: float) -> dict[str, Any]:
    """Recall on faces labelled under 18 and the false-alarm rate on faces labelled 30 or older."""
    minors = flagged_minors = adults = flagged_adults = teens = flagged_teens = 0
    for row, label in pairs:
        face = main_face(row)
        if face is None or face["p_under_18"] is None:
            continue
        flagged = face["p_under_18"] >= threshold
        if label["age"] in MINOR_BINS:
            minors, flagged_minors = minors + 1, flagged_minors + flagged
        elif label["age"] in ADULT_BINS:
            adults, flagged_adults = adults + 1, flagged_adults + flagged
        elif label["age"] == "10-19":
            teens, flagged_teens = teens + 1, flagged_teens + flagged
    return {
        "threshold": threshold,
        "minor_recall_labelled_0_to_9": ratio(flagged_minors, minors),
        "minors_scored": minors,
        "flag_rate_labelled_10_to_19": ratio(flagged_teens, teens),
        "false_flag_rate_labelled_30_plus": ratio(flagged_adults, adults),
        "adults_scored": adults,
    }


def minor_sweep(pairs: list[tuple[Row, Row]], thresholds: Iterable[float]) -> list[dict[str, Any]]:
    """The minor report at every threshold: how recall and the false-alarm rate trade off."""
    return [minor_report(pairs, threshold) for threshold in thresholds]


# --------------------------------------------------------------------------------------
# Things that must not be reported as a real human face
# --------------------------------------------------------------------------------------


def not_human_report(pairs: list[tuple[Row, Row]]) -> dict[str, Any]:
    """Per set, how often something that is not a real human was reported as one, as a non-human face, or not at all."""
    groups: dict[str, list[Row]] = defaultdict(list)
    for row, label in pairs:
        if not row.get("error"):
            groups[label["set"]].append(row)
            groups["ALL"].append(row)

    def summarise(rows: list[Row]) -> dict[str, Any]:
        false_human = sum(len(row["humans"]) > 0 for row in rows)
        non_human = sum(not row["humans"] and row["others"] > 0 for row in rows)
        unsure = sum(not row["humans"] and not row["others"] and row["unsure"] > 0 for row in rows)
        silent = len(rows) - false_human - non_human - unsure
        return {
            "images": len(rows),
            "reported_as_real_human_rate": ratio(false_human, len(rows)),
            "passed_the_whole_policy_rate": ratio(sum(row["status"] == "PASSED" for row in rows), len(rows)),
            "reported_as_non_human_rate": ratio(non_human, len(rows)),
            "left_unsure_rate": ratio(unsure, len(rows)),
            "nothing_reported_rate": ratio(silent, len(rows)),
        }

    return {group: summarise(rows) for group, rows in sorted(groups.items())}


# --------------------------------------------------------------------------------------
# The verification thresholds
# --------------------------------------------------------------------------------------


def accepted_as_human(candidates: list[Row], min_prob: float, min_margin: float) -> bool:
    """Would any candidate of this image be verified as a real human face at these thresholds?"""
    return any(c["best_is_human"] and c["p_human"] >= min_prob and c["margin"] >= min_margin for c in candidates)


def threshold_sweep(
    pairs_human: list[tuple[Row, Row]],
    pairs_not_human: list[tuple[Row, Row]],
    probabilities: Iterable[float],
    margins: Iterable[float],
) -> list[dict[str, Any]]:
    """Image-level recall on portraits and false-human rate on non-human images for every threshold pair."""
    positives = [row["candidates"] for row, _ in pairs_human if not row.get("error")]
    negatives = [row["candidates"] for row, _ in pairs_not_human if not row.get("error")]
    table = []
    for min_prob in probabilities:
        for min_margin in margins:
            hits = sum(accepted_as_human(c, min_prob, min_margin) for c in positives)
            false = sum(accepted_as_human(c, min_prob, min_margin) for c in negatives)
            table.append(
                {
                    "human_min_prob": round(min_prob, 3),
                    "human_min_margin": round(min_margin, 3),
                    "portraits_verified_rate": ratio(hits, len(positives)),
                    "non_human_images_verified_as_human_rate": ratio(false, len(negatives)),
                }
            )
    return table


def best_threshold(table: list[dict[str, Any]], max_false_rate: float) -> dict[str, Any] | None:
    """The pair with the highest portrait recall whose false-human rate stays within ``max_false_rate``."""
    allowed = [
        entry
        for entry in table
        if entry["non_human_images_verified_as_human_rate"] is not None
        and entry["non_human_images_verified_as_human_rate"] <= max_false_rate
        and entry["portraits_verified_rate"] is not None
    ]
    return max(allowed, key=lambda e: (e["portraits_verified_rate"], -e["human_min_prob"]), default=None)


# --------------------------------------------------------------------------------------
# The whole report and the gates
# --------------------------------------------------------------------------------------


def build_report(rows: list[Row], labels: dict[str, Row], minor_threshold: float) -> dict[str, Any]:
    humans, not_humans = human_set(rows, labels), not_human_set(rows, labels)
    return {
        "portraits": {
            "detection": detection_report(humans),
            "gender": gender_report(humans),
            "age": age_report(humans),
            "minor": minor_report(humans, minor_threshold),
        },
        "not_human": not_human_report(not_humans),
        "speed_ms": _speed(rows),
    }


def _speed(rows: list[Row]) -> dict[str, float | None]:
    times = sorted(row["ms"] for row in rows if not row.get("error"))
    if not times:
        return {"p50": None, "p90": None}
    return {"p50": times[len(times) // 2], "p90": times[min(len(times) - 1, int(len(times) * 0.9))]}


def lookup(report: dict[str, Any], path: str) -> Any:
    """``a.b.c`` -> report["a"]["b"]["c"]."""
    value: Any = report
    for part in path.split("."):
        value = value[part]
    return value


def check_gates(report: dict[str, Any], gates: dict[str, dict[str, float]]) -> list[str]:
    """
    Compare measured values with ``{"path": {"min": x}}`` / ``{"path": {"max": y}}`` limits; returns every violation.
    """
    problems = []
    for path, limits in gates.items():
        try:
            value = lookup(report, path)
        except KeyError, TypeError:
            problems.append(f"{path}: not measured")
            continue
        if value is None:
            problems.append(f"{path}: not measured")
        elif "min" in limits and value < limits["min"]:
            problems.append(f"{path}: {value} is below the minimum {limits['min']}")
        elif "max" in limits and value > limits["max"]:
            problems.append(f"{path}: {value} is above the maximum {limits['max']}")
    return problems
