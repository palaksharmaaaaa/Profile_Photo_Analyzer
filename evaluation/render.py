"""
Turn an evaluation report into the Markdown tables of ``docs/EVALUATION_AND_FAIRNESS.md``.

The tables are generated, never typed: ``scripts/render_evaluation.py`` replaces the text between two markers in the
document with ``render(report)``, so every number in it comes from a run of ``scripts/evaluate.py``.
"""

from typing import Any

START_MARKER = "<!-- measured:start -->"
END_MARKER = "<!-- measured:end -->"
BENCHMARK_START_MARKER = "<!-- benchmark:start -->"
BENCHMARK_END_MARKER = "<!-- benchmark:end -->"
SWEEP_PROBABILITIES = (0.5, 0.6, 0.7, 0.8, 0.9)


def percent(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value * 100:.{digits}f} %"


def number(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def table(headers: list[str], rows: list[list[str]], right_from: int = 1) -> str:
    """A GitHub-flavoured Markdown table whose columns from ``right_from`` on are right aligned."""
    alignment = ["---:" if index >= right_from else ":---" for index in range(len(headers))]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(alignment) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]
    return "\n".join(lines)


def run_section(report: dict[str, Any]) -> str:
    run = report["run"]
    counts = ", ".join(f"{name} {count}" for name, count in sorted(run["images_per_set"].items()))
    settings = run["settings"]
    return "\n".join(
        [
            f"Run of {run['date']}. Images: {counts}.",
            f"Settings: verification threshold {settings['human_min_prob']} with margin {settings['human_min_margin']}, "
            f"minor screening threshold {settings['minor_risk_probability']}, SCRFD "
            f"{'on' if settings['enable_scrfd'] else 'off'}.",
        ]
    )


def detection_section(report: dict[str, Any]) -> str:
    d = report["portraits"]["detection"]
    return table(
        ["Outcome on real portraits", "Share"],
        [
            ["At least one real human face found", percent(d["at_least_one_human_rate"])],
            ["Exactly one real human face found", percent(d["exactly_one_human_rate"])],
            ["More than one real human face found", percent(d["several_humans_rate"])],
            ["No real human face found", percent(d["no_human_rate"])],
            ["A non-human face reported as well", percent(d["reported_a_non_human_face_rate"])],
            ["Analysis errors", f"{d['errors']} of {d['images']}"],
        ],
    )


def gender_section(report: dict[str, Any]) -> str:
    rows = [
        [
            group,
            str(entry["faces"]),
            percent(entry["accuracy_when_answered"]),
            percent(entry["unsure_rate"]),
            percent(entry["unsure_rate_labelled_0_to_9"]),
            percent(entry["unsure_rate_labelled_20_plus"]),
            percent(entry["unsure_without_a_score_rate"]),
        ]
        for group, entry in report["portraits"]["gender"].items()
    ]
    headers = [
        "Group",
        "Faces",
        "Accuracy when answered",
        "Left Unsure",
        "Unsure, labelled 0-9",
        "Unsure, labelled 20+",
        "Unsure with no score at all",
    ]
    return table(headers, rows)


def age_section(report: dict[str, Any]) -> str:
    age = report["portraits"]["age"]
    summary = table(
        ["Age measure", "Value"],
        [
            ["Faces with an age bin", str(age["faces"])],
            ["Age left Unsure", percent(age["unsure_rate"])],
            ["Estimate inside the labelled bin", percent(age["inside_labelled_bin_rate"])],
            [
                "Mean distance to the bin, years (a lower bound on the error)",
                number(age["mean_error_to_bin_years_lower_bound"]),
            ],
            [
                "Mean error, years, if ages were spread evenly inside each bin",
                number(age["mean_error_years_if_ages_uniform_in_bin"]),
            ],
        ],
    )
    bands = table(
        ["Estimated age below", "Faces", "Bias (years)", "Error spread (years)", "Spread taken from"],
        [
            [
                number(entry["below_age"], 0) if entry["below_age"] < 100 else "any",
                str(entry["faces"]),
                number(entry["bias_years"]),
                number(entry["spread_years"]),
                entry["source"],
            ]
            for entry in age["error_by_age_band"]
        ],
    )
    return summary + "\n\n" + bands


def age_by_race_section(report: dict[str, Any]) -> str:
    rows = [
        [
            group,
            str(entry["faces"]),
            percent(entry["unsure_rate"]),
            percent(entry["inside_labelled_bin_rate"]),
            number(entry["mean_error_to_bin_years_lower_bound"]),
            percent(entry["minor_recall_labelled_0_to_9"]),
            percent(entry["false_flag_rate_labelled_30_plus"]),
        ]
        for group, entry in report["portraits"]["age_by_race"].items()
    ]
    headers = [
        "Group", "Faces", "Age Unsure", "Inside bin", "Distance to bin (years)", "Labelled 0-9 flagged", "Labelled 30+ flagged",
    ]  # fmt: skip
    return table(headers, rows)


def minor_section(report: dict[str, Any]) -> str:
    rows = [
        [
            number(entry["threshold"]),
            percent(entry["minor_recall_labelled_0_to_9"]),
            percent(entry["flag_rate_labelled_10_to_19"]),
            percent(entry["false_flag_rate_labelled_30_plus"]),
        ]
        for entry in report["minor_thresholds"]
    ]
    first = report["minor_thresholds"][0]
    caption = f"Faces scored: {first['minors_scored']} labelled 0-9, {first['adults_scored']} labelled 30 or older."
    headers = ["`minor_risk_probability`", "Labelled 0-9 flagged", "Labelled 10-19 flagged", "Labelled 30+ flagged"]
    return table(headers, rows) + "\n\n" + caption


def not_human_section(report: dict[str, Any]) -> str:
    rows = [
        [
            group,
            str(entry["images"]),
            percent(entry["reported_as_real_human_rate"]),
            percent(entry["passed_the_whole_policy_rate"]),
            percent(entry["reported_as_non_human_rate"]),
            percent(entry["left_unsure_rate"]),
            percent(entry["nothing_reported_rate"]),
        ]
        for group, entry in report["not_human"].items()
    ]
    headers = [
        "Set",
        "Images",
        "A face reported as real human",
        "Passed the whole policy",
        "Non-human face",
        "Unsure",
        "Nothing",
    ]
    return table(headers, rows)


def thresholds_section(report: dict[str, Any]) -> str:
    chosen = report["run"]["settings"]
    rows = []
    for entry in report["thresholds"]:
        if entry["human_min_margin"] == chosen["human_min_margin"] and entry["human_min_prob"] in SWEEP_PROBABILITIES:
            mark = " (used)" if entry["human_min_prob"] == chosen["human_min_prob"] else ""
            rows.append(
                [
                    f"{entry['human_min_prob']}{mark}",
                    percent(entry["portraits_verified_rate"]),
                    percent(entry["non_human_images_verified_as_human_rate"]),
                ]
            )
    return table(["`human_min_prob`", "Portraits verified", "Non-human images verified as human"], rows)


def speed_section(report: dict[str, Any]) -> str:
    speed = report["speed_ms"]
    return f"Whole analysis, median {number(speed['p50'], 0)} ms and 90th percentile {number(speed['p90'], 0)} ms."


def render(report: dict[str, Any]) -> str:
    """The generated part of the evaluation document."""
    parts = [
        ("", run_section(report)),
        ("#### Real portraits: detection and verification", detection_section(report)),
        ("#### Gender", gender_section(report)),
        ("#### Age", age_section(report)),
        ("#### Age by race group", age_by_race_section(report)),
        ("#### Possible-minor screening", minor_section(report)),
        ("#### Images that must not be reported as a real human face", not_human_section(report)),
        ("#### The verification threshold", thresholds_section(report)),
        ("#### Speed during this run (3 processes sharing the machine)", speed_section(report)),
    ]
    return "\n\n".join(f"{title}\n\n{body}" if title else body for title, body in parts)


def render_benchmark(parallel: dict[str, Any], sequential: dict[str, Any]) -> str:
    """Latency per stage, with parallel inference on and off (output of ``scripts/benchmark_latency.py``)."""
    environment = parallel["environment"]
    header = (
        f"Python {environment['python']}, onnxruntime {environment['onnxruntime']}, "
        f"{environment['logical_cpus']} logical CPUs, {environment['onnx_threads_per_model']} ONNX threads per model, "
        f"image {environment['analysed_resolution'][0]} x {environment['analysed_resolution'][1]} px, "
        f"{environment['iterations']} iterations."
    )
    rows = [
        [
            stage,
            number(sequential["latency_ms"][stage]["p50"], 0),
            number(sequential["latency_ms"][stage]["p90"], 0),
            number(entry["p50"], 0),
            number(entry["p90"], 0),
        ]
        for stage, entry in parallel["latency_ms"].items()
    ]
    headers = ["Stage (ms)", "One at a time p50", "p90", "Parallel p50", "p90"]
    return header + "\n\n" + table(headers, rows)


def replace_between(document: str, start_marker: str, end_marker: str, generated: str) -> str:
    """The document with the text between the two markers replaced by ``generated``."""
    start = document.index(start_marker) + len(start_marker)
    end = document.index(end_marker)
    return f"{document[:start]}\n\n{generated}\n\n{document[end:]}"


def replace_between_markers(document: str, generated: str) -> str:
    """The measured-results part of the document."""
    return replace_between(document, START_MARKER, END_MARKER, generated)
