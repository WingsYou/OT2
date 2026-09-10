#!/usr/bin/env python3
"""Diagnose overlapping sleep-stage intervals in wearable 5-minute artifacts.

This is a read-only cluster audit.  It uses the source paths embedded in the
canonical wearable artifacts, inspects only participants whose saved sleep
audit reports more than one 5-minute stage total per slot, and distinguishes:

* exact duplicate parsed intervals;
* distinct raw intervals that overlap;
* records containing both a direct stage and a stage summary; and
* unusually long intervals or suspicious summary-duration units.

It supports both the legacy V4 representation (used to diagnose the source
problem) and the resolved V5 representation (where any such slot is a failed
invariant).  No raw file or artifact is modified.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

try:
    import build_wearable_5min_sequences as wearable
except ImportError:  # Support ``python -m ot2.audit_wearable_sleep_overlap``.
    from ot2 import build_wearable_5min_sequences as wearable  # type: ignore


SUPPORTED_BUILDER_VERSIONS = {"wearable-5min-4", "wearable-5min-5"}

OVERLAP_ROW_COLUMNS = [
    "person_id",
    "builder_version",
    "source_path",
    "raw_sleep_node_type",
    "n_raw_records_used_by_builder",
    "n_parsed_events",
    "n_valid_events",
    "n_direct_events",
    "n_summary_events",
    "n_records_direct_and_summary",
    "n_invalid_events",
    "n_exact_duplicate_groups",
    "n_exact_duplicate_extra_events",
    "n_events_overlapping_previous",
    "n_intervals_over_24h",
    "duration_minutes_p50",
    "duration_minutes_p99",
    "duration_minutes_max",
    "saved_overlap_slots",
    "saved_max_stage_minutes_in_slot",
    "recomputed_max_stage_minutes_in_slot",
    "max_slot_index",
    "max_slot_start_utc",
    "max_slot_contributor_events",
    "max_slot_unique_stage_intervals",
    "max_slot_exact_duplicate_extra_events",
]

RESOLUTION_ROW_COLUMNS = [
    "person_id",
    "builder_version",
    "n_raw_sleep_records",
    "n_candidate_events",
    "n_invalid_candidate_events",
    "n_exact_duplicate_events_removed",
    "same_stage_overlap_minutes_all_source",
    "cross_stage_conflict_minutes_all_source",
    "cross_stage_conflict_minutes_in_window",
    "n_conflict_slots_in_window",
    "n_slots_with_stage_total_exceedance",
    "max_total_stage_minutes_in_slot",
]


def _torch_load(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.6
        return torch.load(path, map_location="cpu")


def _counter_rows(counter: Counter[Any], limit: Optional[int] = None) -> List[Dict[str, Any]]:
    items = counter.most_common(limit)
    return [{"value": str(value), "count": int(count)} for value, count in items]


def _quantile(values: Sequence[float], probability: float) -> Optional[float]:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def _stage_value(record: Mapping[str, Any]) -> Any:
    return (
        record.get("sleep_stage_state")
        or record.get("sleep_stage")
        or record.get("stage")
    )


def _summary_value(record: Mapping[str, Any]) -> Any:
    return record.get("sleep_stage_summary") or record.get("stage_summary")


def _builder_raw_records(obj: Mapping[str, Any]) -> Tuple[str, List[Mapping[str, Any]]]:
    candidates, audit = wearable._sleep_raw_records(dict(obj))
    return str(audit["sleep_node_type"]), [
        item for item in candidates if isinstance(item, Mapping)
    ]


def _summary_duration_descriptor(value: Any) -> str:
    if isinstance(value, Mapping):
        unit = str(value.get("unit", "<missing-unit>"))
        raw = value.get("value")
        return f"mapping:{unit}:{type(raw).__name__}"
    return f"scalar:<implicit-minutes>:{type(value).__name__}"


def _origin_events(
    records: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Reproduce the current parser while retaining raw-record provenance."""
    events: List[Dict[str, Any]] = []
    mode_counts: Counter[str] = Counter()
    duration_descriptors: Counter[str] = Counter()
    record_key_signatures: Counter[str] = Counter()
    summary_item_key_signatures: Counter[str] = Counter()

    for record_index, record in enumerate(records):
        record_key_signatures[",".join(sorted(str(key) for key in record))] += 1
        direct_stage = _stage_value(record)
        summary = _summary_value(record)
        has_summary = isinstance(summary, list)

        if direct_stage is not None and has_summary:
            mode_counts["direct_and_summary"] += 1
        elif direct_stage is not None:
            mode_counts["direct_only"] += 1
        elif has_summary:
            mode_counts["summary_only"] += 1
        else:
            mode_counts["neither"] += 1

        if direct_stage is not None:
            start, end = wearable._sleep_interval(record)
            events.append(
                {
                    "start": start,
                    "end": end,
                    "stage": wearable._normalize_sleep_stage(direct_stage),
                    "origin": "direct",
                    "record_index": int(record_index),
                    "summary_item_index": None,
                }
            )

        if not has_summary:
            continue

        outer_start, _ = wearable._sleep_interval(record)
        cursor = outer_start
        for item_index, item in enumerate(summary):
            if not isinstance(item, Mapping):
                mode_counts["non_mapping_summary_item"] += 1
                continue
            summary_item_key_signatures[
                ",".join(sorted(str(key) for key in item))
            ] += 1
            nested_stage = _stage_value(item)
            raw_duration = item.get("duration")
            duration_descriptors[_summary_duration_descriptor(raw_duration)] += 1
            duration = wearable._duration_minutes(raw_duration)
            if pd.isna(cursor) or nested_stage is None or duration is None:
                mode_counts["unparsed_summary_item"] += 1
                continue
            end = cursor + pd.Timedelta(minutes=duration)
            events.append(
                {
                    "start": cursor,
                    "end": end,
                    "stage": wearable._normalize_sleep_stage(nested_stage),
                    "origin": "summary",
                    "record_index": int(record_index),
                    "summary_item_index": int(item_index),
                }
            )
            cursor = end

    audit = {
        "record_modes": dict(mode_counts),
        "summary_duration_descriptors": _counter_rows(duration_descriptors),
        "record_key_signatures": _counter_rows(record_key_signatures, 20),
        "summary_item_key_signatures": _counter_rows(
            summary_item_key_signatures, 20
        ),
    }
    return events, audit


def _valid_event(event: Mapping[str, Any]) -> bool:
    start = event["start"]
    end = event["end"]
    return bool(pd.notna(start) and pd.notna(end) and end > start)


def _event_key(event: Mapping[str, Any]) -> Tuple[str, str, str]:
    return (
        pd.Timestamp(event["start"]).isoformat(),
        pd.Timestamp(event["end"]).isoformat(),
        str(event["stage"]),
    )


def _count_distinct_overlaps(events: Sequence[Mapping[str, Any]]) -> int:
    intervals = sorted(
        (
            pd.Timestamp(event["start"]),
            pd.Timestamp(event["end"]),
        )
        for event in events
        if _valid_event(event)
    )
    maximum_end = pd.NaT
    overlaps = 0
    for start, end in intervals:
        if pd.notna(maximum_end) and start < maximum_end:
            overlaps += 1
        if pd.isna(maximum_end) or end > maximum_end:
            maximum_end = end
    return int(overlaps)


def _max_slot_contributors(
    events: Sequence[Mapping[str, Any]],
    slot_start: pd.Timestamp,
    grid_minutes: int,
) -> List[Dict[str, Any]]:
    slot_end = slot_start + pd.Timedelta(minutes=grid_minutes)
    contributors: List[Dict[str, Any]] = []
    for event in events:
        if not _valid_event(event):
            continue
        start = pd.Timestamp(event["start"])
        end = pd.Timestamp(event["end"])
        overlap = max(
            0.0,
            float((min(end, slot_end) - max(start, slot_start)).total_seconds() / 60.0),
        )
        if overlap <= 0:
            continue
        contributors.append(
            {
                "start": start.isoformat(),
                "end": end.isoformat(),
                "stage": str(event["stage"]),
                "origin": str(event["origin"]),
                "record_index": int(event["record_index"]),
                "summary_item_index": event["summary_item_index"],
                "overlap_minutes": float(overlap),
            }
        )
    return contributors


def _audit_one(index_row: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    person_id = str(index_row["person_id"])
    artifact_path = Path(str(index_row["out_path"]))
    artifact = _torch_load(artifact_path)
    config = artifact.get("preprocessing_config", {})
    builder_version = str(config.get("builder_version", ""))
    if builder_version not in SUPPORTED_BUILDER_VERSIONS:
        raise ValueError(
            f"{person_id}: expected one of {sorted(SUPPORTED_BUILDER_VERSIONS)}, "
            f"found {builder_version!r}"
        )

    saved_sleep_audit = artifact["audit"]["activity"]["sleep"]
    overlap_slots = int(saved_sleep_audit["n_slots_with_overlapping_stages"])
    source_path_text = artifact.get("source_paths", {}).get("sleep")
    if not source_path_text:
        raise ValueError(f"{person_id}: overlap is reported but no sleep source exists")
    source_path = Path(str(source_path_text))
    with source_path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)
    if not isinstance(obj, Mapping):
        raise ValueError(f"{person_id}: sleep JSON top level is not an object")

    raw_node_type, records = _builder_raw_records(obj)
    events, origin_audit = _origin_events(records)
    saved_raw_count = saved_sleep_audit.get("n_raw_events")
    if (
        builder_version == "wearable-5min-4"
        and saved_raw_count is not None
        and len(events) != int(saved_raw_count)
    ):
        raise ValueError(
            f"{person_id}: diagnostic parser produced {len(events)} events but "
            f"the saved V4 audit reports {saved_raw_count}"
        )

    valid_events = [event for event in events if _valid_event(event)]
    durations = [
        float(
            (pd.Timestamp(event["end"]) - pd.Timestamp(event["start"])).total_seconds()
            / 60.0
        )
        for event in valid_events
    ]
    exact = Counter(_event_key(event) for event in valid_events)
    exact_duplicate_extra = int(sum(count - 1 for count in exact.values() if count > 1))
    exact_duplicate_groups = int(sum(count > 1 for count in exact.values()))

    activity = artifact["activity"].detach().cpu().numpy()
    feature_names = list(artifact["activity_feature_names"])
    stage_columns = [
        feature_names.index(f"activity__sleep__{stage}_minutes")
        for stage in artifact["sleep_stage_names"]
    ]
    stage_total = activity[:, stage_columns].astype(np.float64).sum(axis=1)
    max_slot_index = int(np.argmax(stage_total))
    max_slot_minutes = float(stage_total[max_slot_index])
    grid_minutes = int(artifact["time_grid"]["grid_minutes"])
    anchor_utc = pd.Timestamp(artifact["anchor"]["anchor_utc"])
    slot_start = anchor_utc + pd.Timedelta(minutes=max_slot_index * grid_minutes)
    contributors = _max_slot_contributors(events, slot_start, grid_minutes)
    contributor_keys = Counter(
        (item["start"], item["end"], item["stage"]) for item in contributors
    )

    direct_events = sum(event["origin"] == "direct" for event in events)
    summary_events = sum(event["origin"] == "summary" for event in events)
    record_modes = origin_audit["record_modes"]
    row = {
        "person_id": person_id,
        "builder_version": builder_version,
        "source_path": str(source_path),
        "raw_sleep_node_type": raw_node_type,
        "n_raw_records_used_by_builder": int(len(records)),
        "n_parsed_events": int(len(events)),
        "n_valid_events": int(len(valid_events)),
        "n_direct_events": int(direct_events),
        "n_summary_events": int(summary_events),
        "n_records_direct_and_summary": int(record_modes.get("direct_and_summary", 0)),
        "n_invalid_events": int(len(events) - len(valid_events)),
        "n_exact_duplicate_groups": exact_duplicate_groups,
        "n_exact_duplicate_extra_events": exact_duplicate_extra,
        "n_events_overlapping_previous": _count_distinct_overlaps(valid_events),
        "n_intervals_over_24h": int(sum(value > 24 * 60 for value in durations)),
        "duration_minutes_p50": _quantile(durations, 0.50),
        "duration_minutes_p99": _quantile(durations, 0.99),
        "duration_minutes_max": max(durations) if durations else None,
        "saved_overlap_slots": overlap_slots,
        "saved_max_stage_minutes_in_slot": float(
            saved_sleep_audit.get(
                "max_total_stage_minutes_in_slot",
                saved_sleep_audit.get("max_stage_minutes_in_slot", 0.0),
            )
        ),
        "recomputed_max_stage_minutes_in_slot": max_slot_minutes,
        "max_slot_index": max_slot_index,
        "max_slot_start_utc": slot_start.isoformat(),
        "max_slot_contributor_events": int(len(contributors)),
        "max_slot_unique_stage_intervals": int(len(contributor_keys)),
        "max_slot_exact_duplicate_extra_events": int(
            sum(count - 1 for count in contributor_keys.values() if count > 1)
        ),
    }
    if not math.isclose(
        row["saved_max_stage_minutes_in_slot"],
        max_slot_minutes,
        rel_tol=1e-6,
        abs_tol=1e-4,
    ):
        raise ValueError(f"{person_id}: saved and recomputed maximum disagree")

    detail = {
        **row,
        **origin_audit,
        "top_exact_duplicate_intervals": [
            {
                "start": key[0],
                "end": key[1],
                "stage": key[2],
                "count": int(count),
            }
            for key, count in exact.most_common(30)
            if count > 1
        ],
        "max_slot_contributors": contributors[:500],
        "max_slot_contributors_truncated": len(contributors) > 500,
    }
    return row, detail


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only diagnosis of overlapping wearable sleep stages"
    )
    parser.add_argument("--wearable-dir", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    wearable_dir = Path(args.wearable_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else wearable_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = wearable_dir / "wearable_5min_index.csv"
    index = pd.read_csv(index_path, dtype={"person_id": "string"})

    candidates: List[Dict[str, Any]] = []
    resolution_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    builder_versions_seen: Counter[str] = Counter()
    n_people_with_overlap_detected = 0
    for index_row in index.to_dict("records"):
        person_id = str(index_row["person_id"])
        try:
            artifact = _torch_load(Path(str(index_row["out_path"])))
            audit = artifact["audit"]["activity"]["sleep"]
            builder_version = str(
                artifact.get("preprocessing_config", {}).get("builder_version", "")
            )
            builder_versions_seen[builder_version or "<missing>"] += 1
            if builder_version not in SUPPORTED_BUILDER_VERSIONS:
                raise ValueError(
                    f"unsupported builder_version {builder_version!r}; expected one "
                    f"of {sorted(SUPPORTED_BUILDER_VERSIONS)}"
                )
            activity = artifact["activity"].detach().cpu().numpy()
            feature_names = list(artifact["activity_feature_names"])
            stage_columns = [
                feature_names.index(f"activity__sleep__{stage}_minutes")
                for stage in artifact["sleep_stage_names"]
            ]
            stage_total = activity[:, stage_columns].astype(np.float64).sum(axis=1)
            grid_minutes = float(artifact["time_grid"]["grid_minutes"])
            actual_overlap_slots = int((stage_total > grid_minutes + 1e-5).sum())
            actual_max_total = float(stage_total.max()) if stage_total.size else 0.0
            saved_overlap_slots = int(
                audit.get("n_slots_with_overlapping_stages", 0)
            )
            saved_max_total = float(
                audit.get(
                    "max_total_stage_minutes_in_slot",
                    audit.get("max_stage_minutes_in_slot", 0.0),
                )
            )
            if actual_overlap_slots != saved_overlap_slots:
                raise ValueError(
                    f"saved overlap count {saved_overlap_slots} disagrees with "
                    f"tensor count {actual_overlap_slots}"
                )
            if not math.isclose(
                actual_max_total, saved_max_total, rel_tol=1e-6, abs_tol=1e-4
            ):
                raise ValueError(
                    f"saved maximum {saved_max_total} disagrees with tensor "
                    f"maximum {actual_max_total}"
                )
            if builder_version == "wearable-5min-5":
                raw_conflict_mask = artifact.get("sleep_conflict_mask")
                if raw_conflict_mask is None:
                    raise ValueError("V5 artifact is missing sleep_conflict_mask")
                conflict_mask = raw_conflict_mask.detach().cpu().numpy().astype(bool)
                if conflict_mask.shape != stage_total.shape:
                    raise ValueError(
                        f"sleep_conflict_mask shape {conflict_mask.shape} does not "
                        f"match slot shape {stage_total.shape}"
                    )
                if int(conflict_mask.sum()) != int(audit.get("n_conflict_slots", 0)):
                    raise ValueError(
                        "sleep_conflict_mask count disagrees with n_conflict_slots"
                    )
                unknown_column = feature_names.index(
                    "activity__sleep__unknown_minutes"
                )
                if np.any(conflict_mask & (activity[:, unknown_column] <= 0)):
                    raise ValueError(
                        "sleep conflict slot has no positive unknown-stage minutes"
                    )
            resolution_row: Dict[str, Any] = {
                "person_id": person_id,
                "builder_version": builder_version,
                "n_raw_sleep_records": np.nan,
                "n_candidate_events": np.nan,
                "n_invalid_candidate_events": np.nan,
                "n_exact_duplicate_events_removed": np.nan,
                "same_stage_overlap_minutes_all_source": np.nan,
                "cross_stage_conflict_minutes_all_source": np.nan,
                "cross_stage_conflict_minutes_in_window": np.nan,
                "n_conflict_slots_in_window": np.nan,
                "n_slots_with_stage_total_exceedance": actual_overlap_slots,
                "max_total_stage_minutes_in_slot": actual_max_total,
            }
            if builder_version == "wearable-5min-5":
                resolution_row.update(
                    {
                        "n_raw_sleep_records": int(
                            audit.get("n_raw_sleep_records", 0)
                        ),
                        "n_candidate_events": int(audit.get("n_candidate_events", 0)),
                        "n_invalid_candidate_events": int(
                            audit.get("n_invalid_candidate_events", 0)
                        ),
                        "n_exact_duplicate_events_removed": int(
                            audit.get("n_exact_duplicate_events_removed", 0)
                        ),
                        "same_stage_overlap_minutes_all_source": float(
                            audit.get("same_stage_overlap_minutes", 0.0)
                        ),
                        "cross_stage_conflict_minutes_all_source": float(
                            audit.get("cross_stage_conflict_minutes", 0.0)
                        ),
                        "cross_stage_conflict_minutes_in_window": float(
                            audit.get("conflict_minutes_in_window", 0.0)
                        ),
                        "n_conflict_slots_in_window": int(
                            audit.get("n_conflict_slots", 0)
                        ),
                    }
                )
            resolution_rows.append(resolution_row)
            if actual_overlap_slots <= 0:
                continue
            n_people_with_overlap_detected += 1
            row, detail = _audit_one(index_row)
            candidates.append({"row": row, "detail": detail})
        except Exception as exc:
            errors.append({"person_id": person_id, "error": repr(exc)})

    rows = pd.DataFrame(
        [item["row"] for item in candidates], columns=OVERLAP_ROW_COLUMNS
    )
    if not rows.empty:
        rows = rows.sort_values(
            ["saved_max_stage_minutes_in_slot", "saved_overlap_slots"],
            ascending=False,
        )
    csv_path = out_dir / "wearable_sleep_overlap_audit.csv"
    json_path = out_dir / "wearable_sleep_overlap_details.json"
    resolution_csv_path = out_dir / "wearable_sleep_resolution_audit.csv"
    rows.to_csv(csv_path, index=False)
    resolution_frame = pd.DataFrame(
        resolution_rows, columns=RESOLUTION_ROW_COLUMNS
    )
    resolution_frame.to_csv(resolution_csv_path, index=False)
    v5_frame = resolution_frame.loc[
        resolution_frame.get("builder_version", pd.Series(dtype="object"))
        == "wearable-5min-5"
    ]
    resolution_summary = {
        "builder_versions_seen": dict(sorted(builder_versions_seen.items())),
        "n_artifacts_successfully_audited": int(len(resolution_frame)),
        "n_v5_artifacts": int(len(v5_frame)),
        "n_v5_people_with_exact_duplicates_removed": int(
            (
                v5_frame.get(
                    "n_exact_duplicate_events_removed", pd.Series(dtype="int64")
                )
                > 0
            ).sum()
        ),
        "n_v5_exact_duplicate_events_removed": int(
            v5_frame.get("n_exact_duplicate_events_removed", pd.Series(dtype="int64")).sum()
        ),
        "n_v5_people_with_cross_stage_conflict_all_source": int(
            (
                v5_frame.get(
                    "cross_stage_conflict_minutes_all_source",
                    pd.Series(dtype="float64"),
                )
                > 0
            ).sum()
        ),
        "v5_cross_stage_conflict_minutes_all_source": float(
            v5_frame.get(
                "cross_stage_conflict_minutes_all_source",
                pd.Series(dtype="float64"),
            ).sum()
        ),
        "n_v5_people_with_cross_stage_conflict_in_window": int(
            (
                v5_frame.get(
                    "cross_stage_conflict_minutes_in_window",
                    pd.Series(dtype="float64"),
                )
                > 0
            ).sum()
        ),
        "v5_cross_stage_conflict_minutes_in_window": float(
            v5_frame.get(
                "cross_stage_conflict_minutes_in_window",
                pd.Series(dtype="float64"),
            ).sum()
        ),
        "n_v5_conflict_slots_in_window": int(
            v5_frame.get(
                "n_conflict_slots_in_window", pd.Series(dtype="int64")
            ).sum()
        ),
        "n_v5_post_resolution_overlap_artifacts": int(
            (
                v5_frame.get(
                    "n_slots_with_stage_total_exceedance",
                    pd.Series(dtype="int64"),
                )
                > 0
            ).sum()
        ),
        "v5_maximum_total_stage_minutes_in_slot": float(
            v5_frame.get(
                "max_total_stage_minutes_in_slot", pd.Series(dtype="float64")
            ).max()
        )
        if not v5_frame.empty
        else 0.0,
        "n_all_versions_with_stage_total_exceedance": int(
            (
                resolution_frame.get(
                    "n_slots_with_stage_total_exceedance",
                    pd.Series(dtype="int64"),
                )
                > 0
            ).sum()
        ),
    }
    json_path.write_text(
        json.dumps(
            {
                "supported_builder_versions": sorted(SUPPORTED_BUILDER_VERSIONS),
                "n_index_rows": int(len(index)),
                "n_people_with_overlap_detected": int(
                    n_people_with_overlap_detected
                ),
                "n_people_with_overlap_diagnosed": int(len(candidates)),
                "n_diagnostic_errors": int(len(errors)),
                "resolution_summary": resolution_summary,
                "errors": errors,
                "people": [item["detail"] for item in candidates],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print("===== SLEEP OVERLAP DIAGNOSIS =====")
    print("index rows:", len(index))
    print("people with overlap detected:", n_people_with_overlap_detected)
    print("people with overlap diagnosed:", len(candidates))
    print("diagnostic errors:", len(errors))
    print("resolution CSV:", resolution_csv_path)
    print("summary CSV:", csv_path)
    print("detail JSON:", json_path)
    print("resolution summary:")
    print(json.dumps(resolution_summary, indent=2))
    if errors:
        print("first errors:", errors[:10])
    if rows.empty:
        if errors:
            print("BLOCKED: audit errors must be resolved before accepting sleep QC")
            raise SystemExit(1)
        else:
            print("PASS: no overlapping sleep-stage slots")
        return

    display_columns = [
        "person_id",
        "saved_overlap_slots",
        "saved_max_stage_minutes_in_slot",
        "n_raw_records_used_by_builder",
        "n_parsed_events",
        "n_direct_events",
        "n_summary_events",
        "n_records_direct_and_summary",
        "n_exact_duplicate_extra_events",
        "n_events_overlapping_previous",
        "n_intervals_over_24h",
        "duration_minutes_p50",
        "duration_minutes_p99",
        "duration_minutes_max",
        "max_slot_contributor_events",
        "max_slot_unique_stage_intervals",
        "max_slot_exact_duplicate_extra_events",
    ]
    print("\n===== TOP OFFENDERS =====")
    print(rows[display_columns].head(args.top_k).to_string(index=False))
    print("\nBLOCKED: raw sleep interval semantics must be resolved before full run")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
