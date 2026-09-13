"""Fixed definitions for seven questionnaire-history classification candidates."""
from pathlib import Path
from preparation_helpers import canonical_hash, read_json

TASK_SOURCES = {
    "kidney_history_binary": ("mhoccur_rnl", 2005200017),
    "lung_history_binary": ("mhoccur_plm", 4186898),
    "cancer_history_binary": ("mhoccur_ca", 4194405),
    "arthritis_history_binary": ("mhoccur_ra", 4291025),
    "osteoporosis_history_binary": ("mhoccur_oa", 80502),
    "digestive_history_binary": ("mhoccur_gi", 4201745),
    "urinary_history_binary": ("mhoccur_ua", 81902),
}
TASKS = tuple(TASK_SOURCES)
SPEC_PATH = Path(__file__).resolve().parent/"configs/history_tasks_v1.json"


def load_spec():
    spec = read_json(SPEC_PATH)
    if spec["schema_version"] != "ot2-history-task-spec-v1" or tuple(spec["tasks"]) != TASKS:
        raise ValueError("Unexpected history-task registry")
    for task, (source, condition_id) in TASK_SOURCES.items():
        item = spec["tasks"][task]
        if item["source_key"] != source or item["condition_concept_id"] != condition_id:
            raise ValueError("Changed source or disease meaning: "+task)
    if spec["observation_concept_id"] != 2005200014 or spec["yes_concept_ids"] != [4188539, 45877994] or spec["no_concept_id"] != 45878245 or spec["refused_concept_id"] != 36311156:
        raise ValueError("Changed questionnaire identity or response mapping")
    return spec


def task_hash(task):
    spec = load_spec()
    return canonical_hash({k: v for k, v in spec.items() if k != "tasks"} | {"task": task, "definition": spec["tasks"][task]})
