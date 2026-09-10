import copy
import tempfile
import unittest
from pathlib import Path

from basic_retrain_fixtures_v1 import source_fixture
from prepare_basic_clinical_training_v1 import prepare, read_csv, write_csv
from basic_retrain_contract_v1 import read_json, write_json, sha256, verify_prepared, canonical_hash
from task2_v2_config import load_model_v2_config


class BasicRetrainInputsTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / "source"
        self.config = source_fixture(self.source)
        self.out = self.root / "experiment"

    def test_physical_23_columns_exact_statistics_and_unchanged_other_inputs(self):
        before = {p.name: sha256(p) for p in self.source.iterdir()}
        result = prepare(self.config, self.out)
        folder = self.out / "inputs/moca_binary"
        plan = read_json(folder / "basic_plan.json")
        columns, rows = read_csv(folder / "clinical_basic.csv")
        self.assertEqual(columns, ["person_id", *plan["retained_feature_names"]])
        self.assertEqual(len(columns), 24)
        self.assertFalse(any("eos_ds" in x or "glucose" in x for x in columns))
        self.assertEqual(rows[0][columns[1]], "0")
        self.assertTrue(all(rows[1][n] == "" for n in columns[1:]))
        old, new = read_json(self.source / "scaler.json"), read_json(folder / "scalers_basic.json")
        for group in old["scalers"]:
            if group != "clinical":
                self.assertEqual(old["scalers"][group], new["scalers"][group])
        for n in columns[1:]:
            self.assertEqual(old["scalers"]["clinical"]["features"][n], new["scalers"]["clinical"]["features"][n])
        original = read_csv(self.source / "master.csv")[1]
        projected = read_csv(folder / "master.csv")[1]
        for a, b in zip(original, projected):
            for n in result["preserved_master_columns"]:
                self.assertEqual(a[n], b[n])
        self.assertEqual(projected[1]["has_clinical"], "False")
        self.assertEqual(result["n_clinical_availability_changed"], 1)
        self.assertEqual(before, {p.name: sha256(p) for p in self.source.iterdir()})
        config = load_model_v2_config(folder / "train_config.yaml", validate_files=True)
        self.assertIsNone(config.raw["training"]["initialization_checkpoint"])
        self.assertEqual(config.raw["training"], read_json(self.config)["training"])
        self.assertEqual(verify_prepared(folder), result)

    def test_check_only_is_read_only_and_prepared_inputs_are_idempotent(self):
        prepare(self.config, self.out, check_only=True)
        self.assertFalse(self.out.exists())
        a = prepare(self.config, self.out)
        b = prepare(self.config, self.out)
        self.assertEqual(a, b)

    def test_column_selection_and_scaler_do_not_depend_on_outcomes(self):
        prepare(self.config, self.out)
        columns, rows = read_csv(self.source / "master.csv")
        for r in rows:
            r["y"] = str(1 - int(r["y"]))
        write_csv(self.source / "master.csv", columns, rows)
        other = self.root / "other"
        prepare(self.config, other)
        for filename in ("clinical_basic.csv", "basic_plan.json", "scalers_basic.json"):
            self.assertEqual((self.out / "inputs/moca_binary" / filename).read_bytes(), (other / "inputs/moca_binary" / filename).read_bytes())

    def test_stale_or_missing_statistics_rejected_and_inputs_not_overwritten(self):
        prepare(self.config, self.out)
        path = self.source / "scaler.json"
        scaler = read_json(path)
        scaler["scalers"]["clinical"]["features"].pop(next(iter(scaler["scalers"]["clinical"]["features"])))
        scaler.pop("scaler_config_hash")
        scaler["scaler_config_hash"] = canonical_hash(scaler)
        write_json(path, scaler)
        with self.assertRaisesRegex(ValueError, "full fitted statistics"):
            prepare(self.config, self.out)

    def test_changed_source_or_prepared_files_fail_closed(self):
        prepare(self.config, self.out)
        cols, rows = read_csv(self.source / "clinical.csv")
        rows[0][cols[1]] = "9"
        write_csv(self.source / "clinical.csv", cols, rows)
        with self.assertRaisesRegex(ValueError, "different source files"):
            prepare(self.config, self.out)
        folder = self.out / "inputs/moca_binary"
        with (folder / "clinical_basic.csv").open("a") as f:
            f.write("corruption")
        with self.assertRaisesRegex(ValueError, "hash/path mismatch"):
            verify_prepared(folder)


if __name__ == "__main__":
    unittest.main()
