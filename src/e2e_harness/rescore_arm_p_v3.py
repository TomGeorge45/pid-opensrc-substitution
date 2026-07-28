"""
Re-scores the 4 saved Arm P v3 job artifact stores (already-computed GPT-5.5-low
entities/relations, from the run on 2026-07-16) using the corrected ground truth
(contract_to_equipment_edges, degree-4 heuristic) instead of PID2Graph's raw edges.

No new LLM calls - this only fixes the grading, not the model output. See
ground_truth.py's contract_to_equipment_edges docstring for why the original
relation-F1=0.0-everywhere result was a metric artifact, not a real score.
"""
import json
import sys

sys.path.insert(0, "src")

from e2e_harness.graph_matcher import match_entities, match_relations
from e2e_harness.ground_truth import contract_to_equipment_edges, equipment_only, parse_graphml_ground_truth
from e2e_harness.holdout import holdout_sheet_paths

from pnid_agent.models.rive_ontology import RiveOntology
from pnid_agent.storage.local_fs import LocalFsArtifactStore

# sheet_id -> (job_dir, job_id) for the FINAL run of each sheet (picked by mtime,
# see conversation - earlier duplicate OPEN100_8 dir from a standalone pre-test discarded)
JOB_DIRS = {
    "PID2GraphOPEN100_8": "/var/folders/71/9qgs5h1s46j9zcbtytxb5lym0000gn/T/tmpevodp2bc",
    "PID2GraphOPEN100_1": "/var/folders/71/9qgs5h1s46j9zcbtytxb5lym0000gn/T/tmpucedzk9n",
    "DatasetPID_246": "/var/folders/71/9qgs5h1s46j9zcbtytxb5lym0000gn/T/tmpilnsnqiz",
    "DatasetPID_443": "/var/folders/71/9qgs5h1s46j9zcbtytxb5lym0000gn/T/tmpsd6n0ibl",
}


def score(entities, relations, gt_equip, gt_edges_contracted, label):
    em = match_entities(entities, gt_equip)
    rm = match_relations(relations, gt_edges_contracted, em)
    return {
        "checkpoint": label, "n_entities": len(entities), "n_relations": len(relations),
        "entity_precision": em.precision, "entity_recall": em.recall, "entity_f1": em.f1,
        "relation_precision": rm.precision, "relation_recall": rm.recall, "relation_f1": rm.f1,
    }


def main():
    sheets = holdout_sheet_paths()
    all_results = []
    for sheet in sheets:
        sheet_id = sheet["sheet_id"]
        job_dir = JOB_DIRS[sheet_id]
        job_id = "job-armPv3-" + sheet_id
        store = LocalFsArtifactStore(job_dir)

        gt_entities, gt_edges_raw = parse_graphml_ground_truth(sheet["graphml_path"])
        gt_equip = equipment_only(gt_entities)
        gt_edges_contracted = contract_to_equipment_edges(gt_entities, gt_edges_raw)
        print(f"\n=== {sheet_id} === GT: {len(gt_equip)} equipment, "
              f"{len(gt_edges_raw)} raw edges -> {len(gt_edges_contracted)} contracted equipment edges")

        checkpoints = {}
        for stage_uri, label in [
            ("stage-11/rive_ontology.json", "pre_13_12"),
            ("stage-13/rive_ontology.json", "post_13"),
            ("stage-12/rive_ontology.json", "post_12"),
        ]:
            raw = store.read_json(job_id, stage_uri)
            rive = RiveOntology.model_validate(raw)
            r = score(rive.entities, rive.relations, gt_equip, gt_edges_contracted, label)
            checkpoints[label] = r
            print(f"  [{label}] entities={r['n_entities']} relations={r['n_relations']} "
                  f"entity F1={r['entity_f1']:.3f} | relation P={r['relation_precision']:.3f} "
                  f"R={r['relation_recall']:.3f} F1={r['relation_f1']:.3f}")

        all_results.append({"sheet_id": sheet_id, "n_gt_entities": len(gt_equip),
                             "n_gt_edges_contracted": len(gt_edges_contracted), "checkpoints": checkpoints})

    with open("/tmp/arm_p_v3_rescored.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nwritten to /tmp/arm_p_v3_rescored.json")
    return all_results


if __name__ == "__main__":
    main()
