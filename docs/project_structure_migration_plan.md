# Project Structure Migration Plan

This document proposes a production-ready reorganization for the ENSAM Smart Navigation System without changing runtime behavior immediately.

The current project works, but it still has a research-workspace shape: production code, experiments, logs, backups, data artifacts, training code, inference code, and graph tools are mixed together. The goal is to move toward a professional AI engineering repository while preserving compatibility with the existing Streamlit application.

## Goals

- Separate runtime application code from training code.
- Separate CV inference from CV retraining.
- Separate NLP runtime code from campus data and configuration.
- Make the navigation graph a clear source of truth.
- Keep generated files, logs, datasets, and checkpoints out of Git.
- Make the repository easier to review by professors, recruiters, and engineers.
- Support retraining, evaluation, deployment, and future expansion.

## Current High-Level Structure

```text
ensam_navigation_app/
|-- app/
|-- cv_engine/
|-- nlp_engine/
|-- src/
|-- scripts/
|-- data/
|-- checkpoints/
|-- outputs/
|-- interactive_map/
|-- docs/
|-- root-level scripts
|-- logs
`-- README.md
```

## Main Issues

| Issue | Impact |
| --- | --- |
| `src/` and `cv_engine/` both contain CV pipeline logic | Unclear source of truth for training and inference |
| `checkpoints/` and `cv_engine/checkpoints/` both exist | Ambiguous active model path |
| `outputs/` and `cv_engine/outputs/` both exist | Evaluation artifacts are scattered |
| `data/` mixes datasets, graph files, reports, backups, and migration artifacts | Hard to distinguish source data from generated data |
| Experimental scripts live at repository root | Root looks unpolished and hard to navigate |
| Local logs are tracked candidates | GitHub repository may expose noise |
| Old TensorFlow `train_model.py` remains beside PyTorch pipeline | Confuses the technical story |
| `map_manager_backup_1000.py` is in runtime package | Backup code may be mistaken for active code |

## Target Folder Structure

```text
ensam-navigation/
|-- app/
|   |-- streamlit/
|   |   |-- main.py
|   |   `-- components/
|   |-- cv/
|   |   |-- inference.py
|   |   |-- model.py
|   |   `-- preprocessing.py
|   |-- nlp/
|   |   |-- pipeline.py
|   |   |-- intent_extractor.py
|   |   |-- entity_resolver.py
|   |   `-- llm_backend.py
|   |-- navigation/
|   |   |-- engine.py
|   |   |-- graph_store.py
|   |   |-- map_manager.py
|   |   `-- instructions.py
|   `-- static/
|       `-- campus_map_2d1.svg
|
|-- training/
|   `-- cv/
|       |-- augment.py
|       |-- build_val.py
|       |-- dataset.py
|       |-- losses.py
|       |-- retrain.py
|       |-- evaluate.py
|       `-- sync_check.py
|
|-- graph/
|   |-- data/
|   |   |-- campus_graph.json
|   |   |-- campus.geojson
|   |   |-- buildings.json
|   |   `-- navigation_aliases.geojson
|   |-- scripts/
|   |   |-- sync_neo4j.py
|   |   |-- validate_graph.py
|   |   |-- audit_routing_graph.py
|   |   `-- migrate_ensam360_graph.py
|   `-- editor/
|       |-- svg_graph_editor.html
|       |-- svg_graph_editor.css
|       `-- svg_graph_editor.js
|
|-- data/
|   |-- raw/
|   |-- gallery/
|   |-- train/
|   |-- val/
|   `-- external/
|
|-- models/
|   |-- checkpoints/
|   |   |-- best_model.pth
|   |   `-- gallery.pkl
|   `-- legacy/
|
|-- outputs/
|   |-- training/
|   |-- evaluation/
|   |-- graphs/
|   `-- reports/
|
|-- configs/
|   |-- app.yaml
|   |-- nlp.yaml
|   |-- training.yaml
|   `-- neo4j.env.example
|
|-- scripts/
|   |-- run_app.py
|   |-- setup_project.py
|   `-- export_demo_assets.py
|
|-- tests/
|   |-- test_cv_inference.py
|   |-- test_nlp_pipeline.py
|   |-- test_graph_validation.py
|   `-- test_navigation_routes.py
|
|-- docs/
|   |-- architecture.md
|   |-- dataset.md
|   |-- retraining.md
|   |-- routing_graph.md
|   `-- assets/
|
|-- notebooks/
|   `-- experiments/
|
|-- docker-compose.yml
|-- requirements.txt
|-- pyproject.toml
|-- README.md
|-- LICENSE
`-- .gitignore
```

## File Movement Plan

### Runtime Application

| Current path | Target path | Notes |
| --- | --- | --- |
| `app/main.py` | `app/streamlit/main.py` | Streamlit entrypoint |
| `app/navigation/engine.py` | `app/navigation/engine.py` | Keep as runtime navigation |
| `app/navigation/graph_store.py` | `app/navigation/graph_store.py` | Keep as runtime Neo4j access |
| `app/navigation/map_manager.py` | `app/navigation/map_manager.py` | Keep as runtime Leaflet/SVG renderer |
| `app/navigation/map_manager_backup_1000.py` | `archive/backups/map_manager_backup_1000.py` | Backup, not production |
| `app/static/campus_map_2d1.svg` | `app/static/campus_map_2d1.svg` | Runtime asset |

### CV Runtime

| Current path | Target path | Notes |
| --- | --- | --- |
| `cv_engine/inference.py` | `app/cv/inference.py` | Runtime prediction code |
| `cv_engine/model.py` | `app/cv/model.py` | Runtime model definition |
| `cv_engine/dataset.py` | Review before moving | May duplicate `src/dataset.py` |
| `cv_engine/loss.py` | `training/cv/losses_legacy.py` or remove | Training concern, not runtime |
| `cv_engine/train.py` | `training/cv/train_legacy.py` | Older training path |
| `cv_engine/evaluate.py` | `training/cv/evaluate_legacy.py` | Older evaluation path |

### CV Training Pipeline

| Current path | Target path | Notes |
| --- | --- | --- |
| `src/build_val.py` | `training/cv/build_val.py` | Validation split generation |
| `src/sync_check.py` | `training/cv/sync_check.py` | Dataset/building consistency |
| `src/augment.py` | `training/cv/augment.py` | Offline augmentation |
| `src/dataset.py` | `training/cv/dataset.py` | Main training dataset implementation |
| `src/losses.py` | `training/cv/losses.py` | Main metric-learning losses |
| `src/retrain.py` | `training/cv/retrain.py` | Main retraining entrypoint |
| `src/evaluate.py` | `training/cv/evaluate.py` | Main evaluation entrypoint |

### NLP

| Current path | Target path | Notes |
| --- | --- | --- |
| `nlp_engine/pipeline.py` | `app/nlp/pipeline.py` | Runtime NLP pipeline |
| `nlp_engine/intent_extractor.py` | `app/nlp/intent_extractor.py` | Runtime extraction |
| `nlp_engine/entity_resolver.py` | `app/nlp/entity_resolver.py` | Runtime resolution |
| `nlp_engine/llm_backend.py` | `app/nlp/llm_backend.py` | Optional Ollama/LLM |
| `nlp_engine/config.yaml` | `configs/nlp.yaml` | Config belongs in configs |
| `nlp_engine/data/campus.geojson` | `graph/data/navigation_aliases.geojson` | Destination aliases |
| `nlp_engine/test_pipeline.py` | `tests/test_nlp_pipeline.py` | Test, not runtime |

### Graph Management

| Current path | Target path | Notes |
| --- | --- | --- |
| `data/campus_graph.json` | `graph/data/campus_graph.json` | Graph source of truth |
| `data/buildings.json` | `graph/data/buildings.json` | CV class to graph node mapping |
| `data/campus.geojson` | `graph/data/campus.geojson` | Map/geographic metadata |
| `scripts/sync_campus_graph.py` | `graph/scripts/sync_neo4j.py` | Neo4j import |
| `scripts/audit_routing_graph.py` | `graph/scripts/audit_routing_graph.py` | Route realism audit |
| `scripts/extract_ensam360_reference.py` | `graph/scripts/extract_ensam360_reference.py` | Migration utility |
| `scripts/convert_seed.py` | `graph/scripts/convert_seed.py` | Migration utility |
| `scripts/seed_neo4j.py` | `graph/scripts/seed_neo4j.py` | Optional seeding |
| `interactive_map/*` | `graph/editor/*` | SVG graph editor |

### Data and Artifacts

| Current path | Target path | Notes |
| --- | --- | --- |
| `data/raw/` | `data/raw/` | Dataset, usually not committed |
| `data/gallery/` | `data/gallery/` | Dataset, usually not committed |
| `data/train/` | `data/train/` | Generated, do not commit |
| `data/val/` | `data/val/` | Usually generated or private |
| `checkpoints/best_model.pth` | `models/checkpoints/best_model.pth` | Active model artifact |
| `checkpoints/gallery.pkl` | `models/checkpoints/gallery.pkl` | Active gallery artifact |
| `cv_engine/checkpoints/*` | `models/legacy/` | Legacy artifacts |
| `outputs/*` | `outputs/training/` or `outputs/evaluation/` | Generated results |

### Archive or Remove

| Current path | Recommendation | Reason |
| --- | --- | --- |
| `train_model.py` | `archive/legacy/train_model_tensorflow.py` | Old TensorFlow approach |
| `scratch_graph.py` | `archive/experiments/scratch_graph.py` | Experimental |
| `generate_geojson.py` | `graph/scripts/generate_geojson.py` or archive | Utility |
| `inspect_geojson.py` | `graph/scripts/inspect_geojson.py` or archive | Utility |
| `perspective_transform.py` | `graph/scripts/perspective_transform.py` or archive | Map utility |
| `crash_log*.txt` | Delete | Runtime noise |
| `streamlit_*.log` | Delete | Runtime noise |
| `model_metrics.json` | `outputs/evaluation/model_metrics_legacy.json` | Obsolete 4-class metrics |

## Migration Phases

Current status:

| Phase | Status |
| --- | --- |
| Phase 0: Repository Hygiene | Done |
| Phase 1: Stabilize Runtime Paths | Done |
| Phase 2: Move Graph Tooling | Not started |
| Phase 3: Move CV Training | Not started |
| Phase 4: Move Runtime CV and NLP | Not started |
| Phase 5: Tests and CI | Partially started |

### Phase 0: Repository Hygiene

No code movement.

- Add `.gitignore`.
- Add this migration plan.
- Create `docs/assets/`.
- Add screenshots and demo images later.
- Do not commit datasets or checkpoints unless intentionally using Git LFS.

Validation:

```bash
python -m compileall -q app cv_engine nlp_engine src scripts
python -m streamlit run app\main.py
```

### Phase 1: Stabilize Runtime Paths

- Add config helpers for paths.
- Keep old paths working.
- Introduce future paths like `models/checkpoints/`.
- Make `data/buildings.json`, `data/campus_graph.json`, and model paths configurable.

Validation:

```bash
python -m compileall -q app cv_engine nlp_engine src scripts
python scripts\sync_campus_graph.py
python -m streamlit run app\main.py
```

### Phase 2: Move Graph Tooling

- Move graph scripts under `graph/scripts/`.
- Move SVG graph editor under `graph/editor/`.
- Keep temporary wrappers in `scripts/` if needed.

Validation:

```bash
python graph\scripts\sync_neo4j.py
python graph\scripts\audit_routing_graph.py
```

### Phase 3: Move CV Training

- Move `src/` training pipeline to `training/cv/`.
- Keep `python -m src.retrain` wrappers temporarily.
- Update README commands only after wrappers are tested.

Validation:

```bash
python -m training.cv.sync_check --data_dir data
python -m training.cv.evaluate --data_dir data
```

### Phase 4: Move Runtime CV and NLP

- Move inference/model code into `app/cv/`.
- Move NLP runtime into `app/nlp/`.
- Update imports in Streamlit app.
- Preserve old package wrappers for compatibility.

Validation:

```bash
python -m compileall -q app
python -m nlp_engine.test_pipeline
python -m streamlit run app\streamlit\main.py
```

### Phase 5: Tests and CI

- Add tests for graph connectivity.
- Add test for every `buildings.json` `node_id` existing in the graph.
- Add smoke test for CV model loading.
- Add NLP alias resolution tests.
- Add GitHub Actions.

Suggested CI checks:

```bash
python -m compileall -q app training graph scripts tests
python -m pytest -q
```

## Compatibility Rules

During migration:

- Do not break `python -m streamlit run app\main.py` until the new entrypoint is ready.
- Do not remove `data/campus_graph.json` until all scripts read from `graph/data/campus_graph.json`.
- Do not remove `checkpoints/best_model.pth` until the app reads from `models/checkpoints/best_model.pth`.
- Keep wrappers for old command paths during the transition.
- Migrate one subsystem at a time and test after each move.

## Recommended GitHub Cleanup Before Publication

- Add `LICENSE`.
- Add `docs/assets/banner.png`.
- Add screenshots:
  - app home,
  - CV prediction,
  - NLP destination resolution,
  - route visualization,
  - Neo4j graph view,
  - confusion matrix,
  - t-SNE,
  - SVG graph editor.
- Add a short demo video link.
- Add a `requirements-dev.txt` or `pyproject.toml` later.
- Consider Git LFS for model files if checkpoints must be published.

## Final Recommendation

Do not perform a full folder move in one operation. The safest path is:

```text
document -> ignore generated files -> centralize config -> move graph tools -> move training -> move runtime packages -> add tests
```

This keeps the current Streamlit app usable while making the repository progressively more professional.
