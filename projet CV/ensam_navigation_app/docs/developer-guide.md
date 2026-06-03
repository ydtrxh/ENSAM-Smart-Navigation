# Developer Guide

This guide explains how contributors should work on the project.

## Development Principles

- Keep runtime code separate from training scripts.
- Keep generated files out of Git.
- Treat `data/campus_graph.json` as the current graph source of truth.
- Treat `data/buildings.json` as the CV-to-navigation mapping source of truth.
- Add tests for every graph or mapping change.
- Avoid hardcoding class labels.

## Useful Commands

Compile Python modules:

```bash
python -m compileall -q app cv_engine nlp_engine src scripts tests
```

Run tests:

```bash
python -m pytest
```

Run graph contract tests without pytest:

```bash
python -c "import tests.test_graph_contract as t; t.test_all_building_node_ids_exist_in_campus_graph(); t.test_campus_graph_edges_reference_existing_nodes(); t.test_campus_graph_edges_have_positive_distance(); print('ok')"
```

Serve documentation locally:

```bash
mkdocs serve
```

Build documentation:

```bash
mkdocs build --strict
```

## Documentation Deployment

### GitHub Pages

```bash
mkdocs gh-deploy
```

### Read the Docs

Read the Docs can build the documentation from `mkdocs.yml`. The Python dependencies are listed in `requirements-dev.txt`.

Recommended Read the Docs settings:

| Setting | Value |
| --- | --- |
| Documentation type | MkDocs |
| Configuration file | `mkdocs.yml` |
| Python requirements | `requirements-dev.txt` |

