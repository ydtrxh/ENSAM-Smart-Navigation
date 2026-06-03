# Installation

This page explains how to install the project for local development and connect it to Neo4j AuraDB.

## Requirements

- Python 3.10 or newer
- Neo4j AuraDB instance
- Git
- Optional: Ollama for local LLM-based destination extraction

## Clone the Repository

```bash
git clone https://github.com/your-username/ensam-smart-navigation-system.git
cd ensam-smart-navigation-system
```

## Create a Virtual Environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

## Install Runtime Dependencies

```bash
pip install -r requirements.txt
```

## Install Developer Documentation Dependencies

```bash
pip install -r requirements-dev.txt
```

This installs MkDocs Material, test tools, and linting tools.

## Configure Neo4j AuraDB

Create a `.env` file at the project root:

```env
NEO4J_URI=neo4j+s://your-aura-instance.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-aura-generated-password
```

The `.env` file is ignored by Git because it contains credentials.

## Synchronize the Graph

After configuring Neo4j, import `data/campus_graph.json` into AuraDB:

```bash
python scripts\sync_campus_graph.py
```

## Verify the Installation

```bash
python -m compileall -q app cv_engine nlp_engine src scripts tests
python -c "import tests.test_graph_contract as t; t.test_all_building_node_ids_exist_in_campus_graph(); print('ok')"
```

If `pytest` is installed:

```bash
python -m pytest
```

