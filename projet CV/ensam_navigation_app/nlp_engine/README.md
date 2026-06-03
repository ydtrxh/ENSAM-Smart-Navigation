# ENSAM Navigation - NLP Engine

Natural-language destination extraction and entity resolution for the ENSAM Meknes campus navigation app.

The module receives text already transcribed by Whisper. Input may be French, Arabic, English, Moroccan Arabic/French mix, or a short informal query.

## Pipeline

```text
Whisper -> NLPPipeline.process() -> NavigationIntent.node_id -> A* routing -> map
Camera -> CV Model -> label -> buildings.json -> node_id -> A* routing -> map
```

The NLP pipeline has two strictly separated stages:

| Stage | File | Role |
| --- | --- | --- |
| 1 | `intent_extractor.py` | LLM extracts only the raw destination mention |
| 2 | `entity_resolver.py` | RapidFuzz resolves the mention to a GeoJSON `node_id` |

The LLM is never trusted for building names or node IDs. Entity resolution is grounded in `nlp_engine/data/campus.geojson`.

## Ollama Setup

```bash
ollama pull llama3.2:3b
ollama serve
```

`config.yaml` defaults to:

```yaml
backend: "ollama"
ollama_model: "llama3.2:3b"
```

Llama 3.2 receives the system prompt as-is. `/no_think` is only applied automatically if a Qwen model name is configured.

## GGUF Backend

For production-style local inference, download a Llama 3.2 3B Instruct GGUF from Hugging Face, preferably a `Q4_K_M` quantization, then set:

```yaml
backend: "llamacpp"
llamacpp_model_path: "models/llama-3.2-3b-instruct.Q4_K_M.gguf"
llamacpp_n_threads: 4
```

## Arabic Limitation

Llama 3.2 has weaker Arabic support than multilingual-first models. Arabic-only utterances may have lower extraction accuracy. The best mitigation is to enrich Arabic aliases in `data/campus.geojson`, because alias quality is the highest-leverage improvement in this module.

## Aliases

Aliases are the primary performance lever of this module. Improving aliases improves resolution more than upgrading models.

To improve a destination, edit its GeoJSON feature:

```json
{
  "properties": {
    "id": "bibliotheque",
    "label": "Bibliothèque",
    "aliases": ["bibliothèque", "bibliotheque", "library", "المكتبة", "biblio"]
  }
}
```

Good aliases include French, English, Arabic, transliteration, abbreviations, and common misspellings.

## Graph Connection

`NavigationIntent.node_id` must exist in the navigation graph imported from:

```text
data/campus_graph.json
```

That graph is synchronized into Neo4j by:

```text
scripts/sync_campus_graph.py
```

The CV route uses:

```text
CV model -> predicted label -> data/buildings.json -> node_id -> A* routing -> map
```

So `data/buildings.json` node IDs must also exist in `data/campus_graph.json`.

## Testing

From `ensam_navigation_app/`:

```bash
python -m compileall -q nlp_engine
python -m nlp_engine.test_pipeline
```

The test script prints each `NavigationIntent` and a final summary:

- total utterances
- resolved
- unknown
- average confidence
- Arabic resolved count
- mixed resolved count
- failed exceptions
