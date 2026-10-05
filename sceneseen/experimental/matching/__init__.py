"""EXPERIMENTAL — DISABLED: automatic product matching (embeddings, vector index, ranking).

The measured accuracy was not good enough to guess brands or products for users, so SceneSeen's
workflow is: the detector finds generic objects, a person identifies the product
(sceneseen/catalog/identification.py). This package is NOT imported by the server, the CLI or
the pipeline. It never runs automatically, never assigns a product and never appears in the UI
or in the final scene output. It stays so the question "is automatic matching accurate enough
yet?" can be re-evaluated against human identifications later (scripts/product_matching_study.py,
docs/EXPERIMENTAL_MATCHING.md).
"""
