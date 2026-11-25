"""OpenAPI metadata and tag descriptions for Support-Sage."""

TAGS_METADATA = [
    {
        "name": "triage",
        "description": "Ticket priority classification and resolution time estimation.",
    },
    {
        "name": "rag",
        "description": "FAISS-powered similar ticket retrieval for context-augmented triage.",
    },
    {
        "name": "batch",
        "description": "Bulk triage for processing multiple tickets in a single request.",
    },
    {
        "name": "monitoring",
        "description": "Model drift detection and performance metrics.",
    },
]

API_DESCRIPTION = """
## Support-Sage

**IT helpdesk ticket triage API** using an XGBoost + LightGBM + RandomForest ensemble.

### Capabilities

- **Priority prediction**: critical / high / medium / low with confidence scores
- **Resolution time estimation**: hours to resolution (LightGBM regressor)
- **RAG retrieval**: FAISS-indexed similar historical tickets
- **Drift detection**: KS-test on confidence, resolution-time, and word-count distributions
- **Batch processing**: up to 50 tickets in a single call
"""
