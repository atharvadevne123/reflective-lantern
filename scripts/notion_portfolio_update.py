#!/usr/bin/env python3
"""
Batch-update Notion portfolio Case Studies database.
Uses the Anthropic SDK to generate descriptions, then patches each page
via the Notion API (cover + tags + description).

Usage:
    pip install anthropic notion-client python-dotenv
    export ANTHROPIC_API_KEY=sk-...
    export NOTION_API_KEY=secret_...
    python notion_portfolio_update.py
"""

import os
import json
import time
from typing import Optional
import anthropic
from notion_client import Client as NotionClient
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
NOTION_API_KEY = os.environ["NOTION_API_KEY"]
COVER_BASE = "https://raw.githubusercontent.com/atharvadevne123/reflective-lantern/main/covers"

# ── Project registry ────────────────────────────────────────────────────────
# page_id: Notion page UUID
# cover:   SVG filename (no extension) under COVER_BASE
# tags:    existing Case Studies multi-select options
# github:  repo name on atharvadevne123

PROJECTS = [
    {
        "page_id": "1c2c64d2-8030-80f2-b711-d22cc503b0af",
        "name": "Healthcare - Heart Failure",
        "cover": "healthcare-heart-failure",
        "tags": ["Healthcare", "Python", "Artificial Intelligence", "Statistics", "Analysis"],
        "github": None,
    },
    {
        "page_id": "34fc64d2-8030-8156-b22b-c22c72fafbba",
        "name": "MedGenomics",
        "cover": "medgenomics",
        "tags": ["Healthcare", "Python", "FastAPI", "PostgreSQL", "REST API"],
        "github": "MedGenomics",
    },
    {
        "page_id": "34fc64d2-8030-81e7-8780-ecd0bb72ea38",
        "name": "Price-Prophet",
        "cover": "price-prophet",
        "tags": ["Python", "Machine Learning", "Streamlit", "Statistics", "Regression"],
        "github": "Price-Prophet",
    },
    {
        "page_id": "34fc64d2-8030-8127-92c2-cb6ed78fcf41",
        "name": "FraudDetectionAI",
        "cover": "frauddetectionai",
        "tags": ["Python", "Artificial Intelligence", "TensorFlow", "Machine Learning"],
        "github": "FraudDetectionAI",
    },
    {
        "page_id": "34fc64d2-8030-812b-b033-fea243efbcfb",
        "name": "Enterprise-Analytics-Platform",
        "cover": "enterprise-analytics-platform",
        "tags": ["Python", "SQL", "Data Warehouse", "Data Visualization", "Analysis"],
        "github": "Enterprise-Analytics-Platform",
    },
    {
        "page_id": "34fc64d2-8030-81ad-84a8-fbdb734c44ec",
        "name": "Stock-Market-Intelligence",
        "cover": "stock-market-intelligence",
        "tags": ["Python", "LLM", "Streamlit", "Artificial Intelligence", "Machine Learning"],
        "github": "Stock-Market-Intelligence",
    },
    {
        "page_id": "34fc64d2-8030-8163-b422-c854f657eb5f",
        "name": "Rag-Sentinel",
        "cover": "rag-sentinel",
        "tags": ["Python", "LLM", "Artificial Intelligence", "FastAPI", "RAG"],
        "github": "Rag-Sentinel",
    },
    {
        "page_id": "34fc64d2-8030-81ec-9538-c4a1732b07e9",
        "name": "Automated-json-pipeline-snowflake-streams-tasks",
        "cover": "automated-json-pipeline",
        "tags": ["Snowflake", "Snowflake SQL", "ETL", "Python", "JSON processing"],
        "github": "Automated-json-pipeline-snowflake-streams-tasks",
    },
    {
        "page_id": "34fc64d2-8030-81a7-8086-decfc53b5b7a",
        "name": "Automated-data-preprocessing-udf-sql-pipeline",
        "cover": "automated-data-preprocessing",
        "tags": ["Snowflake", "SQL", "ETL", "Python", "UDF integration"],
        "github": "Automated-data-preprocessing-udf-sql-pipeline",
    },
    {
        "page_id": "34fc64d2-8030-81df-a7cf-ca78326443b1",
        "name": "reflective-lantern",
        "cover": "reflective-lantern",
        "tags": ["Python", "Artificial Intelligence", "LLM", "Automation"],
        "github": "reflective-lantern",
    },
    {
        "page_id": "34fc64d2-8030-817f-8e06-e9c44e3bc465",
        "name": "Clinical-Trial-Cohort-Matching-System",
        "cover": "clinical-trial-cohort-matching",
        "tags": ["Healthcare", "Python", "LLM", "Artificial Intelligence", "NLP"],
        "github": "Clinical-Trial-Cohort-Matching-System",
    },
    {
        "page_id": "34fc64d2-8030-818f-9f2d-e8cc21174e88",
        "name": "Employee_Management_System_API",
        "cover": "employee-management-system",
        "tags": ["Python", "FastAPI", "PostgreSQL", "SQL", "REST API"],
        "github": "Employee_Management_System_API",
    },
    {
        "page_id": "34fc64d2-8030-81b7-abe1-d5e934d518d1",
        "name": "HR-Dashboard-PowerBI",
        "cover": "hr-dashboard-powerbi",
        "tags": ["SQL", "Data Visualization", "ETL", "Power BI"],
        "github": "HR-Dashboard-PowerBI",
    },
]


def generate_description(client: anthropic.Anthropic, project: dict) -> str:
    """Ask Claude for a 2-sentence portfolio description for a project."""
    github_hint = (
        f"GitHub repo: github.com/atharvadevne123/{project['github']}"
        if project["github"]
        else ""
    )
    prompt = (
        f"Write a concise 2-sentence portfolio description for the project \"{project['name']}\". "
        f"Tags: {', '.join(project['tags'])}. {github_hint} "
        "Be specific, professional, and highlight the technical value. No fluff."
    )
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


def update_notion_page(
    notion: NotionClient,
    page_id: str,
    cover_url: str,
    tags: list[str],
    description: Optional[str] = None,
) -> None:
    """Patch a Notion page: cover image, Tags multi-select, and Description."""
    properties: dict = {
        "Tags": {
            "multi_select": [{"name": t} for t in tags]
        }
    }
    if description:
        properties["Description"] = {
            "rich_text": [{"type": "text", "text": {"content": description}}]
        }

    notion.pages.update(
        page_id=page_id,
        cover={"type": "external", "external": {"url": cover_url}},
        properties=properties,
    )


def main(generate_descriptions: bool = False) -> None:
    ai = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    notion = NotionClient(auth=NOTION_API_KEY)

    results = []
    for project in PROJECTS:
        print(f"→ {project['name']}")
        cover_url = f"{COVER_BASE}/{project['cover']}.svg"

        description = None
        if generate_descriptions:
            description = generate_description(ai, project)
            print(f"  description: {description[:80]}…")

        try:
            update_notion_page(
                notion,
                page_id=project["page_id"],
                cover_url=cover_url,
                tags=project["tags"],
                description=description,
            )
            results.append({"name": project["name"], "status": "ok"})
            print("  ✓ updated")
        except Exception as exc:
            results.append({"name": project["name"], "status": "error", "error": str(exc)})
            print(f"  ✗ {exc}")

        time.sleep(0.3)  # stay under Notion rate limit

    print("\nSummary:")
    for r in results:
        icon = "✓" if r["status"] == "ok" else "✗"
        print(f"  {icon} {r['name']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch-update Notion portfolio pages")
    parser.add_argument(
        "--descriptions",
        action="store_true",
        help="Generate AI descriptions via Claude and write them to Notion",
    )
    args = parser.parse_args()
    main(generate_descriptions=args.descriptions)
