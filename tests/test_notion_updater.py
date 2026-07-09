"""Tests for scripts.notion_portfolio_update."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def test_projects_list_not_empty() -> None:
    from scripts.notion_portfolio_update import PROJECTS

    assert len(PROJECTS) > 0


def test_projects_have_required_fields() -> None:
    from scripts.notion_portfolio_update import PROJECTS

    for project in PROJECTS:
        assert "page_id" in project, f"Missing page_id in {project}"
        assert "name" in project, f"Missing name in {project}"
        assert "cover" in project, f"Missing cover in {project}"
        assert "tags" in project, f"Missing tags in {project}"


def test_projects_page_ids_unique() -> None:
    from scripts.notion_portfolio_update import PROJECTS

    page_ids = [str(p["page_id"]) for p in PROJECTS]
    assert len(page_ids) == len(set(page_ids)), "Duplicate page_ids found"


def test_projects_tags_are_lists() -> None:
    from scripts.notion_portfolio_update import PROJECTS

    for project in PROJECTS:
        tags = project.get("tags")
        assert isinstance(tags, list), f"{project['name']} tags should be a list"
        assert len(tags) > 0, f"{project['name']} tags should not be empty"


def test_cover_base_is_https() -> None:
    from scripts.notion_portfolio_update import COVER_BASE

    assert COVER_BASE.startswith("https://")


def test_update_notion_page_called_with_correct_args() -> None:
    from scripts.notion_portfolio_update import update_notion_page

    mock_notion = MagicMock()
    update_notion_page(mock_notion, "page-123", "https://example.com/cover.svg", ["Python"])
    mock_notion.pages.update.assert_called_once()
    call_kwargs = mock_notion.pages.update.call_args[1]
    assert call_kwargs["page_id"] == "page-123"
    assert call_kwargs["cover"]["external"]["url"] == "https://example.com/cover.svg"


def test_update_notion_page_with_description() -> None:
    from scripts.notion_portfolio_update import update_notion_page

    mock_notion = MagicMock()
    update_notion_page(
        mock_notion, "page-456", "https://example.com/cover.svg", ["Python"], "A great project."
    )
    call_kwargs = mock_notion.pages.update.call_args[1]
    props = call_kwargs["properties"]
    assert "Description" in props


def test_update_notion_page_without_description() -> None:
    from scripts.notion_portfolio_update import update_notion_page

    mock_notion = MagicMock()
    update_notion_page(mock_notion, "page-789", "https://example.com/cover.svg", ["SQL"])
    call_kwargs = mock_notion.pages.update.call_args[1]
    props = call_kwargs["properties"]
    assert "Description" not in props


def test_main_raises_without_notion_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NOTION_API_KEY", raising=False)
    from scripts import notion_portfolio_update as m

    m.NOTION_API_KEY = ""
    with pytest.raises(ValueError, match="NOTION_API_KEY"):
        m.main(generate_descriptions=False)


@pytest.mark.parametrize(
    "project_name",
    ["MedGenomics", "Price-Prophet", "FraudDetectionAI", "Rag-Sentinel", "reflective-lantern"],
)
def test_known_projects_in_registry(project_name: str) -> None:
    from scripts.notion_portfolio_update import PROJECTS

    names = [str(p["name"]) for p in PROJECTS]
    assert project_name in names


def test_cover_base_url_ends_with_covers() -> None:
    from scripts.notion_portfolio_update import COVER_BASE

    assert COVER_BASE.endswith("covers") or "/covers" in COVER_BASE


@pytest.mark.parametrize("project_count", [5, 10])
def test_projects_minimum_count(project_count: int) -> None:
    from scripts.notion_portfolio_update import PROJECTS

    assert len(PROJECTS) >= project_count


def test_update_notion_page_sets_tags_property() -> None:
    from scripts.notion_portfolio_update import update_notion_page

    mock_notion = MagicMock()
    update_notion_page(mock_notion, "page-abc", "https://example.com/img.svg", ["ML", "Python"])
    call_kwargs = mock_notion.pages.update.call_args[1]
    props = call_kwargs["properties"]
    assert "Tags" in props


def test_project_names_are_strings() -> None:
    from scripts.notion_portfolio_update import PROJECTS

    for project in PROJECTS:
        assert isinstance(project["name"], str)
        assert len(project["name"]) > 0


def test_project_covers_non_empty() -> None:
    from scripts.notion_portfolio_update import PROJECTS

    for project in PROJECTS:
        cover = project.get("cover", "")
        assert cover, f"{project['name']} has an empty cover field"


def test_project_names_helper_returns_list() -> None:
    from scripts.notion_portfolio_update import project_names

    names = project_names()
    assert isinstance(names, list)
    assert len(names) > 0


def test_project_names_helper_sorted() -> None:
    from scripts.notion_portfolio_update import project_names

    names = project_names()
    assert names == sorted(names)


def test_project_names_helper_all_strings() -> None:
    from scripts.notion_portfolio_update import project_names

    for name in project_names():
        assert isinstance(name, str)
        assert len(name) > 0


def test_project_names_no_duplicates() -> None:
    from scripts.notion_portfolio_update import project_names

    names = project_names()
    assert len(names) == len(set(names)), "project_names() contains duplicates"


def test_projects_list_has_name_field() -> None:
    from scripts.notion_portfolio_update import PROJECTS

    for project in PROJECTS:
        assert "name" in project, "Project missing name key"


def test_projects_list_is_non_empty() -> None:
    from scripts.notion_portfolio_update import PROJECTS

    assert isinstance(PROJECTS, list)
    assert len(PROJECTS) > 0


def test_project_names_count_matches_projects() -> None:
    from scripts.notion_portfolio_update import PROJECTS, project_names

    assert len(project_names()) == len(PROJECTS)
