"""Tests for app.collection_utils module."""

from __future__ import annotations

import pytest

from app.collection_utils import (
    chunk,
    deep_merge,
    flatten,
    frequency_map,
    unique_ordered,
)


class TestChunk:
    def test_even_split(self) -> None:
        result = chunk([1, 2, 3, 4], 2)
        assert result == [[1, 2], [3, 4]]

    def test_uneven_split(self) -> None:
        result = chunk([1, 2, 3, 4, 5], 2)
        assert result == [[1, 2], [3, 4], [5]]

    def test_single_chunk(self) -> None:
        result = chunk([1, 2, 3], 10)
        assert result == [[1, 2, 3]]

    def test_size_one(self) -> None:
        result = chunk([1, 2, 3], 1)
        assert result == [[1], [2], [3]]

    def test_empty_list(self) -> None:
        assert chunk([], 5) == []

    def test_size_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            chunk([1, 2], 0)

    @pytest.mark.parametrize(
        ("size", "expected_chunks"),
        [(1, 4), (2, 2), (3, 2), (4, 1)],
    )
    def test_chunk_count(self, size: int, expected_chunks: int) -> None:
        result = chunk([1, 2, 3, 4], size)
        assert len(result) == expected_chunks


class TestFlatten:
    def test_flat_list_unchanged(self) -> None:
        assert flatten([1, 2, 3]) == [1, 2, 3]

    def test_nested_list(self) -> None:
        assert flatten([[1, 2], [3, 4]]) == [1, 2, 3, 4]

    def test_mixed_list(self) -> None:
        assert flatten([1, [2, 3], 4]) == [1, 2, 3, 4]

    def test_empty_sublists(self) -> None:
        assert flatten([[], [1], []]) == [1]

    def test_empty_input(self) -> None:
        assert flatten([]) == []

    def test_preserves_order(self) -> None:
        result = flatten([[3, 1], [2]])
        assert result == [3, 1, 2]


class TestUniqueOrdered:
    def test_no_duplicates(self) -> None:
        assert unique_ordered([1, 2, 3]) == [1, 2, 3]

    def test_removes_duplicates(self) -> None:
        assert unique_ordered([1, 2, 2, 3, 1]) == [1, 2, 3]

    def test_preserves_first_occurrence_order(self) -> None:
        result = unique_ordered([3, 1, 2, 1, 3])
        assert result == [3, 1, 2]

    def test_empty_list(self) -> None:
        assert unique_ordered([]) == []

    def test_strings(self) -> None:
        result = unique_ordered(["a", "b", "a", "c"])
        assert result == ["a", "b", "c"]

    @pytest.mark.parametrize(
        ("input_list", "expected"),
        [
            ([1, 1, 1], [1]),
            ([1, 2, 3], [1, 2, 3]),
            ([], []),
        ],
    )
    def test_parametrized(self, input_list: list, expected: list) -> None:
        assert unique_ordered(input_list) == expected


class TestFrequencyMap:
    def test_basic_counts(self) -> None:
        result = frequency_map(["a", "b", "a", "c", "a"])
        assert result["a"] == 3
        assert result["b"] == 1
        assert result["c"] == 1

    def test_sorted_by_count(self) -> None:
        result = frequency_map([1, 1, 1, 2, 2, 3])
        keys = list(result.keys())
        assert keys[0] == 1

    def test_empty_list(self) -> None:
        assert frequency_map([]) == {}

    def test_all_unique(self) -> None:
        result = frequency_map([1, 2, 3])
        assert all(v == 1 for v in result.values())

    @pytest.mark.parametrize(
        ("items", "most_common"),
        [
            (["x", "x", "y"], "x"),
            ([1, 2, 1, 1], 1),
        ],
    )
    def test_most_common_first(self, items: list, most_common: object) -> None:
        result = frequency_map(items)
        assert next(iter(result)) == most_common


class TestDeepMerge:
    def test_non_overlapping_keys(self) -> None:
        result = deep_merge({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_override_wins(self) -> None:
        result = deep_merge({"a": 1}, {"a": 2})
        assert result["a"] == 2

    def test_nested_merge(self) -> None:
        base = {"config": {"debug": False, "level": "INFO"}}
        override = {"config": {"debug": True}}
        result = deep_merge(base, override)
        assert result["config"]["debug"] is True
        assert result["config"]["level"] == "INFO"

    def test_does_not_mutate_base(self) -> None:
        base = {"a": 1}
        deep_merge(base, {"a": 2})
        assert base["a"] == 1

    def test_does_not_mutate_override(self) -> None:
        override = {"b": 2}
        deep_merge({"a": 1}, override)
        assert override == {"b": 2}

    def test_empty_override(self) -> None:
        base = {"a": 1, "b": 2}
        result = deep_merge(base, {})
        assert result == base

    def test_empty_base(self) -> None:
        override = {"x": 10}
        result = deep_merge({}, override)
        assert result == override


class TestPartition:
    def test_basic_partition(self) -> None:
        from app.collection_utils import partition
        evens, odds = partition([1, 2, 3, 4, 5], lambda x: x % 2 == 0)
        assert evens == [2, 4]
        assert odds == [1, 3, 5]

    def test_empty_list(self) -> None:
        from app.collection_utils import partition
        a, b = partition([], lambda x: x > 0)
        assert a == []
        assert b == []

    def test_all_true(self) -> None:
        from app.collection_utils import partition
        a, b = partition([1, 2, 3], lambda x: x > 0)
        assert a == [1, 2, 3]
        assert b == []

    def test_all_false(self) -> None:
        from app.collection_utils import partition
        a, b = partition([1, 2, 3], lambda x: x > 10)
        assert a == []
        assert b == [1, 2, 3]


class TestSlidingWindow:
    def test_basic(self) -> None:
        from app.collection_utils import sliding_window
        result = sliding_window([1, 2, 3, 4, 5], size=3)
        assert result == [[1, 2, 3], [2, 3, 4], [3, 4, 5]]

    def test_step_2(self) -> None:
        from app.collection_utils import sliding_window
        result = sliding_window([1, 2, 3, 4, 5], size=2, step=2)
        assert result == [[1, 2], [3, 4]]

    def test_window_larger_than_list(self) -> None:
        from app.collection_utils import sliding_window
        result = sliding_window([1, 2], size=5)
        assert result == []

    def test_invalid_size_raises(self) -> None:
        from app.collection_utils import sliding_window
        with pytest.raises(ValueError):
            sliding_window([1, 2, 3], size=0)

    def test_invalid_step_raises(self) -> None:
        from app.collection_utils import sliding_window
        with pytest.raises(ValueError):
            sliding_window([1, 2, 3], size=1, step=0)

    @pytest.mark.parametrize(
        ("items", "size", "expected_len"),
        [([1, 2, 3, 4], 2, 3), ([1, 2, 3], 3, 1), ([1, 2, 3, 4, 5], 1, 5)],
    )
    def test_window_counts(self, items: list, size: int, expected_len: int) -> None:
        from app.collection_utils import sliding_window
        assert len(sliding_window(items, size)) == expected_len


class TestGroupBy:
    def test_basic(self) -> None:
        from app.collection_utils import group_by
        result = group_by([1, 2, 3, 4], lambda x: x % 2)
        assert result[0] == [2, 4]
        assert result[1] == [1, 3]

    def test_empty_list(self) -> None:
        from app.collection_utils import group_by
        assert group_by([], lambda x: x) == {}

    def test_all_same_key(self) -> None:
        from app.collection_utils import group_by
        result = group_by(["a", "b", "c"], lambda x: "same")
        assert result == {"same": ["a", "b", "c"]}


class TestFlattenList:
    def test_basic_flatten(self) -> None:
        from app.collection_utils import flatten_list

        assert flatten_list([[1, 2], [3, 4]]) == [1, 2, 3, 4]

    def test_empty_sublists(self) -> None:
        from app.collection_utils import flatten_list

        assert flatten_list([[], [], []]) == []

    def test_single_sublist(self) -> None:
        from app.collection_utils import flatten_list

        assert flatten_list([[1, 2, 3]]) == [1, 2, 3]

    def test_empty_outer(self) -> None:
        from app.collection_utils import flatten_list

        assert flatten_list([]) == []

    def test_strings(self) -> None:
        from app.collection_utils import flatten_list

        assert flatten_list([["a", "b"], ["c"]]) == ["a", "b", "c"]


class TestUniqueSorted:
    def test_removes_duplicates(self) -> None:
        from app.collection_utils import unique_sorted

        assert unique_sorted([3, 1, 2, 1, 3]) == [1, 2, 3]

    def test_already_sorted(self) -> None:
        from app.collection_utils import unique_sorted

        assert unique_sorted([1, 2, 3]) == [1, 2, 3]

    def test_empty(self) -> None:
        from app.collection_utils import unique_sorted

        assert unique_sorted([]) == []

    def test_single_element(self) -> None:
        from app.collection_utils import unique_sorted

        assert unique_sorted([5]) == [5]

    def test_strings_sorted(self) -> None:
        from app.collection_utils import unique_sorted

        assert unique_sorted(["banana", "apple", "apple"]) == ["apple", "banana"]
