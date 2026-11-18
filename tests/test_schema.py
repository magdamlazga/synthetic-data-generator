import pytest

from schema import Column, parse_columns, rows_json_schema, validate_row


def test_parse_columns_reads_options_and_ranges():
    columns = parse_columns([
        ["message", "text", "What the user wrote", ""],
        ["mood", "category", "", "calm, angry , confused"],
        ["rating", "integer", "", "1..5"],
        ["balance", "number", "", "-100 to 250.5"],
        ["", "", "", ""],  # empty line is skipped
    ])
    assert [c.name for c in columns] == ["message", "mood", "rating", "balance"]
    assert columns[1].options == ["calm", "angry", "confused"]
    assert (columns[2].minimum, columns[2].maximum) == (1, 5)
    assert (columns[3].minimum, columns[3].maximum) == (-100, 250.5)


@pytest.mark.parametrize("rows, message", [
    ([["a", "colour", "", ""]], "type must be one of"),
    ([["a", "category", "", "only-one"]], "at least two"),
    ([["a", "integer", "", "lots"]], "range like"),
    ([["a", "text", "", ""], ["a", "text", "", ""]], "appears twice"),
    ([["", "", "", ""]], "at least one column"),
])
def test_parse_columns_explains_mistakes(rows, message):
    with pytest.raises(ValueError, match=message):
        parse_columns(rows)


def test_json_schema_is_strict_mode_compatible():
    schema = rows_json_schema([Column("mood", "category", options=["calm", "angry"]), Column("n", "integer")])
    row = schema["properties"]["rows"]["items"]
    assert row["required"] == ["mood", "n"]
    assert row["additionalProperties"] is False
    assert row["properties"]["mood"]["enum"] == ["calm", "angry"]


COLUMNS = [
    Column("name", "text"),
    Column("age", "integer", minimum=18, maximum=99),
    Column("score", "number"),
    Column("active", "boolean"),
    Column("tier", "category", options=["free", "pro"]),
    Column("joined", "date"),
]
GOOD = {"name": "Ana", "age": 30, "score": 4.5, "active": True, "tier": "pro", "joined": "2026-01-31"}


def test_valid_row_passes_and_extra_keys_are_dropped():
    row, error = validate_row({**GOOD, "unexpected": 1}, COLUMNS)
    assert error is None and row == GOOD


def test_small_unambiguous_fixes_are_accepted():
    row, error = validate_row({**GOOD, "age": "30", "score": "1,200.5", "active": "FALSE", "tier": "Pro"}, COLUMNS)
    assert error is None
    assert row["age"] == 30 and row["score"] == 1200.5 and row["active"] is False and row["tier"] == "pro"


@pytest.mark.parametrize("change, reason", [
    ({"age": 30.5}, "not a whole number"),
    ({"age": True}, "true/false"),
    ({"tier": "enterprise"}, "allowed values"),
    ({"joined": "31/01/2026"}, "YYYY-MM-DD"),
    ({"name": 12}, "expected text"),
])
def test_invalid_rows_are_rejected_with_a_reason(change, reason):
    row, error = validate_row({**GOOD, **change}, COLUMNS)
    assert row is None and reason in error


def test_missing_column_is_rejected():
    row, error = validate_row({k: v for k, v in GOOD.items() if k != "tier"}, COLUMNS)
    assert row is None and "missing 'tier'" in error
