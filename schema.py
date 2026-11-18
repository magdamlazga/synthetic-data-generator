"""
The dataset schema: which columns the user wants, how they become a JSON schema for the model,
and how every generated row is checked before it is accepted.
"""

import datetime as dt
import re
from dataclasses import dataclass, field

COLUMN_TYPES = ["text", "integer", "number", "boolean", "category", "date"]

# "0..100", "0 to 100", "-5 - 10", "1.5–9.9"
RANGE_PATTERN = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*(?:\.\.|to|–|-)\s*(-?\d+(?:\.\d+)?)\s*$")


@dataclass
class Column:
    name: str
    type: str = "text"
    description: str = ""
    options: list[str] = field(default_factory=list)  # allowed values of a category column
    minimum: float | None = None                       # usual range of a number column
    maximum: float | None = None

    def describe(self):
        """One line for the prompt."""
        line = f"- {self.name} ({self.type}): {self.description}".rstrip(": ")
        if self.options:
            line += f" [allowed values: {', '.join(self.options)}]"
        if self.minimum is not None:
            line += f" [usual range: {_fmt(self.minimum)} to {_fmt(self.maximum)}]"
        if self.type == "date":
            line += " [format YYYY-MM-DD]"
        return line


@dataclass
class DatasetSpec:
    description: str
    columns: list[Column]
    name: str = "dataset"


def _fmt(x):
    return str(int(x)) if float(x).is_integer() else str(x)


# ---------- building the spec from the UI table ----------

def parse_columns(table_rows):
    """
    Turn the rows of the UI table [name, type, description, options or range] into Columns.
    Raises ValueError with a message the user can act on.
    """
    columns, seen = [], set()
    for i, raw in enumerate(table_rows, start=1):
        cells = [("" if c is None else str(c)).strip() for c in list(raw) + [""] * 4][:4]
        name, col_type, description, extra = cells
        if not name and not description:
            continue  # empty line in the table
        if not name:
            raise ValueError(f"Row {i}: the column needs a name.")
        if name in seen:
            raise ValueError(f"Column '{name}' appears twice.")
        seen.add(name)

        col_type = (col_type or "text").lower()
        if col_type not in COLUMN_TYPES:
            raise ValueError(f"Column '{name}': type must be one of {', '.join(COLUMN_TYPES)}.")

        column = Column(name=name, type=col_type, description=description)
        if col_type == "category":
            column.options = [o.strip() for o in extra.split(",") if o.strip()]
            if len(column.options) < 2:
                raise ValueError(f"Column '{name}': a category needs at least two comma-separated values.")
        elif col_type in ("integer", "number") and extra:
            match = RANGE_PATTERN.match(extra)
            if not match:
                raise ValueError(f"Column '{name}': write the range like 0..100.")
            low, high = sorted([float(match.group(1)), float(match.group(2))])
            column.minimum, column.maximum = low, high
        columns.append(column)

    if not columns:
        raise ValueError("Add at least one column.")
    return columns


# ---------- JSON schema for structured outputs ----------

def _column_schema(column):
    schema = {
        "text": {"type": "string"},
        "integer": {"type": "integer"},
        "number": {"type": "number"},
        "boolean": {"type": "boolean"},
        "category": {"type": "string", "enum": column.options},
        "date": {"type": "string"},
    }[column.type]
    description = column.description + (" (YYYY-MM-DD)" if column.type == "date" else "")
    if description.strip():
        schema["description"] = description.strip()
    return schema


def rows_json_schema(columns):
    """Schema for {"rows": [ {col: value, ...}, ... ]}, written to satisfy OpenAI's strict mode."""
    row = {
        "type": "object",
        "properties": {c.name: _column_schema(c) for c in columns},
        "required": [c.name for c in columns],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"rows": {"type": "array", "items": row}},
        "required": ["rows"],
        "additionalProperties": False,
    }


# ---------- validating what the model returned ----------

def _coerce(value, column):
    """Return (ok, value_or_error). Accepts small, unambiguous fixes like "42" for an integer."""
    t = column.type
    if t == "text":
        return (True, value) if isinstance(value, str) else (False, f"expected text, got {type(value).__name__}")

    if t in ("integer", "number"):
        if isinstance(value, bool):
            return False, "expected a number, got true/false"
        if isinstance(value, str):
            try:
                value = float(value.strip().replace(",", ""))
            except ValueError:
                return False, f"'{value}' is not a number"
        if not isinstance(value, (int, float)):
            return False, f"expected a number, got {type(value).__name__}"
        if t == "integer":
            if float(value).is_integer():
                return True, int(value)
            return False, f"{value} is not a whole number"
        return True, value

    if t == "boolean":
        if isinstance(value, bool):
            return True, value
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return True, value.strip().lower() == "true"
        return False, f"expected true/false, got {value!r}"

    if t == "category":
        if isinstance(value, str):
            for option in column.options:
                if value.strip().lower() == option.lower():
                    return True, option
        return False, f"{value!r} is not one of the allowed values"

    if t == "date":
        try:
            return True, dt.date.fromisoformat(str(value).strip()).isoformat()
        except ValueError:
            return False, f"{value!r} is not a YYYY-MM-DD date"

    return False, f"unknown type {t}"


def validate_row(row, columns):
    """Return (clean_row, None) if the row fits the schema, else (None, reason)."""
    if not isinstance(row, dict):
        return None, "row is not an object"
    clean = {}
    for column in columns:
        if column.name not in row:
            return None, f"missing '{column.name}'"
        ok, value = _coerce(row[column.name], column)
        if not ok:
            return None, f"'{column.name}': {value}"
        clean[column.name] = value
    return clean, None


def out_of_range(value, column):
    return (column.minimum is not None and isinstance(value, (int, float))
            and not column.minimum <= value <= column.maximum)
