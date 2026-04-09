# nesteddotdict

[![PyPI version](https://img.shields.io/pypi/v/nesteddotdict.svg)](https://pypi.org/project/nesteddotdict/)
[![Python](https://img.shields.io/pypi/pyversions/nesteddotdict.svg)](https://pypi.org/project/nesteddotdict/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A lightweight Python dictionary that supports **dot notation access**, **nested structures**, and **Polars DataFrame persistence** via JSON files.

## Features

- Dot notation and bracket notation interchangeably
- Nested `DotDict` instances with full dot-access
- Dict-like interface: `keys()`, `values()`, `items()`, `len()`, iteration
- Serialize and deserialize to/from JSON files (one file per key)
- Native support for **Polars DataFrames** — schema and data are preserved across read/write cycles
- Support for `datetime.date`, `datetime.datetime`, `datetime.time` serialization

## Installation

```bash
pip install nesteddotdict
```

## Quick start

```python
from nesteddotdict import DotDict

d = DotDict(name="Alice", age=30)
print(d.name)   # Alice
print(d["age"]) # 30

d.city = "Paris"
d["country"] = "France"
```

## Nested structures

`DotDict` instances can be nested to any depth:

```python
d = DotDict(
    user=DotDict(name="Alice", role="admin"),
    config={"debug": True, "level": 3},
)

print(d.user.name)        # Alice
print(d.config["debug"])  # True
```

Plain dicts assigned as values are kept as-is. Use `DotDict(...)` explicitly for nested dot access.

## Initialization patterns

```python
# From keyword arguments
d = DotDict(x=1, y=2)

# From a dict
d = DotDict({"x": 1, "y": 2})

# From another DotDict (shallow copy)
copy = DotDict(d)

# Mixed: dict + kwargs (kwargs take precedence)
d = DotDict({"x": 1, "y": 2}, y=99)  # y == 99
```

## Dict-like interface

```python
d = DotDict(a=1, b=2, c=3)

list(d.keys())    # ['a', 'b', 'c']
list(d.values())  # [1, 2, 3]
dict(d.items())   # {'a': 1, 'b': 2, 'c': 3}
len(d)            # 3

for key in d:
    print(key, d[key])
```

## Conversion

```python
d = DotDict(a=1, inner=DotDict(b=2))

d.to_dict()
# {'a': 1, 'inner': {'b': 2}}

DotDict.from_dict({"a": 1, "b": {"c": 2}})
# DotDict(a=1, b={'c': 2})
```

## JSON persistence

Each key is serialized to its own `.json` file in a directory:

```python
import polars as pl
from nesteddotdict import DotDict

d = DotDict(
    label="experiment_1",
    params={"lr": 0.01, "epochs": 100},
    results=pl.DataFrame({"metric": ["accuracy", "f1"], "value": [0.95, 0.93]}),
)

# Write to disk
d.write_json_files("./my_experiment")
# Creates:
#   my_experiment/label.json
#   my_experiment/params.json
#   my_experiment/results.json  ← DataFrame serialized with schema

# Read back
d2 = DotDict.read_json_sources("./my_experiment")

print(d2.label)              # experiment_1
print(d2.params["lr"])       # 0.01
print(type(d2.results))      # <class 'polars.dataframe.frame.DataFrame'>
print(d2.results)
# shape: (2, 2)
# ┌──────────┬───────┐
# │ metric   ┆ value │
# │ ---      ┆ ---   │
# │ str      ┆ f64   │
# ╞══════════╪═══════╡
# │ accuracy ┆ 0.95  │
# │ f1       ┆ 0.93  │
# └──────────┴───────┘
```

You can also load a subset of keys:

```python
d = DotDict.read_json_sources("./my_experiment", keys=["label", "results"])
```

## Polars support

Polars DataFrames are fully round-tripped through JSON — including schema (dtypes) and temporal types:

```python
import datetime
import polars as pl
from nesteddotdict import DotDict

df = pl.DataFrame({
    "date": [datetime.date(2024, 1, 1), datetime.date(2024, 6, 15)],
    "value": pl.Series([100, 200], dtype=pl.Int32),
})

d = DotDict(df=df)
d.write_json_files("/tmp/data")

restored = DotDict.read_json_sources("/tmp/data")
print(restored.df.schema)
# {'date': Date, 'value': Int32}
```

Supported Polars types: all numeric types, `Date`, `Datetime`, `Time`, `Duration`, `List`, `String`, `Boolean`.

## Running tests

```bash
uv run pytest
```
