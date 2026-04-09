import datetime

import polars as pl
import pytest

from nesteddotdict import DotDict


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

class TestInstantiation:
    def test_from_kwargs(self):
        d = DotDict(name="Alice", age=30)
        assert d.name == "Alice"
        assert d.age == 30

    def test_from_dict(self):
        d = DotDict({"name": "Alice", "age": 30})
        assert d.name == "Alice"
        assert d.age == 30

    def test_from_dotdict(self):
        original = DotDict(x=1, y=2)
        copy = DotDict(original)
        assert copy.x == 1
        assert copy.y == 2

    def test_kwargs_override_dict(self):
        d = DotDict({"x": 1}, x=99)
        assert d.x == 99

    def test_empty(self):
        d = DotDict()
        assert len(d) == 0

    def test_non_string_key_raises(self):
        with pytest.raises(TypeError):
            DotDict({1: "value"})

    def test_invalid_first_arg_raises(self):
        with pytest.raises(TypeError):
            DotDict(42)


# ---------------------------------------------------------------------------
# Attribute access
# ---------------------------------------------------------------------------

class TestAttributeAccess:
    def test_dot_get(self):
        d = DotDict(a=1)
        assert d.a == 1

    def test_dot_set(self):
        d = DotDict()
        d.a = 42
        assert d.a == 42

    def test_dot_delete(self):
        d = DotDict(a=1)
        del d.a
        with pytest.raises(AttributeError):
            _ = d.a

    def test_missing_attr_raises(self):
        d = DotDict()
        with pytest.raises(AttributeError):
            _ = d.missing

    def test_bracket_get(self):
        d = DotDict(a=1)
        assert d["a"] == 1

    def test_bracket_set(self):
        d = DotDict()
        d["a"] = 99
        assert d.a == 99

    def test_bracket_delete(self):
        d = DotDict(a=1)
        del d["a"]
        with pytest.raises(KeyError):
            _ = d["a"]

    def test_missing_key_raises(self):
        d = DotDict()
        with pytest.raises(KeyError):
            _ = d["missing"]

    def test_bracket_non_string_key_raises(self):
        d = DotDict()
        with pytest.raises(TypeError):
            _ = d[42]  # type: ignore[index]


# ---------------------------------------------------------------------------
# Dict-like interface
# ---------------------------------------------------------------------------

class TestDictInterface:
    def test_keys(self):
        d = DotDict(a=1, b=2)
        assert set(d.keys()) == {"a", "b"}

    def test_values(self):
        d = DotDict(a=1, b=2)
        assert set(d.values()) == {1, 2}

    def test_items(self):
        d = DotDict(a=1, b=2)
        assert dict(d.items()) == {"a": 1, "b": 2}

    def test_len(self):
        d = DotDict(a=1, b=2, c=3)
        assert len(d) == 3

    def test_iter(self):
        d = DotDict(a=1, b=2)
        assert set(d) == {"a", "b"}

    def test_repr(self):
        d = DotDict(a=1)
        assert "DotDict" in repr(d)
        assert "a=1" in repr(d)


# ---------------------------------------------------------------------------
# Nested structures
# ---------------------------------------------------------------------------

class TestNested:
    def test_nested_dotdict(self):
        d = DotDict(meta=DotDict(author="Romain", version="1.0"))
        assert d.meta.author == "Romain"
        assert d.meta.version == "1.0"

    def test_nested_plain_dict(self):
        d = DotDict(address={"city": "Paris", "zip": "75001"})
        assert d.address["city"] == "Paris"

    def test_to_dict_flat(self):
        d = DotDict(a=1, b=2)
        assert d.to_dict() == {"a": 1, "b": 2}

    def test_to_dict_nested_dotdict(self):
        d = DotDict(inner=DotDict(x=10))
        result = d.to_dict()
        assert result == {"inner": {"x": 10}}

    def test_to_dict_nested_list(self):
        d = DotDict(items=[1, 2, 3])
        assert d.to_dict() == {"items": [1, 2, 3]}

    def test_from_dict_classmethod(self):
        d = DotDict.from_dict({"a": 1, "b": {"c": 2}})
        assert d.a == 1
        assert d.b == {"c": 2}

    def test_from_dict_list(self):
        result = DotDict.from_dict([{"a": 1}, {"b": 2}])
        assert isinstance(result, list)
        assert result[0].a == 1


# ---------------------------------------------------------------------------
# JSON persistence
# ---------------------------------------------------------------------------

class TestJsonPersistence:
    def test_write_and_read_simple(self, tmp_path):
        d = DotDict(name="Alice", score=42)
        d.write_json_files(str(tmp_path))

        assert (tmp_path / "name.json").exists()
        assert (tmp_path / "score.json").exists()

        restored = DotDict.read_json_sources(str(tmp_path))
        assert restored.name == "Alice"
        assert restored.score == 42

    def test_write_and_read_nested_dict(self, tmp_path):
        d = DotDict(config={"debug": True, "level": 3})
        d.write_json_files(str(tmp_path))

        restored = DotDict.read_json_sources(str(tmp_path))
        assert restored.config == {"debug": True, "level": 3}

    def test_read_with_explicit_keys(self, tmp_path):
        d = DotDict(a=1, b=2, c=3)
        d.write_json_files(str(tmp_path))

        restored = DotDict.read_json_sources(str(tmp_path), keys=["a", "c"])
        assert restored.a == 1
        assert restored.c == 3
        with pytest.raises(AttributeError):
            _ = restored.b

    def test_read_missing_explicit_key_raises(self, tmp_path):
        d = DotDict(a=1)
        d.write_json_files(str(tmp_path))

        with pytest.raises(FileNotFoundError):
            DotDict.read_json_sources(str(tmp_path), keys=["a", "missing"])

    def test_read_empty_directory(self, tmp_path):
        d = DotDict.read_json_sources(str(tmp_path))
        assert len(d) == 0

    def test_read_nonexistent_directory_raises(self):
        with pytest.raises(FileNotFoundError):
            DotDict.read_json_sources("/nonexistent/path")

    def test_to_json_dict(self):
        d = DotDict(x=1, y=2)
        result = d.to_json_dict()
        assert "x.json" in result
        assert result["x.json"] == 1


# ---------------------------------------------------------------------------
# Polars DataFrame persistence
# ---------------------------------------------------------------------------

class TestPolarsPersistence:
    def test_write_and_read_dataframe(self, tmp_path):
        df = pl.DataFrame({"col_a": [1, 2, 3], "col_b": ["x", "y", "z"]})
        d = DotDict(df=df)
        d.write_json_files(str(tmp_path))

        restored = DotDict.read_json_sources(str(tmp_path))
        assert isinstance(restored.df, pl.DataFrame)
        assert restored.df.shape == (3, 2)
        assert restored.df["col_a"].to_list() == [1, 2, 3]

    def test_dataframe_schema_preserved(self, tmp_path):
        df = pl.DataFrame({"int_col": pl.Series([1, 2], dtype=pl.Int32)})
        d = DotDict(df=df)
        d.write_json_files(str(tmp_path))

        restored = DotDict.read_json_sources(str(tmp_path))
        assert restored.df.schema["int_col"] == pl.Int32

    def test_dataframe_with_dates(self, tmp_path):
        df = pl.DataFrame({
            "date_col": [datetime.date(2024, 1, 1), datetime.date(2024, 6, 15)]
        })
        d = DotDict(df=df)
        d.write_json_files(str(tmp_path))

        restored = DotDict.read_json_sources(str(tmp_path))
        assert restored.df["date_col"][0] == datetime.date(2024, 1, 1)

    def test_dataframe_with_datetime(self, tmp_path):
        df = pl.DataFrame({
            "ts": pl.Series(
                [datetime.datetime(2024, 1, 1, 12, 0)],
                dtype=pl.Datetime("us", None)
            )
        })
        d = DotDict(df=df)
        d.write_json_files(str(tmp_path))

        restored = DotDict.read_json_sources(str(tmp_path))
        assert isinstance(restored.df, pl.DataFrame)
        assert restored.df["ts"][0] == datetime.datetime(2024, 1, 1, 12, 0)

    def test_mixed_df_and_scalars(self, tmp_path):
        df = pl.DataFrame({"v": [10, 20]})
        d = DotDict(df=df, label="test", count=2)
        d.write_json_files(str(tmp_path))

        restored = DotDict.read_json_sources(str(tmp_path))
        assert isinstance(restored.df, pl.DataFrame)
        assert restored.label == "test"
        assert restored.count == 2
