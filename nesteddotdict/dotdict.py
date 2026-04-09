import datetime
import glob
import json
import os
import polars as pl
from typing import Any, Dict, List, Optional, Type, TypeVar

# Generic type for DotDict to improve type hinting
T = TypeVar('T', bound='DotDict')

class DotDict:
    """
    A dictionary-like class that allows dot notation access for its top-level string keys.
    Nested dictionaries are stored and returned as standard Python dicts.

    Can be initialized with:
    - Keyword arguments: DotDict(name="Alice", age=30)
    - A dictionary: DotDict({"name": "Alice", "age": 30})
    - A DotDict instance (for shallow copying attributes): DotDict(existing_instance)
    - A combination, where keyword arguments override dictionary keys.
    """

    def __init__(self, initial_data: Optional[Any] = None, **kwargs: Any):
        processed_data: Dict[str, Any] = {}

        if initial_data is not None:
            if isinstance(initial_data, self.__class__): # Handles DotDict and its subclasses
                # Shallow copy attributes from the source DotDict.
                for k, v in initial_data.__dict__.items():
                    processed_data[k] = v
            elif isinstance(initial_data, dict):
                for k, v in initial_data.items():
                    if not isinstance(k, str):
                        raise TypeError(
                            f"All top-level keys in the initial dictionary argument must be strings "
                            f"to be attributes. Found non-string key: {k!r} (type: {type(k).__name__})"
                        )
                    processed_data[k] = v # Values are taken as-is
            else:
                raise TypeError(
                    f"The first positional argument must be a dictionary or an instance of "
                    f"{self.__class__.__name__} (or its subclass), if provided. "
                    f"Got type: {type(initial_data).__name__}"
                )

        # Keyword arguments override any keys from initial_data.
        for k_kw, v_kw in kwargs.items():
            processed_data[k_kw] = v_kw # Values are taken as-is

        # Set attributes using self.key = value, which will invoke __setattr__.
        # With the modified _convert, values will be set as-is.
        for k_final, v_final in processed_data.items():
            setattr(self, k_final, v_final)

    def _convert(self, value: Any) -> Any:
        """
        Previously, this method could recursively convert dicts to DotDicts.
        This recursive conversion for DotDict types is now removed.
        The method currently acts as a pass-through. It's retained structurally
        in case simple, non-recursive per-value transformations are needed on assignment in the future.
        """
        return value

    def __setattr__(self, key: str, value: Any):
        """
        Sets an attribute. Ensures attribute keys are strings.
        The assigned value is processed by _convert (which is now pass-through).
        """
        if not isinstance(key, str):
            raise TypeError(f"Attribute keys for {self.__class__.__name__} must be strings. Got key: {key!r} (type: {type(key).__name__})")

        # self._convert(value) is now effectively just 'value'.
        object.__setattr__(self, key, self._convert(value))

    def __getattr__(self, key: str) -> Any:
        """Gets an attribute using dot notation."""
        try:
            return self.__dict__[key]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'") from None

    def __getitem__(self, key: str) -> Any:
        """Gets an item using bracket notation."""
        if not isinstance(key, str):
            raise TypeError(f"Keys for {self.__class__.__name__} items must be strings. Got: {key!r}")
        try:
            return self.__dict__[key]
        except KeyError:
            raise KeyError(f"Key '{key}' not found in {self.__class__.__name__}") from None

    def __setitem__(self, key: str, value: Any):
        """Sets an item using bracket notation. Delegates to __setattr__."""
        if not isinstance(key, str):
            raise TypeError(f"Keys for {self.__class__.__name__} items must be strings. Got: {key!r}")
        setattr(self, key, value)

    def __delattr__(self, key: str):
        """Deletes an attribute using dot notation."""
        try:
            object.__delattr__(self, key)
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'") from None

    def __delitem__(self, key: str):
        """Deletes an item using bracket notation."""
        if not isinstance(key, str):
            raise TypeError(f"Keys for {self.__class__.__name__} items must be strings. Got: {key!r}")
        try:
            del self.__dict__[key]
        except KeyError:
            raise KeyError(f"Key '{key}' not found in {self.__class__.__name__}") from None

    def to_dict(self) -> dict:
        """
        Recursively converts this DotDict and any manually nested DotDict instances
        back to standard Python dictionaries. Standard dicts and lists contained as values
        are also part of this recursive conversion to ensure a pure Python dict structure.
        """
        return self._to_dict_value_converter(self.__dict__)

    def _to_dict_value_converter(self, value: Any) -> Any:
        """Helper for to_dict to recursively convert values to standard Python types."""
        if isinstance(value, self.__class__): # Handles current DotDict or manually nested ones
            return {k: self._to_dict_value_converter(v) for k, v in value.__dict__.items()}
        elif isinstance(value, dict): # For standard dicts stored as values
            return {k: self._to_dict_value_converter(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._to_dict_value_converter(item) for item in value]
        else:
            return value

    @classmethod
    def from_dict(cls: Type[T], d_obj: Dict[Any, Any]) -> T | Dict[Any, Any] | List[Any] | Any:
        """
        Converts a dictionary into an instance of this class (DotDict).
        Nested dictionaries within d_obj will remain as standard Python dicts
        (not converted to nested DotDicts).
        The recursion here is for traversing the input d_obj structure if it contains
        further dicts/lists that might match the `from_dict` processing logic for their values.
        """
        if isinstance(d_obj, dict):
            if all(isinstance(k, str) for k in d_obj.keys()):
                # cls(d_obj) creates a DotDict. Due to the modified _convert,
                # if d_obj's values are dicts, they will be stored as plain dicts.
                return cls(d_obj)
            else: # Input dict has non-string keys
                # Recursively call from_dict on values; container remains a dict.
                return {k: cls.from_dict(v) for k, v in d_obj.items()} # type: ignore
        elif isinstance(d_obj, list):
            return [cls.from_dict(item) for item in d_obj] # type: ignore
        else:
            return d_obj

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __repr__(self):
        items_repr = ", ".join(f"{k}={v!r}" for k, v in sorted(self.__dict__.items()))
        return f"{self.__class__.__name__}({items_repr})"

    ##################
    # IO methods
    ##################
    @staticmethod
    def _dtype_to_str(dtype: pl.DataType) -> str:
        if isinstance(dtype, pl.Datetime):
            return f"Datetime[{dtype.time_unit}, {dtype.time_zone}]"
        if isinstance(dtype, pl.Duration):
             return f"Duration[{dtype.time_unit}]"
        if isinstance(dtype, pl.List):
            return f"List[{DotDict._dtype_to_str(dtype.inner)}]" # type: ignore
        return str(dtype)

    @staticmethod
    def _str_to_dtype(dtype_str: str) -> pl.DataType:
        if dtype_str.startswith("Datetime["):
            try:
                time_unit_str, time_zone_str = dtype_str[9:-1].split(", ", 1)
                time_zone: Optional[str] = None if time_zone_str == "None" else time_zone_str.strip("'")
                return pl.Datetime(time_unit=time_unit_str, time_zone=time_zone) # type: ignore
            except ValueError as e:
                raise ValueError(f"Error parsing Datetime string: {dtype_str}. Details: {e}")
        if dtype_str.startswith("Duration["):
            time_unit_str = dtype_str[9:-1]
            return pl.Duration(time_unit=time_unit_str) # type: ignore
        if dtype_str.startswith("List[") and dtype_str.endswith("]"):
            inner_dtype_str = dtype_str[5:-1]
            inner_dtype = DotDict._str_to_dtype(inner_dtype_str)
            return pl.List(inner_dtype)
        try:
            return getattr(pl, dtype_str)
        except AttributeError:
            raise ValueError(f"Unsupported or unknown Polars dtype string: {dtype_str}")


    @staticmethod
    def _polars_schema_to_json_schema(pl_schema: Dict[str, pl.DataType]) -> Dict[str, str]:
        return {col_name: DotDict._dtype_to_str(pl_type) for col_name, pl_type in pl_schema.items()}

    @staticmethod
    def _json_schema_to_polars_schema(json_schema: Dict[str, str]) -> Dict[str, pl.DataType]:
        try:
            return {col_name: DotDict._str_to_dtype(str_dtype)
                    for col_name, str_dtype in json_schema.items()}
        except Exception as e:
            raise ValueError(
                f"Error converting JSON schema to Polars schema: {e}. "
                f"Problematic schema part: {json_schema}"
            )

    @staticmethod
    def _cast_data_to_python_types(data: Dict[str, List[Any]], schema: Dict[str, pl.DataType]) -> Dict[str, List[Any]]:
        for col, dtype in schema.items():
            if col not in data:
                continue
            if dtype == pl.Date:
                data[col] = [datetime.date.fromisoformat(elem) if elem is not None else None for elem in data[col]]
            elif isinstance(dtype, pl.Datetime):
                data[col] = [datetime.datetime.fromisoformat(elem) if elem is not None else None for elem in data[col]]
            elif dtype == pl.Time:
                 data[col] = [datetime.time.fromisoformat(elem) if elem is not None else None for elem in data[col]]
            elif isinstance(dtype, pl.Duration): # Assuming duration stored as total microseconds from JSON
                 data[col] = [datetime.timedelta(microseconds=elem) if elem is not None else None for elem in data[col]]
        return data

    def _convert_value_to_json_repr(self, value_to_convert: Any) -> Any:
        """
        Recursively converts a value to its JSON-serializable representation.
        This recursion is for data structure traversal, not for DotDict type conversion.
        """
        if isinstance(value_to_convert, pl.DataFrame):
            df = value_to_convert
            d_data_by_col: Dict[str, List[Any]] = {}
            for col in df.columns:
                col_series = df[col]
                dtype = col_series.dtype
                if dtype == pl.Date:
                    d_data_by_col[col] = [d.isoformat() if d is not None else None for d in col_series.to_list()]
                elif isinstance(dtype, pl.Datetime):
                    d_data_by_col[col] = [dt.isoformat() if dt is not None else None for dt in col_series.to_list()]
                elif dtype == pl.Time:
                    d_data_by_col[col] = [t.isoformat() if t is not None else None for t in col_series.to_list()]
                elif isinstance(dtype, pl.Duration): # Store duration as total microseconds
                    d_data_by_col[col] = [int(td.total_seconds() * 1_000_000) if td is not None else None for td in col_series.to_list()]
                else:
                    d_data_by_col[col] = col_series.to_list()
            return {
                "__dataframe__": True,
                "schema": self._polars_schema_to_json_schema(df.schema),
                "data": d_data_by_col
            }
        # Check for DotDict instance (e.g., self, or if one was manually assigned)
        elif isinstance(value_to_convert, self.__class__):
            return {k: self._convert_value_to_json_repr(v) for k, v in value_to_convert.__dict__.items()}
        elif isinstance(value_to_convert, dict): # Standard dict
            return {k: self._convert_value_to_json_repr(v) for k, v in value_to_convert.items()}
        elif isinstance(value_to_convert, list):
            return [self._convert_value_to_json_repr(item) for item in value_to_convert]
        elif isinstance(value_to_convert, (datetime.datetime, datetime.date, datetime.time)):
            return value_to_convert.isoformat()
        elif isinstance(value_to_convert, datetime.timedelta): # For Polars Duration, already handled. This is for standalone timedelta.
            return value_to_convert.total_seconds() * 1_000_000 # Store as microseconds
        return value_to_convert

    def to_json_dict(self) -> Dict[str, Any]:
        """
        Converts the DotDict instance into a JSON-serializable Python dictionary.
        """
        return {f"{k}.json": self._convert_value_to_json_repr(v) for k, v in self.__dict__.items()}

    def write_json_files(self, dest_path: str):
        """
        Writes each top-level item of the DotDict to a separate JSON file.
        """
        if not os.path.exists(dest_path):
            os.makedirs(dest_path, exist_ok=True)
        elif not os.path.isdir(dest_path):
            raise NotADirectoryError(f"Destination path '{dest_path}' exists but is not a directory.")

        for key, value in self.__dict__.items():
            file_path = os.path.join(dest_path, f"{key}.json")
            json_repr_for_value = self._convert_value_to_json_repr(value)
            try:
                with open(file_path, 'w') as f:
                    json.dump(json_repr_for_value, f, indent=2)
            except TypeError as e:
                raise TypeError(f"Error serializing item '{key}' to JSON: {e}. "
                                f"Problematic value type: {type(json_repr_for_value)}. Value snippet: {str(json_repr_for_value)[:200]}")
            except IOError as e:
                raise IOError(f"Error writing JSON file for key '{key}' to {file_path}: {e}")

    @classmethod
    def _json_to_internal_type_converter(cls: Type[T], json_value: Any) -> Any:
        """
        Recursively converts a loaded JSON value. Reconstructs Polars DataFrames.
        Other JSON objects (dicts) are returned as standard Python dicts.
        """
        if isinstance(json_value, dict):
            if json_value.get("__dataframe__") is True:
                schema_json = json_value.get("schema")
                raw_data = json_value.get("data")
                if schema_json is None or raw_data is None:
                    raise ValueError("Invalid Polars DataFrame JSON: 'schema' or 'data' field missing.")
                try:
                    polars_schema = cls._json_schema_to_polars_schema(schema_json)
                    data_casted = cls._cast_data_to_python_types(raw_data, polars_schema)
                    return pl.DataFrame(data=data_casted, schema=polars_schema) # type: ignore
                except Exception as e:
                    raise ValueError(f"Failed to reconstruct Polars DataFrame from JSON: {e}\n"
                                     f"Schema: {schema_json}\nData (keys): {list(raw_data.keys()) if isinstance(raw_data, dict) else raw_data}")
            else:
                # Recursively process values, but return a standard Python dict.
                return {k: cls._json_to_internal_type_converter(v) for k, v in json_value.items()}
        elif isinstance(json_value, list):
            return [cls._json_to_internal_type_converter(item) for item in json_value]
        elif isinstance(json_value, str): # Attempt to parse common ISO datetime/date/time strings
            try:
                # Refined heuristic for ISO-like strings
                if len(json_value) >= 10 and ('T' in json_value or json_value.count('-') >= 2 or json_value.count(':') >= 2):
                    try: return datetime.datetime.fromisoformat(json_value)
                    except ValueError:
                        try: return datetime.date.fromisoformat(json_value)
                        except ValueError:
                            try: return datetime.time.fromisoformat(json_value)
                            except ValueError: pass # Not a parsable ISO string for these types
            except Exception: pass # Ignore parsing errors, return as string
        # Add handling for numbers that might represent timedeltas (microseconds)
        elif isinstance(json_value, (int, float)) and not isinstance(json_value, bool):
             # This is a generic heuristic. If you store other large numbers, this might misinterpret.
             # Consider a specific marker if you store standalone timedelta numbers.
             pass # Could potentially try to convert to timedelta if a convention is established.
        return json_value

    @classmethod
    def read_json_sources(cls: Type[T], src_path: str, keys: Optional[List[str]] = None) -> T:
        """
        Reads JSON files from a source directory and constructs a new DotDict instance.
        Nested JSON objects will be loaded as standard Python dicts within this DotDict.
        """
        if not os.path.isdir(src_path):
            raise FileNotFoundError(f"Source directory '{src_path}' not found or is not a directory.")

        actual_keys = keys
        if actual_keys is None:
            json_files = glob.glob(os.path.join(src_path, "*.json"))
            if not json_files:
                return cls()
            actual_keys = sorted([os.path.splitext(os.path.basename(f))[0] for f in json_files])

        if not actual_keys and keys is not None and len(keys) == 0:
            return cls()

        data_for_dotdict: Dict[str, Any] = {}
        for key_name in actual_keys: # Renamed 'key' to 'key_name'
            file_path = os.path.join(src_path, f"{key_name}.json")
            if not os.path.exists(file_path):
                if keys is not None:
                    raise FileNotFoundError(f"JSON file for explicit key '{key_name}' not found at {file_path}")
                continue

            try:
                with open(file_path, 'r') as f:
                    raw_json_obj = json.load(f)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(f"Error decoding JSON from file {file_path}: {e.msg}", e.doc, e.pos) from e
            except IOError as e:
                raise IOError(f"Error reading JSON file {file_path}: {e}")

            processed_value = cls._json_to_internal_type_converter(raw_json_obj)
            data_for_dotdict[key_name] = processed_value

        # Creates a single, top-level DotDict. Nested structures are plain dicts/lists.
        return cls(**data_for_dotdict)