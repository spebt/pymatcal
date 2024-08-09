import importlib.resources
from .._schema import v1 as _schema
import json
# load the schema files and expose them as module attributes
schema_dir = importlib.resources.files(_schema)
__all__ = [schema_dir]
for schema_file in schema_dir.iterdir():
    schema_name = schema_file.stem
    with schema_file.open() as f:
        globals()[schema_name] = json.load(f)
        __all__.append(schema_name)
