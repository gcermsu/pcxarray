from .query import pc_query, prepare_data, query_and_prepare

try:
    from importlib.metadata import version
    __version__ = version("pcxarray")
except ImportError:
    __version__ = "unknown"