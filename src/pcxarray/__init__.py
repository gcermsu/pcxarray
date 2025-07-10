from .query import pc_query, get_pc_collections
from .processing import prepare_timeseries, prepare_data, query_and_prepare

try:
    from importlib.metadata import version
    __version__ = version("pcxarray")
except ImportError:
    __version__ = "unknown"