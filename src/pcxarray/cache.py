import os
from pathlib import Path
from joblib import Memory
import tempfile

def get_cache_dir():
    """
    Get or create a cache directory for pcxarray.

    The cache directory is platform-dependent and is created if it does not exist. 
    On Unix-like systems, it defaults to ~/.cache/pcxarray or $XDG_CACHE_HOME/pcxarray. 
    On Windows, it uses %LOCALAPPDATA%/pcxarray or the system temp directory as 
    fallback. The cache directory is used by joblib.Memory to store persistent function
    call results, particularly for expensive operations like downloading Census 
    shapefiles.

    Returns
    -------
    str
        Absolute path to the cache directory on disk.
    """
    # Try user cache directory first
    if os.name == 'nt':  # Windows
        cache_base = os.environ.get('LOCALAPPDATA', tempfile.gettempdir())
    else:  # Unix-like systems
        cache_base = os.environ.get('XDG_CACHE_HOME', 
                                   os.path.expanduser('~/.cache'))
    
    cache_dir = Path(cache_base) / 'pcxarray'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir)

# Create a global memory instance
_cache_dir = get_cache_dir()
memory = Memory(_cache_dir, verbose=0)

# Export the cache decorator
cache = memory.cache
"""
Decorator for caching function outputs to disk using joblib.Memory.

Use as @cache above a function to persist its results between runs. The cache is 
stored in a platform-appropriate directory and helps avoid repeated expensive operations 
like downloading data.

Returns
-------
function
    A decorator that caches the output of the decorated function to disk.
"""