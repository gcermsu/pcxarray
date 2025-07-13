import os
from pathlib import Path
from joblib import Memory
import tempfile
from functools import wraps

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

def cache(func=None, *cache_args, **cache_kwargs):
    """
    Cache the output of a function to disk using joblib.Memory.

    This decorator can be used in two ways:
    1. As ``@cache`` to cache the output of a function with default settings.
    2. As ``@cache(...options...)`` to pass arguments to joblib.Memory.cache, such as cache validation callbacks.

    The cache is stored in a platform-appropriate directory and helps avoid repeated expensive operations
    like downloading data.

    Parameters
    ----------
    func : callable, optional
        The function to be cached. If not provided, the decorator returns a wrapper that accepts a function.
    *cache_args
        Additional positional arguments passed to ``joblib.Memory.cache``.
    **cache_kwargs
        Additional keyword arguments passed to ``joblib.Memory.cache``.

    Returns
    -------
    function
        A decorator or decorated function that caches the output to disk.

    Examples
    --------
    Use with default settings::

        @cache
        def my_function(...):
            ...

    Use with custom cache options::

        @cache(cache_validation_callback=expires_after(minutes=15))
        def my_function(...):
            ...
    """
    
    if func is not None and callable(func):
        # Used as @cache
        @wraps(func)
        def wrapper(*args, **kwargs):
            return memory.cache(func)(*args, **kwargs)
        return wrapper
    else:
        # Used as @cache(...)
        def decorator(f):
            @wraps(f)
            def wrapper(*args, **kwargs):
                return memory.cache(f, *cache_args, **cache_kwargs)(*args, **kwargs)
            return wrapper
        return decorator
