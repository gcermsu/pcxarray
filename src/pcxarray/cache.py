import os
from pathlib import Path
from joblib import Memory
import tempfile

def get_cache_dir():
    """Get the cache directory for pcxarray"""
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