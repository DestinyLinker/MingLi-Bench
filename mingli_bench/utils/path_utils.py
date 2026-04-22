"""
Path utilities for Chinese Fortune Telling Benchmark.
"""

from pathlib import Path
from typing import Optional, List


def find_file_in_hierarchy(filename: str, 
                          start_dir: Optional[Path] = None,
                          max_depth: int = 5) -> Optional[Path]:
    """
    Find a file in directory hierarchy.
    
    Args:
        filename: Name of the file to find
        start_dir: Starting directory (default: current working directory)
        max_depth: Maximum depth to search
        
    Returns:
        Path to the file if found, None otherwise
    """
    current_dir = start_dir or Path.cwd()
    
    for _ in range(max_depth):
        file_path = current_dir / filename
        if file_path.exists():
            return file_path
            
        # Check if we've reached the root
        if current_dir.parent == current_dir:
            break
            
        current_dir = current_dir.parent
    
    return None


def find_data_file(filename: str, 
                   data_dirs: Optional[List[str]] = None) -> Optional[Path]:
    """
    Find a data file in common data directories.
    
    Args:
        filename: Name of the data file
        data_dirs: List of directories to search (default: common data dirs)
        
    Returns:
        Path to the file if found, None otherwise
    """
    if data_dirs is None:
        data_dirs = [
            "data",
            "../data",
            "mingli_bench/data",
            Path(__file__).parent.parent / "data",
        ]
    
    for dir_path in data_dirs:
        path = Path(dir_path)
        if path.exists():
            file_path = path / filename
            if file_path.exists():
                return file_path
    
    # Try finding in hierarchy
    return find_file_in_hierarchy(f"data/{filename}")
