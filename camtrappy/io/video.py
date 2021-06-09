from pathlib import Path
from typing import List


def get_video_paths(folder: str, extensions: List[str]) -> List[str]:
    """Get a list of paths to all videos in a folder.
    """
    video_list = []
    for ext in extensions:
        lst = list(Path(folder).rglob(f"*.{ext}"))
        lst = [str(path_object) for path_object in lst]
        video_list += lst
    return video_list
