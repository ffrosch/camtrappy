from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class Project:

    projectfolder: str
    datafolder: str
    videoformats: List[str] = field(default_factory=list)
    _data: Dict[str, List[str]] = field(default_factory=defaultdict)

    def __post_init__(self):
        if len(self.videoformats) == 0:
            self.videoformats.append('mkv')
        self._data.default_factory = list
        self.get_locations()
        for format in self.videoformats:
            self.get_videos(format)

    def get_locations(self) -> List[str]:
        for location in Path(self.datafolder).iterdir():
            if location.is_dir():
                self._data[str(location.relative_to(self.datafolder))]

    def get_videos(self, format):
        locations = [Path(self.datafolder) / p for p in self._data.keys()]
        for location in locations:
            for video in Path(location).rglob(f'*.{format}'):
                self._data[location.name].append(str(video.relative_to(location)))

    # TODO: parse everything with full path and delegate path handling to
    # this "data" function
    @property
    def data(self, full_path=False):
        if not full_path:
            return dict(self._data)

    # TODO: get all possible information from the video path and store in lists