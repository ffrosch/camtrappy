from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List


@dataclass
class Project:

    name: str
    projectfolder: str
    datafolder: str
    videoformats: List[str] = field(default_factory=list)
    restrict_to: List[str] = field(default_factory=list)
    _data: Dict[str, Dict[str, List[str]]] = field(default_factory=defaultdict)

    def __post_init__(self):
        if len(self.videoformats) == 0:
            self.videoformats.append('mkv')
        self._data.default_factory = lambda: defaultdict(list)
        self.get_locations()
        for format in self.videoformats:
            self.get_videos(format)
        self.get_datetime()

    def get_locations(self):
        for location in Path(self.datafolder).iterdir():
            if location.is_dir():
                name = location.name #str(location.relative_to(self.datafolder))
                if len(self.restrict_to) > 0 and name not in self.restrict_to:
                    pass
                else:
                    self._data[name]

    def get_videos(self, format):
        locations = [Path(self.datafolder) / p for p in self._data.keys()]
        for location in locations:
            print(f'Parsing location: {location.name}')
            for video in Path(location).rglob(f'*.{format}'):
                self._data[location.name]['videos'].append(str(video.relative_to(location)))

    def get_datetime(self, part=-1, slice=(0,15), sep='_', clock_24h=True):
        for location, lists in self._data.items():
            for video in lists['videos']:
                date, time = Path(video).parts[part][slice[0]:slice[1]].split(sep)

                # year as "yyyy"
                if len(date) == 8:
                    Y = '%Y'
                # year as "yy"
                else:
                    Y = '%y'
                date = datetime.strptime(date, f'{Y}%m%d').date()
                self._data[location]['dates'].append(date)

                # hours as "00-24"
                if clock_24h:
                    H = '%H'
                # hours as "00-12"
                else:
                    H = '%I'
                time = datetime.strptime(time, f'{H}%M%S').time()
                self._data[location]['times'].append(time)

    def data(self, full_path=False, sort_by=None, filter_by=None):
        """
        # data-dict
        {location_name: {
            videos: [],
            dates: [],
            times: []
        }

        sort_by : {None, location, date, time}
        filter_by: {None, location, date, time}
        """
        # TODO: implement sort_by
        # TODO: implement filter_by
        out_data = {}
        for location in self._data.keys():
            location_path = location
            video_paths = self._data[location]['videos']
            if full_path:
                location_path = str(Path(self.datafolder) / Path(location))
                video_paths = [str(Path(location_path) / Path(v)) for v in video_paths]
            out_data[location_path] = video_paths
        return out_data

    def data_flat(self):
        out_data = []
        for location in self._data.keys():
            videos = self._data[location]['videos']
            dates = self._data[location]['dates']
            times = self._data[location]['times']
            out_data.extend(zip([location] * len(videos), videos, dates, times))
        return out_data

    def locations(self):
        return list(self._data.keys())
