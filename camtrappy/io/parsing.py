from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from operator import itemgetter
from pathlib import Path
from typing import Dict, List


def datetime_from_filenames(lst, part=-1, slice=(0,15), sep='_', clock_24h=True):
    dates, times = [], []
    for path in lst:
        date, time = Path(path).parts[part][slice[0]:slice[1]].split(sep)

        # year as "yyyy"
        if len(date) == 8:
            Y = '%Y'
        # year as "yy"
        else:
            Y = '%y'
        date = datetime.strptime(date, f'{Y}%m%d').date()
        dates.append(date)

        # hours as "00-24"
        if clock_24h:
            H = '%H'
        # hours as "00-12"
        else:
            H = '%I'
        time = datetime.strptime(time, f'{H}%M%S').time()
        times.append(time)
    return dates, times


def parse_locations(folder: str, restrict_to: List = None):
    locations = []
    for item in Path(folder).iterdir():
        if item.is_dir():
            name = item.name
            if restrict_to and name not in restrict_to:
                pass
            else:
                locations.append(name)
    return locations


def parse_videos(folder: str, format: str = 'mkv'):
    folder = Path(folder)
    print(f'Parsing folder: {folder.name}')

    videos = []
    for video in folder.rglob(f'*.{format}'):
        videos.append(str(video.relative_to(folder)))
    return videos


@dataclass
class ProjectParser:
    """Wrapper to easily parse data from a project.

    self._data: {
        location-folder-name: {
            videos: video-paths,
            dates: video-dates,
            times: video-times
        }
    }
    """

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
        self.get_datetimes()
        # finally make sure the data is sorted by date and time for easier use!
        for location, data in self._data.items():
            zipped = zip(data['videos'], data['dates'], data['times'])
            videos, dates, times = zip(*sorted(zipped, key=itemgetter(1, 2)))
            self._data[location]['videos'] = videos
            self._data[location]['dates'] = dates
            self._data[location]['times'] = times

    @property
    def location_paths(self):
        return [Path(self.datafolder) / p for p in self._data.keys()]

    def get_locations(self):
        locations = parse_locations(self.datafolder, self.restrict_to)
        for location in locations:
            self._data[location]

    def get_videos(self, format):
        for location in self.location_paths:
            videos = parse_videos(location, format)
            self._data[location.name]['videos'] = videos

    def get_datetimes(self, part=-1, slice=(0,15), sep='_', clock_24h=True):
        for location, lists in self._data.items():
            dates, times = datetime_from_filenames(lists['videos'],
                                                   part=part,
                                                   slice=slice,
                                                   sep=sep,
                                                   clock_24h=clock_24h)
            self._data[location]['dates'] = dates
            self._data[location]['times'] = times

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
