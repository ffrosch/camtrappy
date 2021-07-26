from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from operator import itemgetter
from pathlib import Path
from typing import Dict, List

import cv2
import ffmpeg


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


def duration_from_files(lst):
    durations = []
    for video in lst:
        probe = ffmpeg.probe(video)
        duration = probe['format']['duration']
        durations.append(duration)
    return durations


def fps_from_files(lst):
    """Requires FULL PATH!"""
    fps_lst = []
    for file in lst:
        video = cv2.VideoCapture(file)
        fps = video.get(cv2.CAP_PROP_FPS)
        fps_lst.append(fps)
        video.release()
    return fps_lst


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
        self.get_fps()
        self.get_durations()
        # finally make sure the data is sorted by date and time for easier use!
        for location, data in self._data.items():
            zipped = zip(data['videos'], data['dates'], data['times'], data['fps'], data['durations'])
            videos, dates, times, fps, durations = zip(*sorted(zipped, key=itemgetter(1, 2))) # sort by date, time
            self._data[location]['videos'] = videos
            self._data[location]['dates'] = dates
            self._data[location]['times'] = times
            self._data[location]['fps'] = fps
            self._data[location]['durations'] = durations

    @property
    def locations(self):
        return list(self._data.keys())

    @property
    def location_paths(self):
        return [Path(self.datafolder) / p for p in self.locations]

    def video_paths(self, location):
        videos = self._data[location]['videos']
        return [str(Path(self.datafolder) / location / video) for video in videos]

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

    def get_fps(self):
        for location in self.locations:
            videos = self.video_paths(location)
            fps_lst = fps_from_files(videos)
            self._data[location]['fps'] = fps_lst

    def get_durations(self):
        for location in self.locations:
            videos = self.video_paths(location)
            durations = duration_from_files(videos)
            self._data[location]['durations'] = durations

    def data(self, location):
        d = self._data[location]

        paths = d['videos']
        dates = d['dates']
        times = d['times']
        fps = d['fps']
        durations = d['durations']

        return [dict(path=path, date=date, time=time, fps=fps, duration=duration)
                for path, date, time, fps, duration in zip(paths, dates, times, fps, durations)]
