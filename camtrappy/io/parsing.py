from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from operator import itemgetter
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import dateutil
import ffmpeg
import xml.etree.ElementTree as ET


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


class Metadata:
    """Container that holds essential information about a video."""
    __slots__ = ('starttime', 'stoptime', 'fps', 'width', 'height', 'status')
    aliases = {
        'start_time': 'starttime',
        'StartTime': 'starttime',
        'stop_time': 'stoptime',
        'StopTime': 'stoptime',
        'end_time': 'stoptime',
        'endtime': 'stoptime',
        'EndTime': 'endtime',
        'frames_per_second': 'fps',
        'FramesPerSecond': 'fps',
    }

    def __init__(
            self,
            starttime: Optional[datetime] = None,
            stoptime: Optional[datetime] = None,
            fps: Optional[float] = None,
            width: Optional[int] = None,
            height: Optional[int] = None,
            status: Optional[str] = None
        ):

        self.starttime = starttime
        self.stoptime = stoptime
        self.fps = fps
        self.width = width
        self.height = height
        self.status = status

    def __repr__(self):
        return f'Metadata({self.starttime}, {self.stoptime}, {self.fps}, {self.width}, {self.height}, {self.status})'

    def __setattr__(self, name: str, value: Any) -> None:
        """SET the correct attribute with pre-defined aliases.

        Parameters
        ----------
        name : str
            The alias. See class-attribute `aliases` for supported aliases.
        value : Any
            The new value for the alias/attribute.

        Returns
        -------
        None
        """
        name = self.aliases.get(name, name)
        object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> Any:
        """GET the correct attribute with pre-defined aliases.

        Parameters
        ----------
        name : str
            The alias. See class-attribute `aliases` for supported aliases.

        Returns
        -------
        attribute : Any
            Returns the attribute that corresponds to the alias.
        """
        if name == 'aliases':
            raise AttributeError  # http://nedbatchelder.com/blog/201010/surprising_getattr_recursion.html
        name = self.aliases.get(name, name)
        return object.__getattribute__(self, name)

    def __add__(self, other: Metadata) -> Metadata:
        """Merge this instance of `Metadata` with another instance of `Metadata`.

        Attributes of the first (this) instance have priority!
        Attributes from `other` are only merged, if the corresponding attribute
        of this instance is `None`.

        Parameters
        ----------
        other : Metadata
            Another instance of the Metadata-class.

        Returns
        -------
        object : Metadata
            Returns a new instance of Metadata.
        """
        starttime = self.starttime if self.starttime is not None else other.starttime
        stoptime = self.stoptime if self.stoptime is not None else other.stoptime
        fps = self.fps if self.fps is not None else other.fps
        width = self.width if self.width is not None else other.width
        height = self.height if self.height is not None else other.height
        status = self.status if self.status is not None else other.status

        return Metadata(starttime=starttime, stoptime=stoptime, fps=fps,
                        width=width, height=height, status=status)

    def to_dict(self) -> Dict:
        """Return instance attributes as a dictionary."""
        slots = self.__slots__
        return {slot: getattr(self, slot) for slot in slots}

    @classmethod
    def from_dict(cls, data: Dict) -> Metadata:
        """Create a `Metadata` object from a dictionary.

        Parameters
        ----------
        data : Dict

        Returns
        -------
        object : Metadata
        """
        meta = cls()
        for attribute, value in data.items():
            try:
                setattr(meta, attribute, value)
            except AttributeError as e:
                print(e)
                print(f'Possible attributes are: {meta.__slots__}.')
                print(f'Possible aliases are: {meta.aliases.keys()}.')
        return meta


@dataclass
class VideoFile:
    __slots__ = ('path', 'location', 'session', 'sublocation', 'metadata')
    path: str
    location: str
    session: Optional[str]
    sublocation: Optional[str]

    def __post_init__(self) -> None:
        self.metadata: Metadata = None

    def get_metadata(self, priority: str = 'sidecar') -> Metadata:
        """Get Metadata for the video file.

        Notes
        -----
        Multiple available sources are used.
        Supported sources:
        - videofile itself (opened and read with ffmpeg)
        - xml-sidecarfile

        Parameters
        ----------
        priority : str, default='sidecar', {'sidecar', 'ffmpeg'}
            Which source becomes priority in case of conflicting attributes.

        Returns
        -------
        object : Metadata
        """
        if self.metadata is None:
            sidecar = metadata_from_sidecarfile(self.path)
            ffmpeg = metadata_with_ffmpeg(self.path)

            if priority == 'sidecar':
                meta = sidecar + ffmpeg
            if priority == 'ffmpeg':
                meta = ffmpeg + sidecar

        self.metadata = meta
        return meta


def metadata_from_sidecarfile(videopath: str, type: str = 'xml') -> Metadata:
    """Get available essential metadata from a sidecar-file.

    Parameters
    ----------
    videopath : str
        Path to the videofile.
    type : str, default='xml', {'xml'}
        Type of sidecar file.

    Notes
    -----
    Only *.xml files are supported right now.
    """
    def xml_parser(path):
        tree = ET.parse(path)
        root = tree.getroot()
        tags = {}
        for child in root:
            tag = child.tag.lower()
            value = child.text
            tags[tag] = value
        return tags

    parsers = {
        'xml': xml_parser,
    }

    path = Path(videopath).with_suffix(f'.{type}')
    if type not in parsers:
        raise ValueError(f'Sidecar files of type {type} are not supported.')
    if not path.is_file():
        return

    meta = Metadata()
    tags = parsers[type](path)
    for tag, value in tags.items():
        if hasattr(meta, tag): setattr(meta, tag, value)

    return meta


def metadata_with_ffmpeg(path: str) -> Metadata:
    """Get essential Metadata from a videofile by using ffmpeg."""
    filemeta = ffmpeg.probe(path)
    meta = Metadata()

    if 'streams' in filemeta:
        stream = filemeta['streams'][0]

        meta.width = int(stream.get('width'))
        meta.height = int(stream.get('height'))
        meta.fps = int(stream.get('avg_frame_rate').split('/')[0])

    if 'format' in filemeta:
        format = filemeta['format']

        _datetime_string = format.get('tags', {}).get('creation_time')
        _duration = float(format.get('duration'))

        meta.starttime = dateutil.parser.isoparse(_datetime_string)
        meta.stoptime = meta.starttime + timedelta(seconds=_duration)

    return meta


# class Seperator(Enum):
#     EMPTY = ''
#     HYPHEN = '-'
#     UNDERSCORE = '_'


# class DatePattern(Enum):
#     YYYYMMDD = '%Y%m%d'


# class TimePattern(Enum):
#     HHmmss = '%H%M%S'


# @dataclass
# class DatetimeParser:
#     __slots__ = ('date', 'time', 'seperator', 'from_beginning')
#     date: DatePattern
#     time: TimePattern
#     seperator: Seperator
#     from_beginning: bool  # True = datetime is at the beginning of the filename, False = it's at the end

#     def __str__(self) -> str:
#         now = datetime.now()
#         pattern = self._format()
#         string = now.strftime(pattern)
#         return string

#     def _format(self) -> str:
#         pattern = self.date.value + self.seperator.value + self.time.value
#         return pattern

#     def demo(self):
#         return self.__str__()

#     def str_to_datetime(self, string) -> datetime:
#         pattern = self._format()
#         dtime = datetime.strptime(string, pattern)
#         return dtime


# def starttime_from_videofile(path: str,
#                              datetime_parser: DatetimeParser,
#                              seperator: Seperator) -> datetime:

#     # get datetime
#     filename = Path(path).name
#     parts = filename.split(seperator.value)
#     datetime_seperator = datetime_parser.seperator

#     datetime_is_continuous = (datetime_seperator == Seperator.EMPTY) or (datetime_seperator != seperator)
#     if datetime_parser.from_beginning:
#         if datetime_is_continuous:
#             datetime_string = parts[0]
#         else:
#             datetime_string = datetime_seperator.value.join((parts[0], parts[1]))
#     else:
#         if datetime_is_continuous:
#             datetime_string = parts[-1]
#         else:
#             datetime_string = datetime_seperator.value.join((parts[-2], parts[-1]))

#     starttime = datetime_parser.str_to_datetime(datetime_string)
#     return starttime