from __future__ import annotations

import itertools
import os

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from glob import glob
from operator import itemgetter
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import cv2
import dateutil
import ffmpeg
import numpy as np
import xml.etree.ElementTree as ET

from camtrappy.errors import UnresolvedFoldersError
from camtrappy.interfaces import IProject


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
    __slots__ = ('path', 'location', 'sublocation', 'session', 'camera_no', 'metadata')
    path: str
    location: str
    sublocation: Optional[str]
    session: Optional[str]
    camera_no: Optional[str]

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


@dataclass
class Project(IProject):
    """Easy Data Parsing.

    Examples
    --------
    >>> p = Project('c/my_project', '{locations}/videodata/{sublocations}', 'mkv')
    >>> p.folderstructure
    'c/my_project/{location}/videodata/{sublocation}'
    >>> p.folder_hierarchy
    ['location', 'sublocation']
    >>> p.folder_indices
    [2, 4]
    >>> p.folder_structure_wildcards
    'c/my_project/*/videodata/*'
    >>> p.folders
    [
        {'location': 'location1', 'sublocation': 'sublocation1},
        {'location': 'location1', 'sublocation': 'sublocation2},
    ]
    >>> p.format_paths(p.folders)
    [
        'c/my_project/location1/videodata/sublocation1',
        'c/my_project/location1/videodata/sublocation2',
    ]
    >>> videos = p.get_videos(location='location1)  # returns videos for the given location
    >>> videos = p.get_videos()  # returns all videos of the project
    >>> next(videos)
    VideoFile(path='c/my_project/location1/videodata/sublocation1/video1.mkv',
    ...       location='location1', sublocation='sublocation1', camera_no=None)
    """
    projectfolder: str
    folderstructure: str
    videoformat: str

    folderstructure_placeholders: Dict[str, str] = field(init=False)
    _kwargs: Dict[str, str] = field(init=False)

    def __post_init__(self):
        # Add platform specific folder seperator at the end
        # Thus `glob` only ever returns folders
        self.projectfolder = str(Path(self.projectfolder)) + os.sep
        self.folderstructure = str(Path(self.projectfolder) / self.folderstructure) + os.sep

        # possible keyword arguments
        self._kwargs = dict(location=None, sublocation=None, session=None, camera_no=None)

        # placeholders for possible keyword arguments
        # can be used to define the folder structure of a project
        placeholders = {key: f'{{{key}}}' for key in self._kwargs.keys()}

        hierarchy = {}
        for key, value in placeholders.copy().items():
            folders = Path(self.folderstructure).parts
            try:
                index = folders.index(value)
            except:
                del placeholders[key]
            else:
                hierarchy[key] = index

        self._folderorder = dict(sorted(hierarchy.items(), key=lambda item: item[1]))
        self.folderstructure_placeholders = placeholders

    def _check_path_is_valid(self, folders: List[str], **kwargs):
        """Raise an error if a placeholder is missing."""
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        for folder in folders:
            try:
                kwargs.pop(folder)
            except:
                if len(kwargs) > 0:
                    raise UnresolvedFoldersError(folder, kwargs)
                else:
                    break
        return True

    @property
    def folder_hierarchy(self) -> List[str]:
        """The names of folder placeholders in order."""
        return list(self._folderorder.keys())

    @property
    def folder_indices(self) -> List[int]:
        """The indices of folder placeholders in order."""
        return list(self._folderorder.values())

    @property
    def folder_structure_wildcards(self) -> str:
        """The folderstructure with wildcards (*) instead of placeholders.

        Examples
        --------
        >>> Project.folderstructure
        c/data/{location}/
        >>> Project.folder_structure_wildcards
        c/data/*/
        """
        return self.folderstructure.format(
            **{folder: '*' for folder in self.folder_hierarchy}
            )

    @property
    def folders(self) -> List[Dict[str, str]]:
        """Return a list of dictionarys for all folder branches."""
        paths = glob(self.folder_structure_wildcards)
        args = [np.array(Path(path).parts)[self.folder_indices].tolist() for path in paths]
        kwargs_list = [dict(zip(self.folder_hierarchy, folder_names)) for folder_names in args]
        return kwargs_list

    def format_paths(self, kwargs_list: List[Dict[str, str]]) -> List[str]:
        """Generate path strings for all given folder branches."""
        return [self.folderstructure.format(**kwargs) for kwargs in kwargs_list]

    def get_folder(self, **kwargs) -> str:
        """Return path for given location, sublocation, session, camera_no."""
        self._check_path_is_valid(self.folder_hierarchy, **kwargs)

        # this is quite hacky...
        # '{location}'.format() will either replace it with a location name, or with '{location}'
        # because a formatting value must be provided for each formatting placeholder in a string
        arguments = {**self.folderstructure_placeholders, **kwargs}
        raw_path = self.folderstructure.format(**arguments)

        path = raw_path.split('{')[0]
        return path

    def get_videos(self, **kwargs: Dict[str, Optional[str]]) -> Generator[VideoFile, None, None]:
        """Return list of VideoFile objects.

        If keyword arguments are provided, they must match the folder hierarchy.

        Parameters
        ----------
        location : str, optional
        sublocation : str, optional
        session : str, optional
        camera_no : str, optional

        Returns
        -------
        generator : [VideoFile, None, None]
        """
        # make sure the given arguments match the folder hierarchy
        self._check_path_is_valid(self.folder_hierarchy, **kwargs)

        # get all folders matching the arguments
        # if `kwargs` is empty,
        #   all folders from the folder hierarchy will be scanned
        folders_list = [d for d in self.folders
                        if set(kwargs.items()) <= set(d.items())]

        paths = self.format_paths(folders_list)

        videos = [Path(path).rglob(f'*.{self.videoformat}') for path in paths]
        videos = itertools.chain.from_iterable(videos)

        for video_path in videos:
            kwargs = self._kwargs.copy()
            for index, folder in zip(self.folder_indices, self.folder_hierarchy):
                kwargs[folder] = video_path.parts[index]

            yield(VideoFile(str(video_path), **kwargs))



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