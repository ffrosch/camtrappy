from __future__ import annotations

import time

from dataclasses import dataclass
from datetime import date as Date, time as Time
from operator import attrgetter
from pydantic.dataclasses import dataclass as dataclass_pydantic
from typing import Any, Dict, Generator, List, Tuple
from threading import Thread
from queue import Queue

import cv2


@dataclass_pydantic
class Video:

    path: str
    id: int = None
    date: Date = None
    time: Time = None
    fps: int = None
    duration: int = None


    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Video:
        """Create a Video from a dict of video attributes.

        Dict structure is like {attribute_name: attribute}.

        Parameters
        ----------
        data : dict
        """
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Return Video attributes as a dict.

        Dict structure is {attribute_name: attribute}.
        """
        return dict(path=self.path,
                    id=self.id,
                    date=self.date,
                    time=self.time,
                    fps=self.fps,
                    duration=self.duration)


@dataclass_pydantic
class VideoList:

    videos: List[Video]
    id: int = None
    name: str = None
    chronological: bool = False


    def __post_init__(self):
        if self.chronological:
            self.videos.sort(key=attrgetter('date', 'time'))

    @classmethod
    def from_dict(cls,
                  data: List[Dict[str, Any]],
                  id=None,
                  name=None,
                  chronological=True) -> VideoList:
        """Create a VideoList from dictionaries.

        The input can be a dictionary of dictionaries
            {id: {attribute_name: attribute}}
        or a list of dictionaries
            [{attribute_name: attribute}]

        Parameters
        ----------
        data : list | dict
            list or dictionary of dictionaries
        """
        if type(data) == list:
            videolist = [Video.from_dict(d) for d in data]
        elif type(data) == dict:
            videolist = [Video.from_dict(dict(id=id, **attributes))
                         for id, attributes in data.items()]
        else:
            raise ValueError(f'`data` is of type: {type(data)}. '
                             '`data` must be a list of dictionaries '
                             'or a dictionary of dictionaries!')

        return cls(videolist, id, name, chronological)

    @classmethod
    def from_list(cls,
                  data: List[str],
                  id=None,
                  name=None,
                  chronological=False) -> VideoList:
        """Create a VideoList from a list of file paths.

        Parameters
        ----------
        data : list
            list of strings
        """
        return cls([Video(path) for path in data], id, name, chronological)

    def to_dict(self, list_of_dicts: bool = False) -> Dict[str, Any]:
        """Return a dict of videos.

        Dictionary of dictionaries structure is
            {id: {attribute_name: attribute}}
        List of dictionaries structure is
            [attribute_name: attribute]
        """
        if list_of_dicts:
            out = [v.to_dict() for v in self.videos]
        # for this to work `id` MUST NOT be None
        # it should be so that either all items or none
        # has an in. Thus we only check whether the first item
        # fullfills the condition.
        elif self.videos[0].id != None:
            # use walrus operator to create a nested dict in one expression
            # create flat dict attributes
            # pop id and nest the rest of the attributes below id
            # {id: {attribute_name: attribute}}
            out = {(attributes := v.to_dict()).pop('id'): dict(**attributes)
                    for v in self.videos}
        else:
            raise ValueError('Can only return a dict of dicts '
                             'if the videos have an id! '
                             'Try running with `list_of_dicts`=True instead.')
        return out

    def to_list(self) -> List[str]:
        """Return a list of video file paths."""
        return [v.path for v in self.videos]


@dataclass
class VideoLoader:
    """Adapted from imutils.video.filevideostream, with modifications.

    Parameters
    ----------
    videos : Generator
        must yield a tuple like (id, path_str)
    skip_frames : int, default = 9
        define how many frames to skip to speedup analysis
    transform : Any
        # TODO: define how to supply transforms
    queue_size : int, default = 500
        max number of frames in the queue

    Notes
    -------
    Use like this:
    videos = ((id, path) for id, path in enumerate(video_paths))
    skip_frames = 9
    vl = VideoLoader(videos, skip_frames)
    vl.start()

    while vl.more():
        grabbed, frame = vl.read()
        # do more stuff
    """

    videos: Generator[Tuple[int, str], None, None]
    skip_frames: int = 9
    transform: Any = None
    queue_size: int = 500

    def __post_init__(self):
        self.Q = Queue(maxsize=self.queue_size)
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.stopped: bool = False

    def start(self, single=False):
        """Start the update-method within a thread."""
        # TODO: implement arg=single:
        #     Restart object trackers after every single video.
        self.thread.start()
        return self

    def update(self):
        """Loop over the frames in a sequence of files."""

        video_id, video_path = next(self.videos, (None, None))
        self.stream = cv2.VideoCapture(video_path)

        # keep looping infinitely
        while True:
            # stop loop if variable is True
            if self.stopped:
                break

            # ensure there is room in the queue
            if self.Q.full():
                time.sleep(0.1) # Rest 100ms when the queue is full

            # grab frame if queue is not full
            else:
                # skip n frames
                for _ in range(self.skip_frames):
                    _, _ = self.stream.read()

                # read the next frame from the file
                grabbed, frame = self.stream.read()

                # if there were frames left in the stream
                # do transforms and put frame into queue
                if grabbed:
                    # any image preparations can be done here
                    # the thread will be idle a lot anyways
                    if self.transform:
                        frame = self.transform(frame)

                    # add the frame to the queue
                    self.Q.put((video_id, frame))

                # if 'grabbed' is False we have reached the end of the file
                else:
                    # stop file-access on exhausted file
                    self.stream.release()

                    # load next video path if available
                    video_id, video_path = next(self.videos, (None, None))

                    # keep streaming if there are more files
                    if video_path:
                        self.stream = cv2.VideoCapture(video_path)
                    else:
                        self.stopped = True

        # release the final stream
        self.stream.release()

    def read(self):
        """Return next frame in the queue."""
        return self.Q.get() # (video_id, frame)

    def more(self):
        """Return True if queue not empty or stream not stopped."""
        return self._running() or not self.stopped

    def _running(self):
        """Return True if there are still frames in the queue."""
        tries = 0
        # If queue is empty but stream still running, wait a moment
        while self.Q.empty() and not self.stopped and tries < 5:
            time.sleep(0.1)
            tries += 1
        return self.Q.empty()

    def stop(self):
        # tell the thread to stop
        self.stopped = True
        # wait for the stream resources to be released
        self.thread.join()
