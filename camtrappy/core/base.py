from __future__ import annotations
from collections import defaultdict, OrderedDict
from camtrappy.core.transforms import ITransform, Resize, TransformFactory

import time

from dataclasses import dataclass, field, InitVar
from datetime import date as Date, time as Time
from operator import attrgetter
from pydantic.dataclasses import dataclass as dataclass_pydantic
from sqlalchemy.orm import sessionmaker
from typing import Any, Dict, Generator, List, Tuple
from threading import Thread, settrace
from queue import Queue

import cv2
import numpy as np

from camtrappy.db.schema import Video, Location, Project


def location_id_by_name(Session, name):
    # TODO: account for possibility that a location can have the same name
    # but belong to a different project!
    with Session.begin() as session:
        q = session.query(Location)\
            .where(Location.name == name)\
            .one()
        id = q.id
    return id

def get_videos(Session, location_id):
    with Session.begin() as session:
        q = session.query(Video)\
            .where(Video.location_id == location_id)\
            .order_by(Video.date, Video.time)
    return q.all()

@dataclass
class VideoList:

    Session: InitVar[sessionmaker]
    location_id: InitVar[int]
    videos: List[Video] = field(init=False)
    idx: int = field(init=False)

    def __post_init__(self, Session, location_id):
        # idx is needed for __next__ and __iter__
        self.idx = 0
        self.videos = get_videos(Session, location_id)

    def __iter__(self):
        # makes "for x in VideoList" possible
        return self

    def __len__(self):
        return len(self.videos)

    def __next__(self):
        # required for __iter__ to work
        # can also be used for "next(VideoList)"
        try:
            item = self.videos[self.idx]
        except IndexError:
            self.idx = 0
            raise StopIteration()
        self.idx += 1
        return item

    def __nonzero__(self):
        # VideoList can be evaluated to True/False
        return bool(self.videos)

    def __repr__(self):
        return f'{self.videos}'

    def to_dict(self, list_of_dicts: bool = False) -> Dict[int, Dict[str, Any]]:
        """Return a dict of videos.

        Dictionary of dictionaries structure is
            {id: {attribute_name: attribute}}
        """
        # use walrus operator to create a nested dict in one expression
        # create flat dict attributes
        # pop id and nest the rest of the attributes below id
        # {id: {attribute_name: attribute}}
        return {(attributes := v.to_dict()).pop('id'): dict(**attributes)
                for v in self.videos}


@dataclass
class Object:

    id: int
    _data: OrderedDict = field(default_factory=OrderedDict)

    def add(self, video_id, frame_no, bbox):
        video = self._data.get(video_id, OrderedDict())

        bboxes = video.get('bboxes', list())
        bboxes.append(bbox)

        frames = video.get('frames', list())
        frames.append(frame_no)

    def bboxes(self, video_id):
        return self._data[video_id]['bboxes']

    def frames(self, video_id):
        return self._data[video_id]['frames']

    @property
    def last_bbox(self):
        video_id = next(reversed(self._data))
        return self.bboxes(video_id)[-1]

    @property
    def video_ids(self):
        return list(self._data.keys())


@dataclass
class Objects:

    next_object_id: int = 0
    finished_objects: OrderedDict[int, Object] = field(default_factory=OrderedDict)
    current_objects: OrderedDict[int, Object] = field(default_factory=OrderedDict)
    disappeared_objects: OrderedDict[int, Object] = field(default_factory=OrderedDict)

    def register(self, video_id, frame_no, bbox):
        id = self.next_object_id

        object = self.current_objects.get(id, Object(id))
        object.add(video_id, frame_no, bbox)

        self.disappeared_objects[id] = 0

        self.next_object_id += 1

    def deregister(self, object_id):
        self.finished_objects[object_id] = self.current_objects.pop(object_id)
        del self.disappeared_objects[object_id]

    def update(self, bboxes):
        pass

    def save_to_db(self):
        """Pop all items in self.finished_objects and push to database."""
        pass


@dataclass
class Frame:
    """Behaves like a list."""

    video_id: int
    frame_no: int
    _data: List[np.ndarray]

    @property
    def original(self):
        return self._data[0]

    @original.setter
    def original(self, frame):
        self._data[0] = frame

    @property
    def last(self):
        return self._data[-1]

    def __getattr__(self, method):
        return getattr(self._data, method)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        return self._data[item]

    def __setitem__(self, item, value):
        self._data[item] = value


@dataclass
class VideoLoader(VideoList):
    """Adapted from imutils.video.filevideostream, with modifications.

    Parameters
    ----------
    skip_frames : int, default = 9
        define how many frames to skip to speedup analysis
    transform : Any
        # TODO: define how to supply transforms
    queue_size : int, default = 500
        max number of frames in the queue
    """

    # TODO: implement arg=single:
    #     Restart object trackers after every single video.

    skip_n_frames: int = 9
    transforms: TransformFactory = None
    queue_size: int = 500

    def __post_init__(self, Session, location_id):
        super().__post_init__(Session, location_id)

    def start(self, single=False):
        """Start the update-method within a thread."""
        self.Q = Queue(maxsize=self.queue_size)
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.stopped: bool = False
        self.thread.start()

    def update(self):
        """Loop over the frames in a sequence of files."""
        # TODO: add frame number counting

        video = next(self, None)
        self.stream = cv2.VideoCapture(video.fullpath)

        while True:

            if self.stopped:
                break

            if self.Q.full():
                # print('Queue is full, waiting...')
                time.sleep(0.1)

            # grab frame if queue is not full
            else:
                self.skip_frames(self.skip_n_frames) # skip `self.skip_n_frames`

                grabbed, raw_frame = self.stream.read() # grab next frame from file

                if grabbed: # if there was a frame to grab
                    # get frame number
                    frame_no = self.stream.get(cv2.CAP_PROP_POS_FRAMES)
                    # convert to grayscale
                    raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)

                    frame = Frame(video.id, frame_no, [raw_frame])

                    # do transforms on it
                    if self.transforms:
                        frame = self.transforms.transform(frame)

                    # add the frame to the queue
                    self.Q.put(frame)

                else: # `grabbed` is False, end of file reached
                    self.stream.release() # stop file-access on exhausted file
                    video = next(self, None)
                    if video:
                        self.stream = cv2.VideoCapture(video.fullpath)
                    else:
                        self.stopped = True

        # release the final stream
        self.stream.release()

    def read(self):
        """Return next frame in the queue."""
        return self.Q.get(timeout=0.5) # (video_id, frame)

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

    def skip_frames(self, n):
        for _ in range(n):
            _, _ = self.stream.read()

    def stop(self):
        # tell the thread to stop
        self.stopped = True
        # wait for the stream resources to be released
        self.thread.join()

    def reset(self):
        self.idx = 0


@dataclass
class VideoPlayer:

    vl: VideoLoader

    def play(self,
             resize: Resize = True,
             transforms: TransformFactory = None,
             compare: bool = True,
             visitor=None):

        self.vl.start()

        # loop over frames from the video file stream
        while self.vl.more():
            try:
                frame = self.vl.read()
                out_frame = frame.last
            except:
                break

            if resize:
                frame.original = resize.transform(frame.original)

            if transforms:
                frame = transforms.transform(frame)

                if visitor:
                    visitor.apply(frame)

                if compare:
                    for i, t in enumerate(transforms.transforms, 1):
                        name = type(t).__name__
                        self.put_text(frame[i], Transform=name)

                    if len(frame) % 2 != 0:
                        frame.append(np.zeros(frame.original.shape, dtype=frame.original.dtype))
                    size = len(frame)
                    left, right = frame[:size//2], frame[size//2:]
                    out_frame = np.vstack((np.hstack(left), np.hstack(right)))
                else:
                    out_frame = frame.original

            self.put_text(out_frame,
                          QueueSize=self.vl.Q.qsize(),
                          VideoID=frame.video_id)
            cv2.imshow("Frame", out_frame)

            self.act_on_key()

        # do a bit of cleanup
        self.close()

    def close(self):
        cv2.destroyAllWindows()
        self.vl.stop()
        self.vl.reset()

    def put_text(self, frame, **kwargs):
        x, y = 10, 30
        for k, v in kwargs.items():
            cv2.putText(frame, f"{k}: {v}",
                (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
            y += 20

    def act_on_key(self):
        paused = False
        while True:
            k = cv2.waitKey(1)
            if k == 27 or k == ord('q'): # press 'ESC' or 'q' to close
                self.close()
                break
            if k == ord(' '): # pause/unpause with 'space'
                paused = True if not paused else False
            if not paused:
                break
            time.sleep(0.1)
