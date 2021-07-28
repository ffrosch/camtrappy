from __future__ import annotations

import time

from dataclasses import dataclass, field, InitVar
from datetime import date as Date, time as Time
from operator import attrgetter
from pydantic.dataclasses import dataclass as dataclass_pydantic
from sqlalchemy.orm import sessionmaker
from typing import Any, Dict, Generator, List, Tuple
from threading import Thread
from queue import Queue

import cv2

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
    transform: Any = None
    queue_size: int = 500

    def __post_init__(self, Session, location_id):
        super().__post_init__(Session, location_id)
        self.Q = Queue(maxsize=self.queue_size)

    def start(self, single=False):
        """Start the update-method within a thread."""
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
                print('Queue is full, waiting...')
                time.sleep(0.1)

            # grab frame if queue is not full
            else:
                self.skip_frames(self.skip_n_frames) # skip `self.skip_n_frames`

                grabbed, frame = self.stream.read() # grab next frame from file

                if grabbed: # if there was a frame to grab
                    # do transforms on it
                    if self.transform:
                        frame = self.transform(frame)

                    # add the frame to the queue
                    self.Q.put((video.id, frame))

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

    def skip_frames(self, n):
        for _ in range(n):
            _, _ = self.stream.read()

    def stop(self):
        # tell the thread to stop
        self.stopped = True
        # wait for the stream resources to be released
        self.thread.join()

    def show(self):
        self.start()

        # loop over frames from the video file stream
        while self.more():
            video_id, frame = self.read()
            # display the size of the queue on the frame
            cv2.putText(frame, f"Queue Size: {self.Q.qsize()}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # display the video id on the frame
            cv2.putText(frame, f"Video ID: {video_id}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # show the frame
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # do a bit of cleanup
        cv2.destroyAllWindows()
        self.stop()
        self.reset()

    def reset(self):
        self.idx = 0
