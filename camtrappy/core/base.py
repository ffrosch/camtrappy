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

                grabbed, frame = self.stream.read() # grab next frame from file

                if grabbed: # if there was a frame to grab
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # do transforms on it
                    if self.transform:
                        frame = self.transform.transform(frame)

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

    def play(self, base_transforms=None, compare_transforms=None, visitor=None):
        self.vl.start()

        # loop over frames from the video file stream
        while self.vl.more():
            try:
                video_id, frame = self.vl.read()
            except:
                break

            if base_transforms:
                for t in base_transforms:
                    frame = t.transform(frame)

            if compare_transforms:
                frames = [frame]
                for t in compare_transforms:
                    frame = t.transform(frame)
                    frames.append(frame)

                if visitor:
                    visitor.detect(frames[-1])
                    visitor.draw(frames[0])

                if len(frames) % 2 != 0:
                    frames.append(np.zeros(frame.shape, dtype=frame.dtype))
                size = len(frames)
                left, right = frames[:size//2], frames[size//2:]
                frame1 = np.hstack(left)
                frame2 = np.hstack(right)
                frame = np.vstack((frame1, frame2))

            self.put_text(frame, video_id=video_id)
            cv2.imshow("Frame", frame)

            self.act_on_key()

        # do a bit of cleanup
        self.close()

    def close(self):
        cv2.destroyAllWindows()
        self.vl.stop()
        self.vl.reset()

    def put_text(self, frame, **kwargs):
        x, y = 10, 30
        # display the size of the queue on the frame
        cv2.putText(frame, f"Queue Size: {self.vl.Q.qsize()}",
            (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # display more info
        for k, v in kwargs.items():
            y += 20
            cv2.putText(frame, f"{k}: {v}",
                (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

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
