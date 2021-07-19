import time

from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Tuple
from threading import Thread
from queue import Queue

import cv2


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

    def start(self):
        """Start the update-method within a thread."""
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


@dataclass
class VideoAnalysis:
    """Analyse videos.

    Parameters
    ----------

    Notes
    -----
    Use like this:
    videos = {id: {'path': 'path/to/file'}}
    skip_frames = 9
    va = VideoAnalysis(videos)
    va.start(skip_frames)
    """

    videos: Dict[str, any]
    exclude_area: List[int] = None
    road_area: List[int] = None
    setting_noise_reduction: int = None
    # TODO: more settings

    def scan(self):
        # scan_stream (yield picture, current_frame)
        # get signal (video_id, video_path, starttime, duration) from stream if new video begins and log this data to any active object
        pass

    def _yield_videos(self):
        for video_id, video in self.videos.items():
            yield video_id, video['path']

    def start(self, skip_frames=9):
        vl = VideoLoader(self._yield_videos(), skip_frames)
        vl.start()

        # loop over frames from the video file stream
        while vl.more():
            video_id, frame = vl.read()
            # display the size of the queue on the frame
            cv2.putText(frame, f"Queue Size: {vl.Q.qsize()}",
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
        vl.stop()

# What should the data structure look like?
"""
# TODO: define a better structure for video list
# Goal1: it should be guaranteed to be sorted by datetime (can be done in-class)
# Goal2: it should be possible to define max-time-gap between two videos
#   to decide whether to treat them separately
videos = {
    id: {
        'path':,
        'datetime':, # timediff is easier to calculate
        'fps':,
        'duration':,
    }
}

active_objects = {
    id: {
        'video_ids':,
        'video_paths':,
        'coordinates':,
        'bboxes':,
        'images':,
        'starttime':,
        'endtime':,
        'starttime_in_video':,
        'endtime_in_video':
    }
}

collected_objects = [
    {
        'id':,
        'video_ids':,
        'video_paths':,
        'coordinates':,
        'bboxes':,
        'images':,
        'starttime':,
        'endtime':,
        'starttime_in_video':,
        'endtime_in_video':
    },
     ]},
]
"""