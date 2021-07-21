from dataclasses import dataclass
from typing import Dict, List

import cv2

from camtrappy.core.base import VideoLoader


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
        'video_ids': [],
        'video_paths': [],
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