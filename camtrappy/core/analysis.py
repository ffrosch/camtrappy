from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List

import cv2

from camtrappy.core.base import VideoLoader


@dataclass
class ObjectTracker:
    """
    objects: {
        id: {
            video_id: {
                frames: [],
                bboxes: [],
                centroids: []
            },
            lost: bool,
        }
    }
    """


    objects: OrderedDict = None
    min: int = 50

    def detect(self, frame):
        """Search for new objects."""
        contours, _ = cv2.findContours(frame,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
        self.contours = [c for c in contours if cv2.contourArea(c) > self.min]

    def track(self):
        """Track existing objects."""
        pass

    def draw(self, frame):
        """Return frame with marked objects."""
        for c in self.contours:
            rect = cv2.boundingRect(c)
            x, y, w, h = rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)


@dataclass
class VideoAnalysis:

    exclude_area: List[int] = None
    road_area: List[int] = None
    setting_noise_reduction: int = None
    stream: VideoLoader = field(init=False)


# What should the data structure look like?
"""
# TODO: define a better structure for video list
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