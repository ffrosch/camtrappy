from __future__ import annotations
from typing import TYPE_CHECKING

import abc
import types

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List

import cv2


if TYPE_CHECKING:
    from camtrappy.core.base import Frame, VideoLoader


class IVisitor(metaclass=abc.ABCMeta):

    def __init__(self, apply_args: List[str] = None):
        super().__init__()
        self.apply_args = apply_args

        for arg in self.apply_args:
            if not hasattr(self, arg):
                raise AttributeError('One ore more of the specified `apply_args` '
                                     'is not an attribute.')
            if not type(getattr(self, arg)) == types.MethodType:
                raise TypeError('One or more of the specified `apply_args` '
                                'is not a method.')

    @abc.abstractmethod
    def apply(self):
        pass


class ObjectTracker(IVisitor):
    """Visitor Class.

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

    def __init__(self, objects: OrderedDict = None,
                 min_area: int = 50,
                 apply_args: List[str] = None):
        self.objects = objects
        self.min_area = min_area
        super().__init__(apply_args)

    def apply(self, frame: Frame):
        for arg in self.apply_args:
            f = getattr(self, arg)
            f(frame)

    def detect(self, frame):
        """Search for new objects."""
        contours, _ = cv2.findContours(frame.last,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
        self.contours = [c for c in contours if cv2.contourArea(c) > self.min_area]

    def track(self, frame):
        """Track existing objects."""
        pass

    def draw(self, frame):
        """Return frame with marked objects."""
        for c in self.contours:
            rect = cv2.boundingRect(c)
            x, y, w, h = rect
            cv2.rectangle(frame.original, (x, y), (x + w, y + h), 255, 2)


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