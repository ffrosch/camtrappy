from __future__ import annotations
from typing import TYPE_CHECKING

import abc
import types

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np
from scipy.spatial import distance as dist


if TYPE_CHECKING:
    from camtrappy.core.base import Frame, VideoLoader


def bboxes_from_polygons(polygons: List[np.ndarray]) -> List[np.ndarray]:
    """Return Bounding Boxes of a list of Polygons."""
    return [cv2.boundingRect(poly) for poly in polygons]

def bbox_intersects(a, b):
    """Test for intersection of two bboxes."""
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0] + a[2], b[0] + b[2]) - x
    h = min(a[1] + a[3], b[1] + b[3]) - y
    if w < 0 or h < 0:
        return False
    return True

def centroids_from_bboxes(bboxes: List[np.ndarray]) -> List[Tuple]:
    """Calculate centroids for a list of bboxes."""
    # initialize an array of input centroids for the current frame
    inputCentroids = np.zeros((len(bboxes), 2), dtype="int")

    # loop over the bounding box rectangles
    for (i, (startX, startY, width, height)) in enumerate(bboxes):
        # use the bounding box coordinates to derive the centroid
        cX = int((2*startX + width) / 2.0)
        cY = int((2*startY + height) / 2.0)
        inputCentroids[i] = (cX, cY)

    return inputCentroids

def detect_contours(frame: Frame, min_area: int = None) -> List[np.ndarray]:
    """Detect Contours of elements in an image.

    Return a list of contours where every contour
    is a numpy array of (x, y) coordinates.
    """
    contours, _ = cv2.findContours(frame.last,
                                    cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if min_area is not None:
        contours = [c for c in contours if cv2.contourArea(c) > min_area]
    return contours

def draw_bboxes(frame: Frame, bboxes: List[np.ndarray]):
    """Draw bboxes on a frame."""
    for bbox in bboxes:
        x, y, w, h = bbox
        cv2.rectangle(frame.original, (x, y), (x + w, y + h), 255, 2)

def draw_object_ids(frame: Frame, objects):
        for objectID, centroid in objects.items():
            text = f'ID {objectID}'
            img = frame.original
            x, y = centroid
            cv2.putText(img, text, (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 0)
            cv2.circle(img, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 255, 2)

def merge_bboxes(bboxes, eps: float = 1.5):
    """Merge all overlapping bboxes.

    groupRectangles: The function is a wrapper for the
    generic function partition . It clusters all the input rectangles
    using the rectangle equivalence criteria that combines rectangles
    with similar sizes and similar locations.
    The similarity is defined by eps.
    When eps=0 , no clustering is done at all. If eps→+inf,
    all the rectangles are put in one cluster.
    Then, the small clusters containing less than or
    equal to groupThreshold rectangles are rejected.
    In each other cluster, the average rectangle is computed and
    put into the output rectangle list.

    Parameters
    ----------
    bboxes
    eps : float
        Defines how far away rectangles can be from each other
        to be considered for grouping.
    """
    tmp = []
    for bbox in bboxes:
        x, y, w, h = bbox
        # cv2.groupRectangles returns a result for each successful grouping
        # duplicate every bbox to retain non-overlapping bboxes
        tmp.append([x, y, w, h])
        tmp.append([x, y, w, h])
    merged, weights = cv2.groupRectangles(tmp, groupThreshold=1, eps=eps)
    merged = np.asarray(merged).tolist()
    return merged


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


class CentroidTracker(IVisitor):
    """Copied (and adjusted) from pyimagesearch.

    https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
    """
    def __init__(self,
                 min_area: int = None,
                 eps: float = 1.5,
                 maxDisappeared: int= 50,
                 apply_args: List[str] = list()):

        super().__init__(apply_args)

        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.min_area = min_area
        self.eps = eps
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared

    def apply(self, frame: Frame):
        contours = detect_contours(frame, self.min_area)
        bboxes = bboxes_from_polygons(contours)
        bboxes = merge_bboxes(bboxes, eps=self.eps)
        objects = self.update(bboxes)

        draw_bboxes(frame, bboxes)
        draw_object_ids(frame, objects)

    def register(self, centroid):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        # check to see if the list of input bounding box rectangles
        # is empty
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            keys = list(self.disappeared.keys())
            for objectID in keys:
                self.disappeared[objectID] += 1

                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # return early as there are no centroids or tracking info
            # to update
            return self.objects

        inputCentroids = centroids_from_bboxes(rects)

        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value as at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()

            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols:
                    continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        # return the set of trackable objects
        return self.objects


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