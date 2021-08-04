from __future__ import annotations

import abc

from dataclasses import dataclass
from typing import Any, List, TYPE_CHECKING, Tuple

import cv2
import numpy as np
from skimage import filters
from skimage.util import img_as_ubyte


if TYPE_CHECKING:
    from camtrappy.core.base import Frame


class ITransform(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def transform(self):
        pass


@dataclass
class TransformFactory(ITransform):

    transforms: List[ITransform]

    def transform(self, frame: Frame):
        for t in self.transforms:
            transformed = t.transform(frame.last)
            frame.append(transformed)
        return frame


@dataclass
class AdaptiveHistogram(ITransform):

    cliplimit: float = 2.0
    gridsize: Tuple[int, int] = (8, 8)

    def transform(self, frame):
        clahe = cv2.createCLAHE(self.cliplimit, self.gridsize)
        return clahe.apply(frame)


@dataclass
class AdaptiveThreshold(ITransform):

    blocksize: int = 7
    c: int = 2

    def transform(self, frame):
        return cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, self.blocksize, self.c)


@dataclass
class BGS(ITransform):
    # TODO: try library 'pybgs' instead!
    name: str = 'KNN'
    learningrate: int = -1 # -1 = auto, 0 = no learning, 1 = complete reinitialization with every frame

    def __post_init__(self):
        if self.name == 'MOG2':
            self.bgsub = cv2.createBackgroundSubtractorMOG2()
        elif self.name == 'KNN':
            self.bgsub = cv2.createBackgroundSubtractorKNN()
        else:
            raise ValueError('Specify a valid name, e.g. "MOG2" or "KNN".')

    def transform(self, frame):
        return self.bgsub.apply(frame, self.learningrate)

@dataclass
class BgsMOG(ITransform):
    # TODO: try library 'pybgs' instead!
    # TODO: https://docs.opencv.org/4.5.2/df/d5d/namespacecv_1_1bgsegm.html
    #       Try other subtractors and camera motion detection
    hist: int = 250
    nmixtures: int = 5
    background_ratio: float = 0.7
    noise: int = 10
    learningrate: float = -1

    def __post_init__(self):
        self.bgsub = cv2.bgsegm.createBackgroundSubtractorMOG(self.hist,
                                                              self.nmixtures,
                                                              self.background_ratio,
                                                              self.noise)

    def transform(self, frame):
        return self.bgsub.apply(frame, self.learningrate)


@dataclass
class BgsMOGMask(BgsMOG):

    def transform(self, frame):
        mask = self.bgsub.apply(frame, self.learningrate)
        return cv2.bitwise_and(frame, frame, mask=mask)


@dataclass
class BilateralFilter(ITransform):

    diameter: int = 5

    def transform(self, frame):
        d = self.diameter
        return cv2.bilateralFilter(frame, d, d * 2, d / 2)


@dataclass
class CannyEdge(ITransform):

    min: int
    max: int

    def transform(self, frame):
        return cv2.Canny(frame, self.min, self.max)


@dataclass
class GaussianBlur(ITransform):

    kernel: Tuple[int, int] = (5 ,5)

    def transform(self, frame):
        return cv2.GaussianBlur(frame, self.kernel, cv2.BORDER_DEFAULT)


@dataclass
class Gray(ITransform):

    def transform(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


@dataclass
class Gamma(ITransform):

    gamma: float = 1.0

    def transform(self, frame):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / self.gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        return cv2.LUT(frame, table)


@dataclass
class Hysteresis(ITransform):

    low: float = 0.1
    high: float = 0.35

    def transform(self, frame):
        edges = filters.sobel(frame)
        hyst = filters.apply_hysteresis_threshold(edges, self.low, self.high)
        return  img_as_ubyte(hyst)


@dataclass
class Normalize(ITransform):

    def transform(self, frame):
        frame = frame.copy()
        return cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)


@dataclass
class Resize(ITransform):

    percentage: int = 50
    interpolation: Any = cv2.INTER_AREA

    def transform(self, frame):
        width = int(frame.shape[1] * self.percentage / 100)
        height = int(frame.shape[0] * self.percentage / 100)
        dim = (width, height)
        return cv2.resize(frame, dim, self.interpolation)


@dataclass
class Threshold(ITransform):
    """
    types: binary, binary_inv, trunc, tozero, tozero_inv
    """

    min: int = 127
    type: str = 'binary'

    def transform(self, frame):
        if self.type == 'binary':
            _, frame = cv2.threshold(frame, self.min, 255, cv2.THRESH_BINARY)
        if self.type == 'binary_inv':
            _, frame = cv2.threshold(frame, self.min, 255, cv2.THRESH_BINARY_INV)
        if self.type == 'trunc':
            _, frame = cv2.threshold(frame, self.min, 255, cv2.THRESH_TRUNC)
        if self.type == 'tozero':
            _, frame = cv2.threshold(frame, self.min, 255, cv2.THRESH_TOZERO)
        if self.type == 'tozero_inv':
            _, frame = cv2.threshold(frame, self.min, 255, cv2.THRESH_TOZERO_INV)

        return frame
