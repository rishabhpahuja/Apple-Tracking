# vim: expandtab:ts=4:sw=4
import numpy as np


class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    points_2D : array_like
        Bounding box in format `(x1, y1, x2, y2)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    points_2D : ndarray
        Bounding box in format `(top left x, top left y, righ x, right y)`.
    confidence : ndarray
        Detector confidence score.
    class_name : ndarray
        Detector class.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.

    """

    def __init__(self, points_2D,confidence, class_name, point_3D=None):
        self.points_2D = np.asarray(points_2D, dtype=np.float32)
        self.confidence = np.asarray(confidence,np.float32)
        self.class_name = class_name
        self.points_3D=point_3D

    def get_class(self):
        return self.class_name

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, w, h)`, i.e.,
        `(top left x,y,width,height)`.
        """
        ret = self.points_2D.copy()
        ret[2]=abs(ret[0]-ret[2])
        ret[3]=abs(ret[1]-ret[3])
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.points_2D.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
