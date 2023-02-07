# vim: expandtab:ts=4:sw=4
import numpy as np

class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean_2D : ndarray ((N+1)X4)
        Mean vector of the initial state distribution in image plane
    mean_3D : ndarray ((N+1)X3)
        Mean vector of the initial state distribution in world frame
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : array (N,) 
        A unique track identifier for all detections
    n_init : array (N,)
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : array (N,)
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    confidence : ndarray (N,)

    Attributes
    ----------
    mean_2D : ndarray
        Mean vector of the initial state distribution in image plane
    mean_3D : ndarray
        Mean vector of the initial state distribution in world frame
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : ndarray
        A unique track identifier.
    hits : ndarray
        Total number of measurement updates.
    age : ndarray
        Total number of frames since first occurance.
    time_since_update : nndarray
        Total number of frames since last measurement update.
    state : TrackState ndarray
        The current track state of all the tracks
    """

    def __init__(self, mean_2D=None, mean_3D=None,covariance_3D=None, track_id=None, n_init=None, max_age=None, confidence=None,
                  class_name=None, initialized=False):
        self.mean_2D = mean_2D
        self.mean_3D=mean_3D
        self.covariance_3D = covariance_3D
        self.track_id = track_id
        self.confidence=confidence

        if initialized:
            self.hits = np.ones(len(self.mean_3D))
            self.age = np.ones(len(self.mean_3D))
            self.time_since_update = np.zeros(len(self.mean_3D))
            self.state = np.array([TrackState.Tentative]*len(confidence))
        else:
            self.state=[]

        self._n_init = n_init
        self._max_age = max_age
        self.class_name = class_name
        self.initialized=initialized

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret
    
    def get_class(self):
        return self.class_name

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean_3D, self.covariance_3D = kf.predict(self.mean_3D.flatten(), self.covariance_3D)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection, index):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        self.mean_3D,self.covariance_3D = kf.update(
            detection.to_xyah(),self.mean_3D,self.covariance_3D,detection.points_3D)
        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update[index] = 0
        if self.state[index] == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self,i):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state[i] == TrackState.Tentative:
            self.state[i] = TrackState.Deleted
        # if self.state==TrackState.Confirmed:
        #     self.state==TrackState.Tentative
        elif self.time_since_update[i] > self._max_age[i]:
            self.state[i] = TrackState.Deleted

    def is_tentative(self,i):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state[i] == TrackState.Tentative

    def is_confirmed(self,i):
        """Returns True if this track is confirmed."""
        return self.state[i] == TrackState.Confirmed

    def is_deleted(self,i):
        """Returns True if this track is dead and should be deleted."""
        return self.state[i] == TrackState.Deleted
