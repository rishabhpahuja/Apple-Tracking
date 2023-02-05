# vim: expandtab:ts=4:sw=4
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric,max_age=75, n_init=1,v=np.array([0,0,20]),mean=np.array([0,0,0]),_st_weight_position=2):
        self.metric = metric
        # self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter(v,_st_weight_position)
        self.tracks = []
        self._next_id = 1
        self.mean=mean
        self.covariance=np.array([_st_weight_position,_st_weight_position,_st_weight_position])

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """

        self.tracks.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        # import ipdb;ipdb.set_trace()
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        # import ipdb;ipdb.set_trace()
        if len(unmatched_detections)!=0:
            self._initiate_track(detections,unmatched_detections)
        UNMATCHED_TRACKS=[self.tracks[i] for i in unmatched_tracks]
        UNMACTHED_DETECTIONS=[detections[i] for i in unmatched_detections]
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)
        
        return UNMATCHED_TRACKS,UNMACTHED_DETECTIONS

    def _match(self, detections):

        '''
        detections: object of class detection
        '''

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        # import ipdb;ipdb.set_trace()
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()] #Unmatched tracks from previous frame

        # Associate confirmed tracks using appearance features.
        matches, unmatched_tracks, unmatched_detections = linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)
        # import ipdb; ipdb.set_trace()
        unmatched_tracks = list(set(unmatched_tracks + unconfirmed_tracks))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection, unmatched_detections):
        '''
        detection: object of class detection consisting of all the detections
        unmactched_detection: list consisting of indices of bbox of image space that are unmatched detections
        '''
        mean, covariance = self.kf.initiate(detection.points_3D,unmatched_detections)
        self.mean=np.vstack((self.mean,mean))
        self.covariance=np.vstack((self.covariance,covariance))
        import ipdb;ipdb.set_trace()
        class_name = detection.get_class()
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            feature=detection.feature, class_name=class_name, confidence=detection.confidence,points_3D=detection.points_3D,mean_3D=mean_3D,\
                covariance_3D=covariance_3D))
        self._next_id += 1
