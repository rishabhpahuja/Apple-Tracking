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
        self.tracks = Track(initialized=False)
        self._next_id = np.array([],dtype=np.uint16)
        self.mean_3D=mean
        self.covariance_3D=np.array([_st_weight_position,_st_weight_position,_st_weight_position])
        self.mean_2D=np.array([[0,0,0,0]])
        self.confidence=np.array([0])

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
                self.kf, detections[detection_idx],track_idx)
        for i,track_idx in enumerate(unmatched_tracks):
            self.tracks.mark_missed(i)
        # import ipdb;ipdb.set_trace()
        if len(unmatched_detections)!=0:
            self._initiate_track(detections,unmatched_detections)
        # UNMATCHED_TRACKS=[self.tracks[i] for i in unmatched_tracks]
        # UNMACTHED_DETECTIONS=[detections[i] for i in unmatched_detections]
        # self.tracks = [t for t in self.tracks if not t.is_deleted()]
        
        # return UNMATCHED_TRACKS,UNMACTHED_DETECTIONS

    def _match(self, detections):

        '''
        detections: object of class detection
        '''

        def gated_metric(tracks, dets, track_indices):
            import ipdb; ipdb.set_trace()
            detection = dets.points_3D.reshape((-1,3))
            cost_matrix = self.metric.distance(detection, tracks, track_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        # import ipdb;ipdb.set_trace()

        
        # confirmed_tracks = [
        #     i for i in range(len(self.tracks.state)) if self.tracks.is_confirmed(i)]
        # unconfirmed_tracks = [
        #     i for i in range(len(self.tracks.state)) if not self.tracks.is_confirmed(i)] #Unmatched tracks from previous frame
        # import ipdb; ipdb.set_trace()

        # Associate confirmed tracks using appearance features.
        matches, unmatched_tracks, unmatched_detections = linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections)
        # import ipdb; ipdb.set_trace()
        # unmatched_tracks = list(set(unmatched_tracks + unconfirmed_tracks))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection, unmatched_detections):
        '''
        detection: object of class detection consisting of all the detections
        unmactched_detection: list consisting of indices of bbox of image space that are unmatched detections
        '''
        
        mean, covariance = self.kf.initiate(detection.points_3D,unmatched_detections)
        # import ipdb; ipdb.set_trace()        
        if self.covariance_3D.ndim==1:
            self.covariance_3D=np.eye(3,3*len(mean)+3)*self.covariance_3D[0]
        else:
            if covariance.shape[1]<self.covariance_3D.shape[1]:
                covariance=np.hstack((covariance,np.zeros((covariance.shape[0],self.covariance_3D.shape[1]-covariance.shape[1]))))
            elif covariance.shape[1]>self.covariance_3D.shape[1]:
                self.covariance_3D=np.hstack((self.covariance_3D,np.zeros((self.covariance_3D.shape[0],covariance.shape[1]-self.covariance_3D.shape[1]))))

        self.covariance_3D=np.vstack((self.covariance_3D,covariance))
        self.mean_3D=np.vstack((self.mean_3D,mean))

        self.mean_2D=np.vstack((self.mean_2D,detection.points_2D[unmatched_detections]))
        self.confidence=np.concatenate((self.confidence,detection.confidence[unmatched_detections]),axis=0)
        self._next_id=np.arange(len(self.confidence))
        class_name = detection.get_class()

        self.tracks=Track(
            mean_2D=self.mean_2D, mean_3D=self.mean_3D,covariance_3D=self.covariance_3D, track_id=self._next_id, n_init=np.ones(len(self.confidence))*self.n_init,
             max_age=np.ones(len(self.confidence))*self.max_age,class_name=class_name,confidence=self.confidence,initialized=True)
