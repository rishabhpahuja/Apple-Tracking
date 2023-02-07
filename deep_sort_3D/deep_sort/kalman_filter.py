# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg


"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self,v,_st_weight_postion):
        ndim,ndim_3D, dt = 4, 3, 1.

        # Create Kalman filter model matrices.
        # self.vel=np.array([20,0,0])
        #To find obersvation states
        self._update_mat = np.eye(ndim, 2 * ndim) #4,8
        self._update_mat_3D = np.eye(ndim_3D, 2*ndim_3D)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = _st_weight_postion
        self.v=v

    def init_matrix(self, length):

        self.motion_mat=np.eye(3,length)
        self.jac_mat=np.eye(length,length)

        return self.motion_mat, self.jac_mat
    
    def initiate(self, measurement, indices=None):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        # import ipdb;ipdb.set_trace()
        mean = measurement.reshape((-1,3))[indices]
        # mean[4]=-33
        covariance = np.zeros((3*len(mean),3*len(mean)+3))
    
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 3n+3 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 3n+3,3n+3 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        std_pos = [self._std_weight_position,self._std_weight_position,self._std_weight_position]  
        motion_model, jac_mat =self.init_matrix(len(mean))

        # import ipdb;ipdb.set_trace()

        motion_cov=np.eye(len(mean),len(mean))*20
        motion_cov[:3,:3] = np.diag(np.square(std_pos))
        # import ipdb; ipdb.set_trace()
        mean = mean+np.dot(motion_model.T, self.v)
        # import ipdb; ipdb.set_trace()
        covariance = np.linalg.multi_dot((
            jac_mat, covariance, jac_mat.T)) + motion_cov #P_{n+1,n}

        return mean.reshape((-1,3)), covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        std = [
            2*self._std_weight_position * mean[3],
            2*self._std_weight_position * mean[3],
            1e-1,
            2*self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std)) #R
        # import ipdb; ipdb.set_trace()
        mean = np.dot(self._update_mat, mean) #H*x_n_cap, shows the observation, the states we are interested in
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T)) #(H*P_n_cap*H.T)
        return mean, covariance + innovation_cov #(H*P_n_cap*H.T+R)

    def update(self, mean, covariance, measurement, mean_3D,covariance_3D,measurement_3D):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        projected_mean, projected_cov = self.project(mean, covariance) #predicted state from H*x_n_cap, (H*P_n_cap*H.T+Q)
        projected_mean_3D, projected_cov_3D = self.project_3D(mean_3D, covariance_3D) #predicted state from H*x_n_cap, (H*P_n_cap*H.T+R)
        # import ipdb; ipdb.set_trace()
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        chol_factor_3D, lower_3D = scipy.linalg.cho_factor(
            projected_cov_3D, lower=True, check_finite=False)
        
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        kalman_gain_3D = scipy.linalg.cho_solve(
            (chol_factor_3D, lower_3D), np.dot(covariance_3D, self._update_mat_3D.T).T,
            check_finite=False).T

        innovation = measurement - projected_mean
        innovation_3D = measurement_3D - projected_mean_3D

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_mean_3D = mean_3D + np.dot(innovation_3D, kalman_gain_3D.T)

        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        new_covariance_3D = covariance_3D - np.linalg.multi_dot((
            kalman_gain_3D, projected_cov_3D, kalman_gain_3D.T))

        return new_mean, new_covariance, new_mean_3D, new_covariance_3D

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.

        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.

        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        # import ipdb; ipdb.set_trace()
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0) #shape: (10,1), means squred maha distance of one track with all detections
        return squared_maha

    