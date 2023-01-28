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

    def __init__(self):
        ndim,ndim_3D, dt = 4, 3, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt        
        self._motion_mat[0,4]=-33
       
        self._motion_mat_3D=np.eye(2*ndim_3D,2*ndim_3D)
        for i in range(ndim_3D):
            self._motion_mat_3D[i, ndim_3D + i] = dt
        self._motion_mat_3D[0,3]=-20
        self._motion_mat_3D[0,5]=0
        #To find obersvation states
        self._update_mat = np.eye(ndim, 2 * ndim) #4,8
        self._update_mat_3D = np.eye(ndim_3D, 2*ndim_3D)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        # mean[4]=-33
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        # std = [
        #     self._std_weight_position * measurement[3],
        #     self._std_weight_position * measurement[3],
        #     1e-2,
        #     self._std_weight_position * measurement[3],
        #     self._std_weight_velocity * measurement[3],
        #     self._std_weight_velocity * measurement[3],
        #     1e-5,
        #     self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def initiate_3D(self, points_3D,aspect_ratio):
        """Create track from unassociated measurement.

        Parameters
        ----------
        points_3D : ndarray
            Coordinates of fruit in 3D plane

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (6 dimensional) and covariance matrix (6x6
            dimensional) of the new track. 

        """
        mean_pos = points_3D
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        # mean[4]=-20
        std = [
            2 * self._std_weight_position * aspect_ratio,
            2 * self._std_weight_position * aspect_ratio,
            2 * self._std_weight_position * aspect_ratio,
            2 * self._std_weight_velocity * aspect_ratio,
            2 * self._std_weight_velocity * aspect_ratio,
            2 * self._std_weight_velocity * aspect_ratio    ]

        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance, mean_3D, covariance_3D):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.
        mean_3D: ndarray
            The 6 dimensional mean vector of the object state in real world
            at the previous time step.
        covariance_3D: ndarray
            The 6X6 dimensional covariance matrix of the object state in real
            world at the previous time step. 

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]
        std_pos_3D = [
            self._std_weight_position * mean_3D[0],
            self._std_weight_position * mean_3D[1],
            1e-3
        ]
        std_vel_3D = [
            0.001,
            0.001,
            1e-5
            ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel])) #Q matrix for 2D
        motion_cov_3D= np.diag(np.square(np.r_[std_pos_3D, std_vel_3D])) #Q matrix for 3D

        mean = np.dot(self._motion_mat, mean)
        mean_3D = np.dot(self._motion_mat_3D, mean_3D)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov #P_{n+1,n}
        covariance_3D = np.linalg.multi_dot((
            self._motion_mat_3D, covariance_3D, self._motion_mat_3D.T)) + motion_cov_3D #P_{n+1,n}

        return mean, covariance, mean_3D, covariance_3D

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

    def project_3D(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (3 dimensional array).
        covariance : ndarray
            The state's covariance matrix (3x3 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        std = [
            1e-3,
            1e-3,
            1e-3,
            ]
        innovation_cov = np.diag(np.square(std)) #R
        # import ipdb;ipdb.set_trace()
        mean = np.dot(self._update_mat_3D, mean) #H*x_n_cap
        covariance = np.linalg.multi_dot((
            self._update_mat_3D, covariance, self._update_mat_3D.T)) #(H*P_n_cap*H.T)
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

    def gating_distance_3D(self, mean, covariance, measurements,
                            only_position=False):
            """Compute gating distance between state distribution and measurements.

            A suitable distance threshold can be obtained from `chi2inv95`. If
            `only_position` is False, the chi-square distribution has 3 degrees of
            freedom, otherwise 2.

            Parameters
            ----------
            mean : ndarray
                Mean vector over the state distribution (3 dimensional).
            covariance : ndarray
                Covariance of the state distribution (3x3 dimensional).
            measurements : ndarray
                An Nx3 dimensional matrix of N measurements, each in
                format (x, y, z) where (x, y, z) is the bounding box center
                position
            only_position : Optional[bool]
                If False, distance computation is done with respect to the bounding
                box center position only.

            Returns
            -------
            ndarray
                Returns an array of length N, where the i-th element contains the
                squared Mahalanobis distance between (mean, covariance) and
                `measurements[i]`.

            """
            # import ipdb; ipdb.set_trace()
            mean, covariance = self.project_3D(mean, covariance)

            cholesky_factor = np.linalg.cholesky(covariance)
            d = measurements - mean
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False,
                overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
