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
        mean = measurement[indices]
        covariance = np.eye(3*len(mean),3*len(mean))
    
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

        motion_cov=np.zeros((len(mean),len(mean)))
        motion_cov[:3,:3] = np.diag(np.square(std_pos))
        mean = mean+np.dot(motion_model.T, self.v)
        # import ipdb; ipdb.set_trace()
        covariance = np.linalg.multi_dot((
            jac_mat, covariance, jac_mat.T)) + motion_cov #P_{n+1,n}

        return mean.reshape((-1,3)), covariance

    def innovation_matrix(self, nos):
        '''
        nos: int
            number of tracks to be updated

        '''

        diag_matrix=np.eye(3*nos-3,3*nos-3)
        first_row=np.zeros((3,3*nos-3)).flatten()
        first_column=np.zeros((3*nos-3,3)).flatten()
        for i in range(3*(3*nos-3)):
            if i%4==0 or i%4==1:
                first_row[i]=self._std_weight_position
                first_column[i]=self._std_weight_position
            
        # import ipdb; ipdb.set_trace()
        first_row=first_row.reshape((3,-1))
        first_column=first_column.reshape((-1,3))
        rover_mat=np.eye(3,3)*self._std_weight_position*1.5
        inn_mat=np.vstack((np.hstack((rover_mat,first_row)),
                            np.hstack((first_column,diag_matrix))))
                
        return inn_mat

    def get_H_matrix():
        pass
    
    def project(self, mean, covariance,matches,h_matrix):
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
        
        innovation_cov =np.square(self.innovation_matrix(len(mean))) #R
        mean = np.dot(h_matrix, mean.flatten()) #H*x_n_cap, shows the observation, the states we are interested in
        covariance = np.linalg.multi_dot((
            h_matrix, covariance, h_matrix.T)) #(H*P_n_cap*H.T)
        return mean.reshape((-1,3)), covariance + innovation_cov #(H*P_n_cap*H.T+R)

    def update(self, measurement,mean, covariance,matches):
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
        matches=np.vstack((np.array([0,0]),matches))
        h_matrix=np.eye(len(mean.flatten()),len(mean.flatten()))
        projected_mean, projected_cov = self.project(mean, covariance,matches,h_matrix) #predicted state from H*x_n_cap, (H*P_n_cap*H.T+Q)
        for x,y in matches:
            mean[x]=projected_mean[x]
            covariance[3*x:3*x+3,3*x:3*x+3]=projected_cov[3*x:3*x+3,3*x:3*x+3]
        # import ipdb; ipdb.set_trace()
            chol_factor, lower = scipy.linalg.cho_factor(
                projected_cov[3*x:3*x+3,3*x:3*x+3], lower=True, check_finite=False)
            
            kalman_gain = scipy.linalg.cho_solve(
                (chol_factor, lower), np.dot(covariance[3*x:3*x+3,3*x:3*x+3], h_matrix[3*x:3*x+3,3*x:3*x+3]).T,
                check_finite=False).T

            innovation = measurement.points_3D[y] - projected_mean[x]

            mean[x] = mean[x] + np.dot(innovation, kalman_gain.T)

            covariance[3*x:3*x+3,3*x:3*x+3] = covariance[3*x:3*x+3,3*x:3*x+3] - np.linalg.multi_dot((
                kalman_gain, projected_cov[3*x:3*x+3,3*x:3*x+3], kalman_gain.T))


        return mean, covariance

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

    