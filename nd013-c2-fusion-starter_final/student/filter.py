# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 

class Filter:
    '''Kalman filter class'''
    def __init__(self):
        pass

    def F(self):
        ############
        # TODO Step 1: implement and return system matrix F
        ############

        dim_v = int(params.dim_state/2)
        F = np.identity(params.dim_state)
        F[0:dim_v, dim_v:] = np.identity(dim_v)*params.dt
        return np.matrix(F)
        
        ############
        # END student code
        ############ 

    def Q(self):
        ############
        # TODO Step 1: implement and return process noise covariance Q
        ############
        dim_v = int(params.dim_state/2)
        noise_p = np.power(params.dt,3)/3*params.q
        noise_v = params.dt*params.q
        noise_cross = np.power(params.dt,2)/2*params.q
        Q = np.identity(params.dim_state)
        Q[0:dim_v, 0:dim_v] = np.identity(dim_v)*noise_p
        Q[dim_v:, dim_v:] = np.identity(dim_v)*noise_v
        Q[0:dim_v, dim_v:] = np.identity(dim_v)*noise_cross
        Q[dim_v:, 0:dim_v] = np.identity(dim_v)*noise_cross
        
        return np.matrix(Q)
        
        ############
        # END student code
        ############ 

    def predict(self, track):
        ############
        # TODO Step 1: predict state x and estimation error covariance P to next timestep, save x and P in track
        ############

        F = self.F()
        Q = self.Q()
        
        # calculate prediction results
        x_pred = F*track.x
        p_pred = F * track.P * F.transpose() + Q
        

        track.set_x(x_pred)
        track.set_P(p_pred)

        ############
        # END student code
        ############ 

    def update(self, track, meas):
        ############
        # TODO Step 1: update state x and covariance P with associated measurement, save x and P in track
        ############
        # retrive current prediction results
        x_pred = track.x
        p_pred = track.P
        
        # retrive jacobian matrix at current prediction result
        H = meas.sensor.get_H(x_pred)
        
        # retrive value of (real measurement - current prediction)
        gamma = self.gamma(track, meas)
        
        # calculate S
        S = self.S(track, meas, H)
        
        # calculate kalman filter
        K = p_pred * H.transpose() * np.linalg.inv(S)
        
        # update x
        x = x_pred + K*gamma
        
        # update covariance P
        I = np.identity(params.dim_state)
        p = (I - K*H)*p_pred
        
        
        # set p and x

        track.set_x(x)
        track.set_P(p)

        ############
        # END student code
        ############ 
        track.update_attributes(meas)
    
    def gamma(self, track, meas):
        ############
        # TODO Step 1: calculate and return residual gamma
        ############

        z = meas.z - meas.sensor.get_hx(track.x)
        return z
        ############
        # END student code
        ############ 

    def S(self, track, meas, H):
        ############
        # TODO Step 1: calculate and return covariance of residual S
        ############

        S = (H * track.P * H.transpose()) + meas.R
        return S
        
        ############
        # END student code
        ############ 