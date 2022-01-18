# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Data association class with single nearest neighbor association and gating based on Mahalanobis distance
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# add project directory to python path to enable relative imports
import os
import sys

# imports
import numpy as np
from scipy.stats.distributions import chi2

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import misc.params as params


class Association:
    '''Data association class with single nearest neighbor association and gating based on Mahalanobis distance'''

    def __init__(self):
        self.association_matrix = np.matrix([])
        self.unassigned_tracks = []
        self.unassigned_meas = []

    def associate(self, track_list, meas_list, KF):

        ############
        # TODO Step 3: association:
        # - replace association_matrix with the actual association matrix based on Mahalanobis distance (see below) for all tracks and all measurements
        # - update list of unassigned measurements and unassigned tracks
        ############

        N = len(track_list) # N tracks
        M = len(meas_list) # M measurements
        
        # initialize association matrix
        self.association_matrix = np.inf*np.ones((N,M)) 
        
        # initialize unassigned track list and unassigned measurement list
        # list are indices for each track and measurement
        self.unassigned_tracks = list(range(N))
        self.unassigned_meas = list(range(M))
        
        # loop over all tracks and all measurements to set up association matrix
        for i in range(N): 
            track = track_list[i]
            for j in range(M):
                meas = meas_list[j]
                
                # calculate MHD of each track and measurement pair
                MHD_dist = self.MHD(track, meas, KF)
                
                # if MHD is smaller than limit, update association matrix
                if self.gating(MHD_dist, meas.sensor):
                    self.association_matrix[i,j] = MHD_dist
        ############
        # END student code
        ############ 

    def get_closest_track_and_meas(self):
        ############
        # TODO Step 3: find closest track and measurement:
        # - find minimum entry in association matrix
        # - delete row and column
        # - remove corresponding track and measurement from unassigned_tracks and unassigned_meas
        # - return this track and measurement
        ############
        A = self.association_matrix
        if np.min(A) == np.inf:
            return np.nan, np.nan

        # the following only works for at most one track and one measurement
        i, j = np.unravel_index(np.argmin(self.association_matrix, axis=None), self.association_matrix.shape)
        update_track = self.unassigned_tracks[i]
        update_meas = self.unassigned_meas[j]

        # remove from list
        self.unassigned_tracks.remove(update_track)
        self.unassigned_meas.remove(update_meas)
        self.association_matrix = np.delete(self.association_matrix, i, 0)
        self.association_matrix = np.delete(self.association_matrix, j, 1)

        ############
        # END student code
        ############ 
        return update_track, update_meas

    def gating(self, MHD, sensor):
        ############
        # TODO Step 3: return True if measurement lies inside gate, otherwise False
        ############

        limit = chi2.ppf(params.gating_threshold, df=sensor.dim_meas)
        if MHD < limit:
            return True
        else:
            return False

        ############
        # END student code
        ############ 

    def MHD(self, track, meas, KF):
        ############
        # TODO Step 3: calculate and return Mahalanobis distance
        ############

        # retrive measurement matrix
        H = meas.sensor.get_H(track.x)
        
        # calculate gamma
        gamma = KF.gamma(track, meas)
        
        # calculate S
        S = KF.S(track, meas, H)
        
        # calculate MHD
        MHD = gamma.transpose()*np.linalg.inv(S)*gamma
        
        return MHD

        ############
        # END student code
        ############

    def associate_and_update(self, manager, meas_list, KF):
        # associate measurements and tracks
        self.associate(manager.track_list, meas_list, KF)

        # update associated tracks with measurements
        while self.association_matrix.shape[0] > 0 and self.association_matrix.shape[1] > 0:

            # search for next association between a track and a measurement
            ind_track, ind_meas = self.get_closest_track_and_meas()
            if np.isnan(ind_track):
                print('---no more associations---')
                break
            track = manager.track_list[ind_track]

            # check visibility, only update tracks in fov    
            if not meas_list[0].sensor.in_fov(track.x):
                continue

            # Kalman update
            print('update track', track.id, 'with', meas_list[ind_meas].sensor.name, 'measurement', ind_meas)
            KF.update(track, meas_list[ind_meas])

            # update score and track state 
            manager.handle_updated_track(track)

            # save updated track
            manager.track_list[ind_track] = track

        # run track management 
        manager.manage_tracks(self.unassigned_tracks, self.unassigned_meas, meas_list)

        for track in manager.track_list:
            print('track', track.id, 'score =', track.score)