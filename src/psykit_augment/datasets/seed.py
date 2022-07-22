import datetime as dt
import glob
import os
import zipfile

import mne
import numpy as np
from mne.channels import make_standard_montage
from scipy.io import loadmat

from psykit_augment.datasets import _download as dl
from psykit_augment.datasets._base import BaseDataset


SEED_PREPROCESSED_PATH = "data/seed_dataset/Preprocessed_EEG"
LABELS_PATH = "data/seed_dataset/Preprocessed_EEG"


class SEED(BaseDataset):
    "SEED"

    def __init__(self):
        super().__init__(
            subjects=[1, 2, 3, 4, 6, 7, 8, 9, 10],
            sessions_per_subject=15*3,
            events=dict(Target=2, NonTarget=1),
            code="SEED Dataset",
            interval=[0, 1],
            paradigm="emotion",
            doi="10.1016/j.jneumeth.2007.03.005",
        )

    def _get_single_run_data(self, file_path):
        pass

    def _get_single_subject_data(self, subject):
        """

        :param subject:
        :return:
        """

        file_path_list = self.data_path(subject, SEED_PREPROCESSED_PATH)
        labels = loadmat()
        sessions = {}

        for session_file_path in sorted(file_path_list):
            session_name = session_file_path.split("_")[-1]
            mat = loadmat(session_file_path)
            for trial_key, trial_eeg in mat.items():

                if session_name not in sessions.keys():
                    sessions[session_name] = {}

                if trial_key not in ['__header__', '__version__', '__globals__']:
                    sessions[session_name][trial_key] = trial_eeg
        return sessions

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """

        :param subject:
        :param path:
        :param force_update:
        :param update_path:
        :param verbose:
        :return:
        """
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        pattern = os.path.join(f"{subject}_*.mat")
        subject_paths = sorted(glob.glob(path + pattern))

        return subject_paths
