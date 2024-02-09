from data_handler import save_data_frame, check_dir

import matplotlib.pyplot as plt
import pandas as pd
import sys


class Subject:
    def __init__(self, label, motion, heart_rate, step):
        self.id = label.subject_id
        self.label = label
        self.motion = motion
        self.heart_rate = heart_rate
        self.step = step

    def plot(self):
        fig = plt.figure()
        ax = plt.subplot(111)
        pd.DataFrame(self.label.prepared_data[1]).plot(ax=ax)
        pd.DataFrame(self.heart_rate.normalized_data[1]).plot(ax=ax)
        pd.DataFrame(self.step.normalized_data[1]).plot(ax=ax)
        pd.DataFrame(self.motion.normalized_data[1]).plot(ax=ax)
        check_dir(sys.path[0] + "/figs")
        ax.legend(["label", "heart_rate", "step", "motion"])
        fig.savefig('figs/full_figure_{}.png'.format(self.id))

    def save_normalized_data(self):
        save_data_frame("motions", self.motion.normalized_data, self.id)
        save_data_frame("labels", self.label.prepared_data, self.id)
        save_data_frame("steps", self.step.normalized_data, self.id)
        save_data_frame(
            "heart_rates",
            self.heart_rate.normalized_data,
            self.id)

    def subject_data(self, with_motion=True):
        num_samples = self.label.prepared_data.shape[0]
        data = {
            "subject_id": [self.id] * num_samples,
            "heart_rate": list(self.heart_rate.normalized_data.iloc[:, -1]),
            "motion": list(self.motion.normalized_data.iloc[:, -1]),
            "step": list(self.step.normalized_data.iloc[:, -1]),
            "sleep_phase": list(self.label.prepared_data.iloc[:, -1]),
        }
        if not with_motion:
            del data["motion"]
        return pd.DataFrame(data)

    def subject_data_cut_at_last_measured(self, feature_type):
        data = self.subject_data()
        if feature_type == "motion":
            last_idx = self.motion.find_last_measured_data(0)
        elif feature_type == "heart_rate":
            last_idx = self.heart_rate.find_last_measured_data(0)
        elif feature_type == "step":
            last_idx = self.step.find_last_measured_data(0)
        return data.iloc[:last_idx]


class SubjectCollector:
    def __init__(self, subjects) -> None:
        self.subjects = subjects

    def generate_sleep_data(self):
        data = []
        for s in self.subjects:
            data.append(s.subject_data())
        return pd.concat(data, ignore_index=True)

    def generate_sleep_data_w_cut(self, feature_type):
        data = []
        for s in self.subjects:
            data.append(s.subject_data_cut_at_last_measured(feature_type))
        return pd.concat(data, ignore_index=True)
