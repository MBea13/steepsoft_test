import pandas as pd
import numpy as np


class Feature:
    def __init__(
            self,
            subject_id,
            data,
            start_time,
            end_time,
            labeled_data,
            prepared_used=False) -> None:
        self.__subject_id = subject_id
        self.__data = data
        self.__start_time = start_time
        self.__end_time = end_time
        self.__labeled_data = labeled_data
        if prepared_used:
            self._normalized_data = data.copy()

    @property
    def subject_id(self):
        return self.__subject_id

    @property
    def labeled_data(self):
        return self.__labeled_data

    @property
    def normalized_data(self):
        if hasattr(self, '_normalized_data'):
            return self._normalized_data
        else:
            self._normalized_data = self.normalize()
        return self._normalized_data

    def __insert_new_data(self, data, idx, time, val):
        """Inserts new row into DataFrame

        Parameters
        ----------
        data : pd.DataFrame
            used DataFrame
        idx : int
            row index for insertion
        time : float/int
            time/first value
        val : list
            rest of the (feature) values

        Returns
        -------
        pd.DataFrame
            DataFrame with inserted row
        """
        if idx == -1:
            idx = data.shape[0] + 1
        line = pd.DataFrame([[time, *val]])
        df2 = pd.concat([data.iloc[:idx], line, data.iloc[idx:]]
                        ).reset_index(drop=True)
        return df2

    def _parse_data(self):
        """Cuts plus values, if there is no exact start/end, automatically fills its place with the gven interval limit
        and the value will be equal with the next found value

        Returns
        -------
        _type_
            _description_
        """
        df = self.__data.loc[(self.__data[0] >= self.__start_time) & (
            self.__data[0] <= self.__end_time)]
        if df.shape[0] > 0:
            if df.iloc[0][0] != self.__start_time:
                df = self.__insert_new_data(df, 0, self.__start_time, [
                                            i for i in df.iloc[0][1:]])
            if df.iloc[-1][0] != self.__end_time:
                df = self.__insert_new_data(
                    df, -1, self.__end_time, [i for i in df.iloc[-1][1:]])
        return df

    def __merge_with_labeled(self, data):
        """Merges feature and labeled DataFrames, so if a labeled time does not exist in the feature DataFrame, a new row will be inserted (where time will be from the labeled set, values will be the duplicates of the previous row)

        Parameters
        ----------
        data : pd.DataFrame
            feature data

        Returns
        -------
        pd.DataFrame
            merged data
        """
        if data.shape[0] > 0:
            times = list(self.__labeled_data[0])
            if data.shape[1] == 2:
                data.columns = ['A', 'B']
            else:
                data.columns = ['A', 'B', 'C', 'D']
            # Create a new DataFrame with the new 'A' values and matching 'B'
            # values
            new_df = pd.DataFrame({'A': times})

            # Concatenate the original DataFrame with the new DataFrame
            concatenated_df = pd.concat([data, new_df]).sort_values(by='A')

            # Fill missing values using forward fill
            for c in data.columns[1:]:
                concatenated_df[c] = concatenated_df[c].fillna(method='ffill')
            concatenated_df = concatenated_df.drop_duplicates()
            return concatenated_df
        else:
            data = self.__labeled_data.copy()
            for idx in range(data.shape[0]):
                data.loc[idx, list(range(data.shape[1]))[1:]] = 0
            if data.shape[1] == 2:
                data.columns = ['A', 'B']
            else:
                data.columns = ['A', 'B', 'C', 'D']
            return data

    def __prepare_data(self):
        data = self._parse_data()
        return self.__merge_with_labeled(data)

    def normalize(self):
        """Calculates the mean of features between label timestamps and normalizes the calculated set

        Returns
        -------
        pd.DataFrame
            normalized data with mean values
        """
        data = self.__prepare_data()
        times = list(self.__labeled_data[0])
        lim1 = times[0]
        lim2 = times[1]
        temp_dict = {lim1: [lim1, data.loc[data['A'] == lim1]['B'].values[0]]}
        for i in range(2, len(times)):
            temp_dict[lim2] = [
                lim2, data[(data['A'] < lim2) & (data['A'] >= lim1)]['B'].mean()]
            lim1 = lim2
            lim2 = times[i]
        temp_dict[lim2] = [
            lim2, data[(data['A'] < lim2) & (data['A'] >= lim1)]['B'].mean()]
        # scaler = MinMaxScaler(feature_range=(0, 1))
        temp_df = pd.DataFrame(temp_dict).astype("float").transpose()
        # dataset = scaler.fit_transform(np.array(temp_df.iloc[:, -1]).reshape(-1, 1))
        # temp_df.iloc[:, -1] = dataset.flatten()
        return temp_df

    def find_last_measured_data(self, val):
        measurements = list(self.normalized_data.iloc[:, -1])
        found = False
        for i in range(1, len(measurements), 3):
            temp_list = measurements[i:]
            if all(p == val for p in temp_list) and len(temp_list) > 0:
                found = True
                break
        if not found:
            i = len(measurements)
        return i


class HeartRate(Feature):
    def __init__(
            self,
            subject_id,
            data,
            start_time,
            end_time,
            labeled_data,
            prepared_used=False) -> None:
        """Heart rate class

        Parameters
        ----------
        subject_id : int
        data : pd.DataFrame
        start_time : int
            first time value, where legit label was found (use Label_obj.start_time)
        end_time : int
            last time value, where legit label was found (use Label_obj.start_time)
        labeled_data : pd.DataFrame
            prepared label data (use Label_obj.prepared_data)
        prepared_used : bool, optional
            if True loaded data will be saved automatically as prepared, by default False
        """
        super().__init__(
            subject_id,
            data,
            start_time,
            end_time,
            labeled_data,
            prepared_used)


class Step(Feature):
    def __init__(
            self,
            subject_id,
            data,
            start_time,
            end_time,
            labeled_data,
            prepared_used=False) -> None:
        """Step class

        Parameters
        ----------
        subject_id : int
        data : pd.DataFrame
        start_time : int
            first time value, where legit label was found (use Label_obj.start_time)
        end_time : int
            last time value, where legit label was found (use Label_obj.start_time)
        labeled_data : pd.DataFrame
            prepared label data (use Label_obj.prepared_data)
        prepared_used : bool, optional
            if True loaded data will be saved automatically as prepared, by default False
        """
        super().__init__(
            subject_id,
            data,
            start_time,
            end_time,
            labeled_data,
            prepared_used)


class Motion(Feature):
    def __init__(
            self,
            subject_id,
            data,
            start_time,
            end_time,
            labeled_data,
            prepared_used=False) -> None:
        """Motion class

        Parameters
        ----------
        subject_id : int
        data : pd.DataFrame
        start_time : int
            first time value, where legit label was found (use Label_obj.start_time)
        end_time : int
            last time value, where legit label was found (use Label_obj.start_time)
        labeled_data : pd.DataFrame
            prepared label data (use Label_obj.prepared_data)
        prepared_used : bool, optional
            if True loaded data will be saved automatically as prepared, by default False
        """
        super().__init__(
            subject_id,
            data,
            start_time,
            end_time,
            labeled_data,
            prepared_used)

    def normalize(self):
        """Calculates the sum of distance between points of features between label timestamps and normalizes the calculated set

        Returns
        -------
        pd.DataFrame
            normalized data with sum of distances values
        """
        data = self._parse_data()
        times = list(self.labeled_data[0])
        def dist(x, y): return np.linalg.norm(x - y)
        temp_dict = {times[0]: [times[0], 0]}
        times_idx = 1
        lim = times[times_idx]
        p1 = np.array(data.iloc[0][1:])
        p2 = np.array(data.iloc[1][1:])
        s = 0
        for _, row in data.iloc[2:].iterrows():
            p2 = np.array(row[1:])
            if row[0] <= lim:
                s += dist(p1, p2)
            else:
                temp_dict[lim] = [lim, s]
                s = 0
                times_idx += 1
                lim = times[times_idx]
            p1 = p2
        for i in range(times_idx, len(times)):
            lim = times[i]
            temp_dict[lim] = [lim, 0]
        # scaler = MinMaxScaler(feature_range=(0, 1))
        temp_df = pd.DataFrame(temp_dict).astype("float").transpose()
        # dataset = scaler.fit_transform(np.array(temp_df.iloc[:, -1]).reshape(-1, 1))
        # temp_df.iloc[:, -1] = dataset.flatten()
        return temp_df
