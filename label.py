class Label:
    def __init__(self, subject_id, data):
        """Label class

        Parameters
        ----------
        subject_id : int
        data : pd.DataFrame
        """
        self.subject_id = subject_id
        self.__data = data
        self.__prepared_data = self.__parse_labeled_data()
        self.__start_time, self.__end_time = self.__define_sleep_interval()

    @property
    def prepared_data(self):
        return self.__prepared_data

    @property
    def start_time(self):
        return self.__start_time

    @property
    def end_time(self):
        return self.__end_time

    def __define_sleep_interval(self):
        """Defines start and end times (first and last values from column1) where legit label was found

        Returns
        -------
        tuple
            start and end time
        """
        return self.prepared_data[0].iloc[0], self.prepared_data[0].iloc[-1]

    def __parse_labeled_data(self):
        """Cuts rows from DataFrame where the label is not recognized

        Returns
        -------
        pd.DataFrame
            data with only legit labels
        """
        df = self.__data.loc[(self.__data[1] > -1)]
        df.set_index(0, inplace=True, drop=False)
        return df
