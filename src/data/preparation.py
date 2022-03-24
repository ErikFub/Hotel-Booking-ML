import numpy as np
import pandas as pd
import os
from datetime import timedelta


class BookingData(pd.DataFrame):
    def __init__(self, data: pd.DataFrame = None):
        if data is not None:  # if data is passed, initialize with that data
            super().__init__(data)
        else:  # if data is not passed, get data and then initialize
            data = self._get_data()
            super().__init__(data)

    @staticmethod
    def _get_data() -> pd.DataFrame:
        exec_directory = os.getcwd()
        main_directory = "Hotel-Booking-ML"
        main_dir_path = exec_directory[:exec_directory.find(main_directory) + len(main_directory)]
        raw_data_files_location = f"{main_dir_path}/data/raw"
        df = pd.read_csv(f"{raw_data_files_location}/hotel_booking.csv")
        return df

    def prepare(self):
        self._handle_na()
        self._enrich()
        self._drop_cols()
        self._set_dtypes()

    def _handle_na(self):
        self._update_inplace(self[self['children'].notna()])

    def _enrich(self):
        # Getting arrival and booking dates right
        month_mapping = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7,
                         'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
        self['arrival_date_month'] = self['arrival_date_month'].apply(lambda m: month_mapping[m])
        self['arrival_date'] = pd.to_datetime(dict(year=self.arrival_date_year, month=self.arrival_date_month,
                                                   day=self.arrival_date_day_of_month))
        self['booking_date'] = self.apply(lambda r: r['arrival_date'] - timedelta(days=r['lead_time']), axis=1)
        self['booking_date_day_of_week'] = self['booking_date'].dt.weekday

        # Number of Guests
        self['guests'] = self['adults'] + self['children'] + self['babies']

        # Length of stay
        self['stay_total_nights'] = self['stays_in_weekend_nights'] + self['stays_in_week_nights']

    def _drop_cols(self):
        self.drop(columns=['company'], axis=1, inplace=True)

    def _set_dtypes(self):
        self['children'] = self['children'].astype(int)

    def correlation(self):
        return pd.DataFrame(self).corr()

    @property
    def resort(self):
        return self[self['hotel'] == 'Resort Hotel']

    @property
    def city(self):
        return self[self['hotel'] == 'City Hotel']

    @property
    def _constructor(self):
        """Ensures preservation of class when performing operations on an instance of BookingData"""
        return BookingData



if __name__ == "__main__":
    df = BookingData()
    print(df.columns)