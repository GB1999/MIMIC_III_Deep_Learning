import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
from pathlib import Path

class MIMICDataLoader:
    def __init__(self, data_dir, cache_dir='./cache'):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Define standard column names (lowercase)
        self.std_columns = {
            'subject_id': ['subject_id', 'SUBJECT_ID'],
            'hadm_id': ['hadm_id', 'HADM_ID'],
            'charttime': ['charttime', 'CHARTTIME'],
            'itemid': ['itemid', 'ITEMID'],
            'value': ['value', 'VALUE'],
            'admittime': ['admittime', 'ADMITTIME'],
            'dischtime': ['dischtime', 'DISCHTIME'],
            'hospital_expire_flag': ['hospital_expire_flag', 'HOSPITAL_EXPIRE_FLAG'],
            'gender': ['gender', 'GENDER'],
            'dob': ['dob', 'DOB']
        }
    
    def _get_actual_columns(self, df, required_columns):
        """Map standard column names to actual column names in the DataFrame"""
        column_mapping = {}
        for std_col in required_columns:
            possible_names = self.std_columns[std_col.lower()]
            found_col = next((col for col in possible_names if col in df.columns), None)
            if found_col is None:
                raise ValueError(f"Could not find column {std_col} in DataFrame")
            column_mapping[found_col] = std_col.lower()
        return column_mapping

    def _load_or_cache(self, filename, columns, nrows=1000000):
        cache_path = self.cache_dir / f"{filename.stem}_cached.pkl"
        
        if cache_path.exists():
            print(f"Loading cached {filename.name}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        print(f"Loading {filename.name} from CSV")
        df = pd.read_csv(self.data_dir / filename, nrows=nrows)
        
        # Get actual column names and rename
        column_mapping = self._get_actual_columns(df, columns)
        df = df[list(column_mapping.keys())].rename(columns=column_mapping)
        
        with open(cache_path, 'wb') as f:
            pickle.dump(df, f)
        
        return df
    
    def _clean_numeric_values(self, df, value_column='value'):
        """Clean numeric values by removing non-numeric characters and converting to float"""
        def convert_to_float(x):
            if pd.isna(x):
                return np.nan
            if isinstance(x, (int, float)):
                return float(x)
            try:
                # Remove '%' and other common symbols, then convert to float
                x = str(x).strip()
                x = x.replace('%', '').replace(',', '')
                return float(x)
            except (ValueError, TypeError):
                return np.nan
        
        df[value_column] = df[value_column].apply(convert_to_float)
        return df
    
    def _calculate_time_to_event(self, group_data, admissions_data):
        """Calculate hours remaining until discharge/death for each timestamp"""
        admit_info = admissions_data[
            admissions_data['hadm_id'] == group_data['hadm_id'].iloc[0]
        ].iloc[0]
        
        end_time = pd.to_datetime(admit_info['dischtime'])
        
        # Calculate hours remaining for each timestamp
        hours_remaining = [(end_time - ts).total_seconds() / 3600 
                          for ts in group_data['charttime']]
        
        return np.array(hours_remaining)
    
    def load_data(self, sequence_length=10):
        # Load and cache individual datasets with standardized column names
        chartevents = self._load_or_cache(
            Path('CHARTEVENTS.csv'),
            ['subject_id', 'hadm_id', 'charttime', 'itemid', 'value']
        )
        labevents = self._load_or_cache(
            Path('LABEVENTS.csv'),
            ['subject_id', 'hadm_id', 'charttime', 'itemid', 'value']
        )
        admissions = self._load_or_cache(
            Path('ADMISSIONS.csv'),
            ['subject_id', 'hadm_id', 'admittime', 'dischtime', 'hospital_expire_flag']
        )
        patients = self._load_or_cache(
            Path('PATIENTS.csv'),
            ['subject_id', 'gender', 'dob']
        )

        # Clean numeric values
        chartevents = self._clean_numeric_values(chartevents)
        labevents = self._clean_numeric_values(labevents)

        # Convert timestamps
        for df in [chartevents, labevents]:
            df['charttime'] = pd.to_datetime(df['charttime'])
        admissions['admittime'] = pd.to_datetime(admissions['admittime'])
        admissions['dischtime'] = pd.to_datetime(admissions['dischtime'])
        patients['dob'] = pd.to_datetime(patients['dob'])

        # Process data
        data = admissions.merge(patients, on='subject_id', how='left')
        data = data[data['hadm_id'].isin(chartevents['hadm_id'].unique())]

        vitals = chartevents.merge(
            data[['subject_id', 'hadm_id', 'hospital_expire_flag']],
            on=['subject_id', 'hadm_id'],
            how='inner'
        )
        labs = labevents.merge(
            data[['subject_id', 'hadm_id', 'hospital_expire_flag']],
            on=['subject_id', 'hadm_id'],
            how='inner'
        )

        # Drop rows with NaN values before pivot
        vitals = vitals.dropna(subset=['value'])
        labs = labs.dropna(subset=['value'])

        combined_events = pd.concat([vitals, labs], ignore_index=True)
        pivoted_data = combined_events.pivot_table(
            index=['subject_id', 'hadm_id', 'charttime', 'hospital_expire_flag'],
            columns='itemid',
            values='value',
            aggfunc='first'
        ).reset_index()

        # Handle missing values more carefully
        pivoted_data = pivoted_data.ffill().bfill()  # Forward fill then backward fill
        
        # Drop any remaining rows with NaN values
        pivoted_data = pivoted_data.dropna()
        
        if len(pivoted_data) == 0:
            raise ValueError("No valid data remaining after cleaning and removing NaN values")

        pivoted_data.sort_values(by=['subject_id', 'hadm_id', 'charttime'], inplace=True)

        # Modify the sequence preparation to include time remaining
        X, y = [], []
        grouped = pivoted_data.groupby(['subject_id', 'hadm_id'])
        
        # First, fit the scaler on all feature data
        all_features = pivoted_data.drop(columns=['subject_id', 'hadm_id', 'charttime', 'hospital_expire_flag'])
        self.scaler.fit(all_features.astype(float))
        
        for _, group in grouped:
            group_features = group.drop(columns=['subject_id', 'hadm_id', 'charttime', 'hospital_expire_flag'])
            time_remaining = self._calculate_time_to_event(group, admissions)
            
            # Now transform the group features using the fitted scaler
            group_features_scaled = self.scaler.transform(group_features.astype(float))
            
            if len(group_features_scaled) > sequence_length:
                for i in range(len(group_features_scaled) - sequence_length):
                    X.append(group_features_scaled[i:i + sequence_length])
                    y.append(time_remaining[i + sequence_length])
        
        if not X:
            raise ValueError("No sequences could be created. Check if the data has enough consecutive timestamps.")
        
        # Normalize the time remaining values
        y = np.array(y)
        y_mean = np.mean(y)
        y_std = np.std(y)
        y_normalized = (y - y_mean) / y_std
        
        print(f"\nTarget variable statistics:")
        print(f"Mean time remaining: {y_mean:.2f} hours")
        print(f"Std time remaining: {y_std:.2f} hours")
        print(f"Min time remaining: {np.min(y):.2f} hours")
        print(f"Max time remaining: {np.max(y):.2f} hours\n")
        
        return np.array(X), y_normalized, (y_mean, y_std) 