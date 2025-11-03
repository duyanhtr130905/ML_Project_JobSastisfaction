"""
Module xử lý tiền xử lý dữ liệu
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def load_data(self, filepath):
        """Đọc dữ liệu từ CSV"""
        df = pd.read_csv(filepath)
        print(f"Data loaded: {df.shape}")
        return df

    def basic_info(self, df):
        """Thống kê cơ bản về dữ liệu"""
        info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum()
        }
        return info

    def handle_missing_values(self, df, strategy='mean'):
        """Xử lý giá trị thiếu"""
        df_clean = df.copy()

        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns

        # Xử lý numeric
        if strategy == 'mean':
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(
                df_clean[numeric_cols].mean()
            )
        elif strategy == 'median':
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(
                df_clean[numeric_cols].median()
            )

        # Xử lý categorical
        df_clean[categorical_cols] = df_clean[categorical_cols].fillna(
            df_clean[categorical_cols].mode().iloc[0]
        )

        return df_clean

    def encode_categorical(self, df, method='onehot'):
        """Encode biến categorical"""
        df_encoded = df.copy()
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns

        if method == 'onehot':
            df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols,
                                        drop_first=True)
        elif method == 'label':
            for col in categorical_cols:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])
                self.label_encoders[col] = le

        return df_encoded

    def normalize_features(self, df, exclude_cols=None):
        """Chuẩn hóa các features"""
        df_normalized = df.copy()

        if exclude_cols is None:
            exclude_cols = []

        numeric_cols = df_normalized.select_dtypes(include=[np.number]).columns
        cols_to_normalize = [col for col in numeric_cols if col not in exclude_cols]

        df_normalized[cols_to_normalize] = self.scaler.fit_transform(
            df_normalized[cols_to_normalize]
        )

        return df_normalized

    def prepare_train_test(self, df, target_col, test_size=0.2, random_state=42):
        """Chia dữ liệu train/test"""
        X = df.drop(target_col, axis=1)
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        return X_train, X_test, y_train, y_test