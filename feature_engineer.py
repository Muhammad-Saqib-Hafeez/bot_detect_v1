# feature_engineer.py
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        df['ua_lower'] = df['user_agent'].str.lower()
        df['ua_length'] = df['user_agent'].str.len()
        df['has_mozilla'] = df['ua_lower'].str.contains('mozilla', na=False)
        df['path_depth'] = df['path'].str.count('/')
        df['is_api_path'] = df['path'].str.contains('storage|api', na=False)
        df['is_static'] = df['path'].str.contains('js|css|ico', na=False)
        return df