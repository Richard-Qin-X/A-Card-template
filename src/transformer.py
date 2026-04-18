import pandas as pd
import numpy as np
from src.binning import RobustDecisionTreeBinner
from src.woe_transformer import calculate_woe_iv_single_feature

class GlobalWoeTransformer:
    def __init__(self, max_bins=4, min_samples_leaf=0.05):
        self.max_bins = max_bins
        self.min_samples_leaf = min_samples_leaf
        self.woe_maps = {}
        self.iv_values = {}
        self.binners = {}
        self.num_cols = []
        self.cat_cols = []

    def fit(self, df, target_col='target'):
        features = [c for c in df.columns if c != target_col]
        self.num_cols = df[features].select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols = df[features].select_dtypes(exclude=[np.number]).columns.tolist()

        for col in self.num_cols:
            binner = RobustDecisionTreeBinner(max_bins=self.max_bins, min_samples_leaf=self.min_samples_leaf)
            binner.fit(df, col, target_col)
            self.binners[col] = binner
            
            temp_binned = binner.transform(df, col)
            woe_map, iv, _ = calculate_woe_iv_single_feature(pd.concat([temp_binned, df[target_col]], axis=1), col, target_col)
            self.woe_maps[col] = woe_map
            self.iv_values[col] = iv

        for col in self.cat_cols:
            woe_map, iv, _ = calculate_woe_iv_single_feature(df, col, target_col)
            self.woe_maps[col] = woe_map
            self.iv_values[col] = iv
        
        return self

    def transform(self, df):
        df_woe = pd.DataFrame(index=df.index)
        
        for col in self.num_cols:
            binned_series = self.binners[col].transform(df, col)
            df_woe[col] = binned_series.map(self.woe_maps[col]).astype(float)
            
        for col in self.cat_cols:
            df_woe[col] = df[col].map(self.woe_maps[col]).astype(float)
            
        return df_woe

    def get_iv_report(self):
        """输出 IV 报表用于特征筛选"""
        report = pd.Series(self.iv_values).sort_values(ascending=False)
        return report