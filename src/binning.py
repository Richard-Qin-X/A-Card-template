import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

class RobustDecisionTreeBinner:
    """
    决策树自动分箱器, 基于信息熵寻找最优分箱边界, 确保每个箱子至少有一定比例的样本.
    适用于连续特征的分箱, 能够有效捕捉非线性关系, 同时通过 min_samples_leaf 参数防止过拟合.
    """
    def __init__(self, max_bins=4, min_samples_leaf=0.05, random_state=42):
        self.max_bins = max_bins
        self.min_samples_leaf = min_samples_leaf 
        self.random_state = random_state
        self.bins_ = None
        self.feature_name_ = None

    def fit(self, df: pd.DataFrame, feature_col: str, target_col: str = 'target'):
        """寻找最优分箱边界"""
        self.feature_name_ = feature_col
        temp_df = df[[feature_col, target_col]].dropna()
        X = temp_df[[feature_col]].values
        y = temp_df[target_col].values

        clf = DecisionTreeClassifier(
            criterion='entropy',
            max_leaf_nodes=self.max_bins,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state
        )
        clf.fit(X, y)

        thresholds = clf.tree_.threshold
        boundaries = np.unique(thresholds[thresholds != -2.0]).tolist()

        self.bins_ = [-np.inf] + boundaries + [np.inf]
        return self.bins_

    def transform(self, df: pd.DataFrame, feature_col: str, return_str=False) -> pd.Series:
        """应用边界进行切分"""
        if self.bins_ is None:
            raise ValueError("错误: 请先调用 fit() 寻找边界。")
        
        binned = pd.cut(df[feature_col], bins=self.bins_, duplicates='drop')
        
        if return_str:
            return binned.astype(str)
        return binned

    def evaluate_bins(self, df: pd.DataFrame, feature_col: str, target_col: str = 'target') -> pd.DataFrame:
        """
        分箱诊断, 评估当前分箱的分布和违约率, 检查是否符合单调性.
        """
        temp_s = self.transform(df, feature_col)
        
        stats = df.groupby(temp_s, observed=False)[target_col].agg(['count', 'sum', 'mean']).reset_index()
        stats.columns = ['分箱区间', '样本量(Total)', '坏样本(Bad)', '违约率(Bad Rate)']
        
        stats['样本占比'] = stats['样本量(Total)'] / stats['样本量(Total)'].sum()
        
        stats['违约率(Bad Rate)'] = stats['违约率(Bad Rate)'].apply(lambda x: f"{x:.2%}")
        stats['样本占比'] = stats['样本占比'].apply(lambda x: f"{x:.2%}")
        
        return stats[['分箱区间', '样本量(Total)', '样本占比', '坏样本(Bad)', '违约率(Bad Rate)']]

import os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)).replace('src', '')
df_clean = pd.read_parquet(os.path.join(PROJECT_ROOT, 'data', 'german_clean.parquet'))
binner = RobustDecisionTreeBinner(max_bins=4, min_samples_leaf=0.05)

bins = binner.fit(df_clean, feature_col='age', target_col='target')
print(f"找到的最优边界: {bins}")

report = binner.evaluate_bins(df_clean, feature_col='age')
print("\n年龄分箱")
print(report.to_string(index=False))