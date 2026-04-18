import pandas as pd
import numpy as np

def calculate_woe_iv_single_feature(df: pd.DataFrame, feature_col: str, target_col: str = 'target') -> tuple:
    """
    计算单一类别特征的 WOE 映射表和总 IV 值。
    
    参数:
    - df: 包含特征和目标变量的 DataFrame
    - feature_col: 需要计算的特征列名
    - target_col: 目标变量列名 (必须是 0=好, 1=坏)
    
    返回:
    - woe_dict: 字典, {特征取值: 对应的 WOE 值}
    - total_iv: 浮点数, 该特征的整体 Information Value
    - detail_df: DataFrame, 计算过程的详细报表 (用于审核和生成报告)
    """
    
    grouped = df.groupby(feature_col, observed=False)[target_col].agg(['count', 'sum'])
    grouped.columns = ['Total', 'Bad']
    
    grouped['Good'] = grouped['Total'] - grouped['Bad']
    
    total_bad = grouped['Bad'].sum()
    total_good = grouped['Good'].sum()
    
    epsilon = 1e-4
    
    grouped['Good_Dist'] = (grouped['Good'] + epsilon) / (total_good + epsilon)
    grouped['Bad_Dist'] = (grouped['Bad'] + epsilon) / (total_bad + epsilon)
    
    grouped['WOE'] = np.log(grouped['Good_Dist'] / grouped['Bad_Dist'])
    
    grouped['IV'] = (grouped['Good_Dist'] - grouped['Bad_Dist']) * grouped['WOE']
    
    total_iv = grouped['IV'].sum()
    
    woe_dict = grouped['WOE'].to_dict()
    
    return woe_dict, total_iv, grouped

def process_all_features(df: pd.DataFrame, features: list, target_col: str = 'target', iv_threshold: float = 0.02) -> pd.DataFrame:
    """
    批量计算所有特征的 IV 值，并剔除 IV 极低的无效特征。
    """
    iv_summary = {}
    
    print(f"{'特征名称':<25} | {'IV 值':<10} | {'预测能力评估'}")
    print("-" * 60)
    
    for feat in features:
        _, iv, _ = calculate_woe_iv_single_feature(df, feat, target_col)
        iv_summary[feat] = iv
        
        if iv < 0.02:
            eval_str = "无预测力 (建议剔除)"
        elif iv < 0.1:
            eval_str = "弱预测力"
        elif iv < 0.3:
            eval_str = "中等预测力"
        else:
            eval_str = "强预测力 (重点关注)"
            
        print(f"{feat:<25} | {iv:.4f}     | {eval_str}")
        
    return iv_summary

import os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)).replace('src', '')
df_clean = pd.read_parquet(f"{PROJECT_ROOT}/data/german_clean.parquet")

print("\n")
print("[支票账户状态] WOE 计算")
woe_dict, iv, detail_df = calculate_woe_iv_single_feature(df_clean, 'status_checking')
print(detail_df[['Total', 'Good', 'Bad', 'WOE', 'IV']].round(4))
print(f"\n该特征总 IV 值: {iv:.4f}")

print("\n全局类别特征 IV ")
categorical_features = [
    'status_checking', 'credit_history', 'purpose', 'savings_account', 
    'employment_since', 'personal_status_sex', 'other_debtors', 'property', 
    'other_installments', 'housing', 'job', 'telephone', 'foreign_worker'
]

iv_results = process_all_features(df_clean, categorical_features)