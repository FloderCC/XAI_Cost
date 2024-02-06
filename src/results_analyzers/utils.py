from pandas import DataFrame


def divide_dataframe_regarding_mcc(df: DataFrame):
    condition = df['MCC'] > 0.8
    return {
        'lp': df[~condition],
        'hp': df[condition],
    }
