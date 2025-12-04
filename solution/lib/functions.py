#functions.py

import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, welch


def bandpass(sig, fs, low=10, high=10000, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, sig)
	
def softmax(z):
    exp_z = np.exp(z - np.max(z))  # subtract max for numerical stability
    return exp_z / np.sum(exp_z)


def scaled_softmax(df, df1):
    """
    Applies softmax over every 5 rows of df, then scales
    so that the sum of each block equals the corresponding df1 value.
    
    Args:
        df (pd.DataFrame): Input dataframe with 150 rows.
        df1 (pd.DataFrame or pd.Series): 30 rows, each corresponding to a 5-row block in df.
        
    Returns:
        pd.DataFrame: 150-row dataframe with scaled softmax values.
    """
    df = df.copy()
    result = []
    num_rows = len(df)
    
    # Check input assumptions
    assert num_rows % 5 == 0, "df must have rows divisible by 5"
    assert len(df1) == num_rows // 5, "df1 must match number of 5-row groups"
    
    for i in range(0, num_rows, 5):
        block = df.iloc[i:i+5]
        block_values = block.to_numpy()
        
        # Softmax over each column
        #exp_block = np.exp(block_values - np.max(block_values, axis=0))
        #softmax_block = exp_block / np.sum(exp_block, axis=0)
        softmax_block = softmax(block_values)
        
        # Scale so each column sums to df1's value
        scale = df1.iloc[i // 5].to_numpy() if isinstance(df1, pd.DataFrame) else df1.iloc[i // 5]
        scaled_block = softmax_block * scale
        
        result.append(pd.DataFrame(scaled_block, columns=df.columns, index=block.index))
    
    return pd.concat(result)  
    
def scaled_cumulative(df_, df1):
    """
    Takes the sum over every 5 rows of df, then scales
    so that the sum of each block equals the corresponding df1 value.
    
    Args:
        df (pd.DataFrame): Input dataframe with 150 rows.
        df1 (pd.DataFrame or pd.Series): 30 rows, each corresponding to a 5-row block in df.
        
    Returns:
        pd.DataFrame: 150-row dataframe with scaled values.
    """
    df = df_.copy()
    result = []
    num_rows = len(df)
    df_columns = df.columns if isinstance(df, pd.DataFrame) else [df.name or 'value']
    
    # Check input assumptions
    assert num_rows % 5 == 0, "df must have rows divisible by 5"
    assert len(df1) == num_rows // 5, "df1 must match number of 5-row groups"
    
    for i in range(0, num_rows, 5):
        block = df.iloc[i:i+5]
        block_values = block.to_numpy()
        
        # Normalize
        norm_block = block_values / sum(block_values)
        
        # Scale so each column sums to df1's value
        scale = df1.iloc[i // 5].to_numpy() if isinstance(df1, pd.DataFrame) else df1.iloc[i // 5]
        scaled_block = norm_block * scale
        
        result.append(pd.DataFrame(scaled_block, columns=df_columns, index=block.index))
    
    return pd.concat(result)