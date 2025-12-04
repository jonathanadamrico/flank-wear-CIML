import os
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from lib.data_loader import DataLoader
from lib.feature_engineering import get_features, subtract_features_from_cut01
from scipy.interpolate import PchipInterpolator
import gc

def main(controller_path = '../tcdata/Controller_Data',
        sensor_path = '../tcdata/Sensor_Data',
        output_path = '../work/result.csv',
        ):
    """
    Run evaluation on all controller and sensor datasets, 
    extract features, apply the trained model, and save predictions.

    Output:
    - result.csv saved to /work directory
    """
    
    #lower = 0 # incremental flank wear cannot be negative
    #upper = 30 # physically implausible for incremental flank wear to reach
    cut_01 = [33, 38, 36] # first cut labels
    evalset_list = [1,2,3]
    cut_list = list(range(2, 27))

    l = [[2,3,4,5,6],
     [7,8,9,10,11],
     [12,13,14,15,16],
     [17,18,19,20,21],
     [22,23,24,25,26]]

    value_to_sublist = {val: sublist.index(val) for sublist in l for val in sublist}

    cut01_features = pd.read_csv('eval_features.csv') # first cut extracted features
    loader = DataLoader(controller_path, sensor_path, mod=True, one_cut=False)
    records, features = [], []

    for set_no in evalset_list:
        for cut_no in cut_list:
            #print(f'Processing Cut {cut_no} in Set {set_no}...')
            cut_index = value_to_sublist.get(cut_no, -1)

            if cut_no == 1:    
                sensor_df = loader.get_sensor_data(set_no, cut_no)
            elif cut_no in [2,7,12,17,22]:
                all_sensor_df = loader.get_sensor_data(set_no, cut_no)
                sensor_df = all_sensor_df[cut_index]
            else:
                sensor_df = all_sensor_df[cut_index]
            controller_df = loader.get_controller_data(set_no, cut_no)
            tmp_features = get_features(controller_df, sensor_df)
            
            records.append([f'evalset_{set_no:02d}', cut_no])
            features.append(tmp_features)

    ext_features = pd.DataFrame(features) 
    
    del features, sensor_df, all_sensor_df, controller_df, tmp_features
    gc.collect()
    
    #======================#
    # Per-tool differencing
    #======================#
    
    features_diff_df = subtract_features_from_cut01(ext_features, evalset_list, cut01_features)
    ext_features = pd.concat([ext_features, 
                     features_diff_df.add_prefix('diff_')], 
                    axis=1)
    
    #===================#
    # Flatten
    #===================#
    n_steps = 5 # cumulative cuts per tool
    n_features = ext_features.shape[1]
    arr = ext_features.to_numpy()

    samples = arr.shape[0] // n_steps
    arr_reshaped = arr.reshape(samples, n_steps, n_features)

    # flatten timesteps into one long feature vector
    X = arr_reshaped.reshape(samples, n_steps * n_features)
    columns = [f"{col}_t{t}" for t in range(n_steps) for col in ext_features.columns]
    flat_df = pd.DataFrame(X, columns=columns)

    #===================#
    # Ridge
    #===================#
    feature_cols = ['AE_skew_t2', 'AE_entropy_t2', 'diff_accel_x_mean_t1', 
                    'accel_x_mean_t4', 'AE_entropy_t3']
    
    with open('model/scalers.pkl', 'rb') as f:
        scalers = pickle.load(f)
    with open('model/y_scalers.pkl', 'rb') as f:
        y_scalers = pickle.load(f)

    preds = []
    for fold in range(6):
        # Load the model
        with open(f'model/ridge_model{fold}.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
        scaled_features = scalers[fold].transform(flat_df[feature_cols])
        pred = loaded_model.predict(scaled_features)
        pred = y_scalers[fold].inverse_transform(pred.reshape(-1,1))
        preds.append(pred)

    # Mean preds
    mean_cumulative_preds = np.mean(preds, axis=0).ravel()

    # Create DataFrame and process results
    result_df = pd.DataFrame(records, columns=['set_num', 'cut_num'])

    #===================#
    # XGB
    #===================#
    mat_features = xgb.DMatrix(ext_features)
    preds = []
    for fold in range(6):
        # Load the model and data
        with open(f'model/xgb_model_{fold}.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
        pred = loaded_model.predict(mat_features, iteration_range=(0, loaded_model.best_iteration + 1))
        preds.append(pred)

    result_df['xgb_pred'] = np.mean(preds, axis=0)

    # Interpolation
    yhat_incs = []
    for set_no in evalset_list:
        x = [1,6,11,16,21,26]
        y = mean_cumulative_preds[(set_no-1)*5:(set_no)*5].tolist()
        y.insert(0, cut_01[set_no - 1])
        
        # PCHIP enforces monotonicity if data is monotonic
        pchip = PchipInterpolator(x, np.cumsum(y))
        yhat = pchip(range(1,27))
        yhat_inc = np.diff(yhat, prepend=0)
        yhat_incs.extend(yhat_inc[1:])

    # Add predictions to the DataFrame
    result_df['ridge_pred'] = yhat_incs
    
    #=====================#
    # Decision Tree Detect
    #=====================#
    feature_cols = ['accel_x_rms','accel_y_rms']

    preds = []
    for fold in range(6):
        # Load the model
        with open(f'model/decisiontree_detect_{fold}.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
        pred = loaded_model.predict(ext_features[feature_cols])
        preds.append(pred)
    
    result_df['decisiontree_detect'] = np.mean(preds, axis=0)
    
    #===================#
    # Decision Tree
    #===================#
    feature_cols = ['accel_x_rms','accel_y_rms']

    preds = []
    for fold in range(6):
        # Load the model
        with open(f'model/decisiontree_model_{fold}.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
        pred = loaded_model.predict(ext_features[feature_cols])
        preds.append(pred)
    
    result_df['decisiontree_pred'] = np.mean(preds, axis=0)
    
    # Ensemble
    result_df['pred'] = np.mean(result_df[['xgb_pred','ridge_pred']], axis=1)
    result_df.loc[result_df['decisiontree_detect']==True, 'pred'] =  result_df['decisiontree_pred']
    
    result_df[['set_num','cut_num','pred']].to_csv(output_path, index=False)  

if __name__ == '__main__':
    main()