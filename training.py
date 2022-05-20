import os
import pandas as pd
import numpy as np
from feature_engineering import feature_engineering


def data_preparation(path: str) -> np.ndarray:
    """
    Converts all MIDI files into a single dataframe with features

    Input       : path to folder containing training or testing data
    Input_type  : String

    Output      : Numpy array with columns as features and rows corresp to a MIDI file
    Output_type : Numpy Array
    """
    feature_names = ['tempo', 'number_beats', 'number_notes', 'number_downbeats', 'percentage_downbeats', 'length',
                     'number_notes_solo', 'number_instruments', 'notes_density', 'percentage_notes_solo',
                     'tempo_change_frequency', 'resolution',
                     'note_duration_mean', 'note_duration_std_dev', 'note_duration_min', 'note_duration_25p',
                     'note_duration_50p', 'note_duration_75p', 'note_duration_max',
                     'note_velocity_mean', 'note_velocity_std_dev', 'note_velocity_min', 'note_velocity_25p',
                     'note_velocity_50p', 'note_velocity_75p', 'note_velocity_max',
                     'note_pitch_mean', 'note_pitch_std_dev', 'note_pitch_min', 'note_pitch_25p', 'note_pitch_50p',
                     'note_pitch_75p', 'note_pitch_max', 'percentage_pitch_class1', 'percentage_pitch_class2',
                     'percentage_pitch_class3', 'percentage_pitch_class4', 'percentage_pitch_class5',
                     'percentage_pitch_class6', 'percentage_pitch_class7', 'percentage_pitch_class8',
                     'percentage_pitch_class9', 'percentage_pitch_class10', 'percentage_pitch_class11',
                     'percentage_pitch_class12']

    df = pd.DataFrame(columns=feature_names)
    for file_name in os.listdir(path):
        df.loc[len(df)] = feature_engineering(path)
    df = df.dropna()
    return df

if __name__ == '__main':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help= 'Enter path of folder containing training or testing data')
    parser.add_argument('output', help= 'Enter name of output csv file. [e.g. name.csv]')
    args = parser.parse_args()
    df = data_preparation(args.input)
    df.to_csv(args.output, index=False)
    X_train = df.to_numpy()
    model_LOF = LocalOutlierFactor(n_neighbors=5, novelty=True, metric='cosine').fit(X_train)
    model_Isolation_Forest = IsolationForest(random_state=0).fit(X_train)


