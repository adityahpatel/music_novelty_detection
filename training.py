import os
import pandas as pd
from pandas import DataFrame

from feature_engineering import feature_engineering
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
for file_name in os.listdir('training_data'):
    file = 'training_data' + '/' + file_name
    df.loc[len(df)] = feature_engineering(file)
