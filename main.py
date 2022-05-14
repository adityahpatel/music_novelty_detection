import tensorflow as tf, keras, pretty_midi, os
import csv, pretty_midi,re
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.metrics import Precision, Accuracy, AUC, Recall
import warnings
warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", None)
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
from matplotlib.pyplot import figure
from sklearn.metrics import jaccard_score as jaccard_score


def feature_engineering(file_path: str) -> list:
    """
    This function extracts hand crafted features related to beats, pitch/keys and instruments used.

    Input       : Takes as input the path of the MIDI file
    Input_type  : String

    Output      : Outputs a bank of hand crafted features extracted from the MIDI file
    Output_type : List
    """

    try:
        track = pretty_midi.PrettyMIDI(file_path)

        # global tempo - the overall perceived speed of the song
        tempo = track.estimate_tempo()

        number_beats = len(track.get_beats())

        number_downbeats = len(track.get_downbeats())

        percentage_downbeats = number_downbeats / number_beats

        length = track.get_end_time()  # seconds -needed to adjust other features longer song vs shorter song

        number_notes = track.get_onsets().shape[0]  ## needed for note density i.e. number_notes/length

        number_notes_solo = len(set(track.get_onsets()))  # how many notes were not hit simultaneously?

        notes_density = number_notes / length

        percentage_notes_solo = (number_notes - number_notes_solo) / number_notes

        """
        how many times per second does tempo change in this song? 

        """
        tempo_change_frequency = len(track.get_tempo_changes()[0]) / length

        """
        Binning of the frequencies in the whole song into standard 12 MIDI buckets
        """
        percentage_pitch_classes = list(track.get_pitch_class_histogram())

        """
        Resolution is the number of MIDI clocks per quarter note. 
        """

        resolution = track.resolution

        """
        how many instruments were used totaly 0-127 options - midi calls instrument as program
        """
        number_instruments = len(np.unique([instrument.program for instrument in track.instruments]))

        L = []

        for instrument in track.instruments:
            for note in instrument.notes:
                L.append([instrument.program, note.start, note.end, note.end - note.start,
                          note.velocity, note.pitch])
        df = pd.DataFrame(np.array(L))
        df.columns = ['instrument', 'start', 'end', 'duration', 'velocity', 'pitch']

        """
        Which keys are used the most out of 128 keys?  Categorical
        """
        top_keys = df['pitch'].value_counts().index
        first_highest_used_key = pretty_midi.note_number_to_name(top_keys[0])
        second_highest_used_key = pretty_midi.note_number_to_name(top_keys[1])

        """
        Low level Features related to the notes characteristics
        How long was the note played?
        How hard what that note hit?
        What was the frequency of that note (key name or numerical frequency)
        Extract the mean, std_dev, minimum, p25, p50, p75, maximum of above features
        """
        duration, velocity, note_pitch = [], [], []

        note_duration, note_velocity, note_pitch = [], [], []

        descriptor_duration = df['duration'].describe()
        descriptor_velocity = df['velocity'].describe()
        descriptor_pitch = df['pitch'].describe()

        for i in range(1, 8):
            note_duration.append(descriptor_duration[i])
            note_velocity.append(descriptor_velocity[i])
            note_pitch.append(descriptor_pitch[i])

        row_features = [tempo, number_beats, number_notes, number_downbeats, percentage_downbeats, length,
                        number_notes_solo, number_instruments, notes_density, percentage_notes_solo,
                        tempo_change_frequency, resolution, first_highest_used_key, second_highest_used_key,
                        note_duration, note_velocity, note_pitch, percentage_pitch_classes]

        feature_bank = []
        for feature in row_features:
            if type(feature) is list:
                for f in feature:
                    feature_bank.append(f)
            else:
                feature_bank.append(feature)
        return feature_bank
    except:
        pass

def data_preparation(mode: str, path: str) -> np.ndarray:
    """
    Converts all MIDI files into a single dataframe with features

    Input 1     : Mode: One of the strings 'train' or 'test'
    Input_type  : String

    Input 2     : If training set, path to PS1 folder, else for 'test', path to PS2 folder
    Input_type  : String

    Output      : Numpy array with columns as features and rows corresp to a MIDI file
    Output_type : Numpy Array
    """
    feature_names = ['tempo', 'number_beats', 'number_notes', 'number_downbeats', 'percentage_downbeats', 'length',
                     'number_notes_solo', 'number_instruments', 'notes_density', 'percentage_notes_solo',
                     'tempo_change_frequency', 'resolution', 'first_highest_used_key', 'second_highest_used_key',
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

    if mode == 'train':
        composers = ['Bach', 'Beethoven', 'Brahms', 'Schubert']
        for composer in composers:
            root = "MusicNet/PS1/" + composer + "/"
            directory = os.listdir(root)
            for file_name in directory:
                file_path = root + file_name
                df.loc[len(df)] = feature_engineering(file_path)

    elif mode == 'test':
        directory = os.listdir(path)
        for file_name in directory:
            root = path
            file_path = path + file_name
            df.loc[len(df)] = feature_engineering(file_path)

    df = df.drop(columns=['first_highest_used_key', 'second_highest_used_key'])
    df = df.dropna()
    # df_array = MinMaxScaler().fit_transform(df)

    df_array = df.to_numpy()
    return df_array

X_train = data_preparation('train', 'MusicNet/PS1/')
X_test = data_preparation('test', 'MusicNet/PS2/')

def visualize(A:np.array):
    feature_names = ['tempo', 'number_beats', 'number_notes', 'number_downbeats', 'percentage_downbeats','length',
                         'number_notes_solo', 'number_instruments', 'notes_density', 'percentage_notes_solo',
                        'tempo_change_freq','resolution',
                        'note_dur_mean', 'note_dur_std_dev', 'note_dur_min', 'note_dur_25p',
                         'note_dur_50p', 'note_dur_75p', 'note_duration_max',
                         'note_vel_mean', 'note_vel_std_dev', 'note_vel_min', 'note_vel_25p',
                         'note_vel_50p', 'note_vel_75p', 'note_vel_max',
                         'note_pitch_mean', 'note_pitch_std_dev', 'note_pitch_min', 'note_pitch_25p', 'note_pitch_50p',
                         'note_pitch_75p', 'note_pitch_max', 'pitch_class1', 'pitch_class2',
                        'pitch_class3', 'pitch_class4', 'percentage_pitch_class5',
                         'pitch_class6', 'pitch_class7', 'pitch_class8',
                        'pitch_class9', 'pitch_class10', 'pitch_class11',
                        'pitch_class12']

    df = pd.DataFrame(X_train, columns = feature_names)
    plt.figure(figsize=(15, 18))
    for i in range(1,46):
        ax = plt.subplot(9,5,i)
        ax.hist(df[feature_names[i-1]])
        ax.set_title(feature_names[i-1])
    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
    return None
visualize(X_train)


def inference():
    """
    Runs 2 unsupervised learning models on X_test: Local Outlier Factor algorithm and Isolation Forest
    Prints plots for each of models

    Return: None
    """

    model_LOF = LocalOutlierFactor(n_neighbors=5, novelty=True, metric='cosine').fit(X_train)
    model_Isolation_Forest = IsolationForest(random_state=0).fit(
        X_train)  # I assume contamination='auto' as I have no insight

    print('\n------------------- LOF Algorithm X_test predictions    --------------\n')

    print(model_LOF.predict(X_test), end='\n\n')

    print('\n-------------------- Isolation Forest Algorithm X_test predictions    --------------\n')
    print(model_Isolation_Forest.predict(X_test))

    print('\n ------------------ How Similar are the predictions from 2 models? -----------------------\n')

    jaccard_similarity = jaccard_score(model_LOF.predict(X_test), model_Isolation_Forest.predict(X_test))
    print('The models agree on {} % of observations of X_test'.format(round(jaccard_similarity * 100), 3))

    print('\n -----------------------  Novelty Decision Scores Visualized -----------------------\n')
    models = [model_LOF, model_Isolation_Forest]

    plt.figure(figsize=(12, 4))
    for i in [1, 2]:
        ax = plt.subplot(1, 2, i)
        ax.scatter(np.arange(len(X_test)), sorted(models[i - 1].decision_function(X_test)));
        plt.title('LOF Algorithm Decision Scores');
        plt.xlabel('Individual Datapoint instance', fontsize=16)
        plt.ylabel('Decision Score', fontsize=16)
        plt.yticks(fontsize=15)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
        plt.text(13, 0.005, "\n\ndecision threshold", rotation=0)
        if i == 1:
            ax.set_title('LOF Algorithm', fontsize=16)
        else:
            ax.set_title('Isolation Forest Algorithm', fontsize=16)
    return None