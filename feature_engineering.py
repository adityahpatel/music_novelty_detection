import pretty_midi
import pandas as pd
import numpy as np
import argparse
import pickle
import warnings
import sys
import os
import yaml

def feature_engineering_single_file(file_path: str) -> list:
    """
    This function extracts handcrafted features related to beats, pitch/keys and instruments used.

    Input       : Takes as input the path of the MIDI file
    Input_type  : String

    Output      : Outputs a bank of handcrafted features extracted from the MIDI file
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
                        tempo_change_frequency, resolution, note_duration, note_velocity, note_pitch,
                        percentage_pitch_classes]

        feature_bank = []
        for feature in row_features:
            if type(feature) is list:
                for f in feature:
                    feature_bank.append(f)
            else:
                feature_bank.append(feature)
        return feature_bank
    except Exception as e:
        print(f"ATTENTION: {e} error has occurred")

def feature_engineering_all_files(folder_name: str) -> np.ndarray:
    """
    Each MIDI file is converted into its features.
    A numpy array is returned which contains a subarray corresp to each training MIDI file
    :rtype: np.ndarray
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
    for file_name in os.listdir(folder_name):
        file_path = folder_name + '/' + file_name
        df.loc[len(df)] = feature_engineering_single_file(file_path)
    df = df.dropna()
    array = df.to_numpy()
    return array

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    with open('parameters.yaml', 'r') as f:
        parameters = yaml.safe_load(f)

    # dump the training dataset numpy array into pickle file
    pickle.dump(feature_engineering_all_files(parameters['folders']['training']),
                file=open(parameters['artifacts']['training'], 'wb'))

    # dump the testing dataset numpy array into pickle file
    pickle.dump(feature_engineering_all_files(parameters['folders']['testing']),
                file=open(parameters['artifacts']['testing'], 'wb'))

    print(f"Training set features saved in {parameters['artifacts']['training']}, testing set features saved in \
    {parameters['artifacts']['testing']}")



