import pretty_midi
import pandas as pd
import numpy as np


def feature_engineering(file_path: str) -> list:
    """
    This function extracts hand crafted features related to beats, pitch/keys and instruments used.

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


if __name__ == '__main__':
    import argparse
    import pickle
    import warnings
    import sys

    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Enter file path of a single MIDI file (e.g. abc/cd.mid)')
    parser.add_argument('output', help='Enter name to store output (e.g. output.pkl)')
    args = parser.parse_args()

    try:
        assert args.input[-4:] == '.mid'
    except AssertionError:
        print('Invalid file path entered. Must end in .pkl \nStopping execution . . .')
        sys.exit()                                            # without sys.exit(), it will continue further execution

    try:
        assert args.output[-4:] == '.pkl'
    except AssertionError:
        print("Invalid output file name. Must end in .pkl. \nStopping execution . . .")
        sys.exit()

    L = feature_engineering(args.input)
    pickle.dump(L, file=open(args.output, "wb+"))
    print('Code Successfully executed!')
