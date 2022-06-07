import os                 #  to pass bash shell commands
import argparse
import yaml

def file_name_formatter(folder: str) -> None:
    """
    Formats the name of all files in input folder by replacing spaces with underscore.
    This is useful to prevent errors when issuing command line arguments.
    """
    for file_name in os.listdir(folder):
        source = folder + "/" + file_name
        destination = folder + "/" + file_name.replace(' ', '_')
        os.rename(source, destination)
    print(f"{len(os.listdir(folder))} Midi files are now imported in project folder '{folder}'")

if __name__ == '__main__':
    # This script loads data stored anywhere into the python folder.
    # Takes 2 user inputs of paths where training and unseen test set are loaded

    print('Data Loading begins . . .')
    with open('parameters.yaml', 'r') as f:
        parameters = yaml.safe_load(f)

    # parser = argparse.ArgumentParser()
    # parser.add_argument('user_input_training', help='enter full path of folder containing training set midi files')
    # parser.add_argument('user_input_testing', help='enter full path of folder containing unseen test set midi files')
    # args = parser.parse_args()

    # remove old training data
    os.system("rm -rf Training_data")
    os.system("rm -rf Unseen_test_data")
    os.system("mkdir {Training_data,Unseen_test_data}")

    # assumes training data is in form of folders corresp to each composer
    # assumes testing data is in single folder with no sub-folders
    source_training = parameters['data']['training'] + '/*/**'
    source_testing = parameters['data']['testing'] + '/*'
    os.system("cp %s training_data" % source_training)
    os.system("cp %s Unseen_test_data" % source_testing)

    file_name_formatter('training_data')
    file_name_formatter('Unseen_test_data')
    print('. . . Data loading ends')




