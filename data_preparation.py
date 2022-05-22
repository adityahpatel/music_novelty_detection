import os

def file_name_formatter(folder: str) -> None:
    """
    Formats the name of the file by removing spaces and adding underscore
    """
    for file_name in os.listdir(folder):
        source = folder + "/" + file_name
        destination = 'training_data' + "/" + file_name.replace(' ', '_')
        os.rename(source, destination)
    print('File names formatting successful')


if __name__ == '__main__':
    import argparse
    import os                 # to pass bash shell commands

    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='enter full path of folder containing training set midi files')
    args = parser.parse_args()
    source = args.input + '/*/**'
    os.system("cp %s training_data" % source)

    file_name_formatter('training_data')


