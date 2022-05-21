
if __name__ == '__main__':
    import argparse
    import os                 # to pass bash shell commands

    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='enter full path of folder containing training set midi files')
    args = parser.parse_args()
    source = args.input + '/*/**'
    os.system("cp %s training_data" % source)

