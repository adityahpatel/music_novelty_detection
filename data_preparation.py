# os.system("mv /Users/namita/Desktop/SFL_ProductionGrade/training_data/*/**\
#           /Users/namita/Desktop/SFL_ProductionGrade/training_data")
# os.system("rmdir /Users/namita/Desktop/SFL_ProductionGrade/training_data/*")
import os



if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='enter full path of folder containing training set midi files')
    args = parser.parse_args()
    source = args.input + '/*/**'
    print(source)
    # os.mkdir('training_data')
    os.system("mv %s training_data" % source)
    # os.system("rmdir /Users/namita/Desktop/SFL_ProductionGrade/training_data/*")
