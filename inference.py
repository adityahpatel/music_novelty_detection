import pickle
import os
import yaml

if __name__ == '__main__':
    with open('parameters.yaml', 'r') as f:
        parameters = yaml.safe_load(f)

    # load test set
    X_test_array = pickle.load(file=open(parameters['artifacts']['testing'], "rb"))
    os.system("rm -rf Artifacts/Results")
    os.system("mkdir Artifacts/Results")
    with open(parameters['artifacts']['Results'], 'w+') as f:
        for pickle_file in os.listdir(parameters['artifacts']['models']):
            path = parameters['artifacts']['models'] + '/' + pickle_file
            model = pickle.load(file=open(path, 'rb'))
            f.write(pickle_file + "\n")
            f.write(str(model.predict(X_test_array)) + "\n\n")
    print(f" Success: Results are saved as {parameters['artifacts']['Results']}")



