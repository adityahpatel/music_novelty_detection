import yaml
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import yaml
import pickle

if __name__ == '__main__':
    with open("parameters.yaml", "r") as f:
        parameters = yaml.safe_load(f)

    X_train_array = pickle.load(file=open(parameters['artifacts']['training'], "rb"))

    # instantiate and fit the models
    model_LOF = LocalOutlierFactor(n_neighbors=5, novelty=True, metric='cosine').fit(X_train_array)
    model_Isolation_Forest = IsolationForest(random_state=0).fit(X_train_array)

    # dump the 'trained' models as pickle objects
    pickle.dump(model_LOF, file=open('Artifacts/models/model_LOF.pkl', 'wb'))
    pickle.dump(model_Isolation_Forest, file=open('Artifacts/models/model_Isolation_Forest.pkl', 'wb'))
    print('Models are now trained')
