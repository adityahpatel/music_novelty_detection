import tensorflow as tf, keras, pretty_midi, os
from keras.layers import Conv1D, Dense, MaxPooling1D, Flatten, Dropout, LSTM
from keras.layers import Conv1D, Dense, MaxPool1D, Flatten
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

