all: Artifacts/Results/predictions.txt
training: Artifacts/models/*
data: Artifacts/training_data.pkl

Artifacts/Results/predictions.txt: inference.py Artifacts/models/*
	python inference.py

Artifacts/models/*: training.py Artifacts/training_data.pkl
	python training.py

Artifacts/training_data.pkl Artifacts/Unseen_data.pkl: feature_engineering.py Training_data Unseen_test_data
	python feature_engineering.py

Training_data Unseen_test_data: parameters.yaml data_preparation.py
	python data_preparation.py



