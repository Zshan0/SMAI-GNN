# SMAI-GNN
Codebase and repository for SMAI '21 Project


# Code structure

- `run.py` => Responsible for parsing all the arguments given by the user, and creating the object of the `model`. Also responsible for managing the dataset.
- `model.py`=> The class for GraphCNN model and MLP model


# How to run this model

1. First unzip the datasets
```bash
unzip dataset.zip
```
2. Create a virtual environment
```bash
python3 -m venv venv
```
3. Install the required dependencies
```bash
python3 install -r requirements.txt
```
4. Run the `run.py` file
```bash
python3 run.py
```
