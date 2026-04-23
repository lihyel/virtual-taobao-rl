from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pickle


def main():
    states = np.load("data/states.npy")
    actions = np.load("data/actions.npy")


    knn_model = KNeighborsRegressor(n_jobs=-1)
    knn_model.fit(states, actions)

    with open("models/knn.pkl", "wb") as f:
        pickle.dump(knn_model, f)

    rf_model = RandomForestRegressor(n_jobs=-1)
    rf_model.fit(states, actions)

    with open("models/rf.pkl", "wb") as f:
        pickle.dump(rf_model, f)
    

if __name__ == '__main__':
    main()

