# custom class that calculate accuracy at the neighborhood level
# input: surpriselib predictions (after training a model)
# the class is initialized with the model predictions and uses them to map them to all the users in a dictionary

from collections import defaultdict
import pandas as pd
import numpy as np

class NeighborhoodAccuracy:

    def __init__(self, predictions):

        self.neighborhood_mae = defaultdict(list)
        self.neighborhood_rmse = defaultdict(list)
        self.map_users = defaultdict(list)

        # map the predictions to each user
        for uid, iid, true_r, est, _ in predictions:
            self.map_users[uid].append((iid, true_r, est))

    def compute_neighborhood_mae(self, neighbors):

        # append the neighborhood ratings to every user and calculate the mae at each iteration
        for uid, user_ratings in list(self.map_users.items()):
            # work on a copy of the user_ratings list rather to avoid overwriting it
            user_neighbors_ratings = user_ratings.copy()

            for neighbor in neighbors[uid]:
                neighbor_ratings = self.map_users[neighbor]
                user_neighbors_ratings.extend(neighbor_ratings)

            # calculate the mae for every user in the testset (neighborhood cetered at the user if the neighborhood list is not empty)
            mae = np.mean([float(abs(true_r - est))
                    for (_, true_r, est) in user_neighbors_ratings])
                        
            self.neighborhood_mae[uid].extend((mae, len(user_neighbors_ratings)))

        neighborhood_mae_df = pd.DataFrame.from_dict(self.neighborhood_mae, orient='index') \
                        .reset_index() \
                        .sort_values(by=['index']) \
                        .rename({'index' : 'user_id', 0 : 'mae', 1 : 'neighborhood_size'}, axis=1)

        return neighborhood_mae_df

    def compute_neighborhood_rmse(self, neighbors):

        # append the neighborhood ratings to every user and calculate the rmse at each iteration
        for uid, user_ratings in list(self.map_users.items()):
            # work on a copy of the user_ratings list rather to avoid overwriting it
            user_neighbors_ratings = user_ratings.copy()

            for neighbor in neighbors[uid]:
                neighbor_ratings = self.map_users[neighbor]
                user_neighbors_ratings.extend(neighbor_ratings)

            # calculate the rmse for every user in the testset (neighborhood cetered at the user if the neighborhood list is not empty)
            mse = np.mean([float((true_r - est)**2)
                   for (_, true_r, est) in user_neighbors_ratings])
            rmse = np.sqrt(mse)
                        
            self.neighborhood_rmse[uid].extend((rmse, len(user_neighbors_ratings)))

        neighborhood_rmse_df = pd.DataFrame.from_dict(self.neighborhood_rmse, orient='index') \
                        .reset_index() \
                        .sort_values(by=['index']) \
                        .rename({'index' : 'user_id', 0 : 'rmse', 1 : 'neighborhood_size'}, axis=1)
        
        return neighborhood_rmse_df