'''
    returns the top-k neighbors of all users in the dataset in a defaultdict
'''
from collections import defaultdict

def get_knn(ratings, algo, k=10):

    neighbors = defaultdict(list)

    # retrieve all users in the dataset and use set() to remove duplicates
    raw_user_ids = set(ratings.user_id.to_list())

    for uid in raw_user_ids:

        # Retrieve inner id of the user
        user_inner_id = algo.trainset.to_inner_uid(uid)
        # Retrieve inner ids of the nearest neighbors of the user.
        user_neighbors = algo.get_neighbors(user_inner_id, k=k)
        # Convert inner ids of the neighbors raw-ids.
        user_neighbors = (algo.trainset.to_raw_uid(inner_id) 
                            for inner_id in user_neighbors)

        neighbors[uid] = list(user_neighbors)

    return neighbors