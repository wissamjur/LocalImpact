'''
    returns the top-k neighbors of all users in the dataset in a defaultdict
'''
from collections import defaultdict

def get_knn(data, clustering_algorithms, nbhd_size=10):

    nbhd_clusters = dict()

    # retrieve all unique users in the dataset
    raw_user_ids = set(data.user_id.to_list())

    for index, algo_nbhd in enumerate(clustering_algorithms):
      nbhds = defaultdict(list)

      for uid in raw_user_ids:
        # Retrieve inner id of the user
        user_inner_id = algo_nbhd.trainset.to_inner_uid(uid)
        # Retrieve inner ids of the nearest neighbors of the user.
        user_neighbors = algo_nbhd.get_neighbors(user_inner_id, k=nbhd_size)
        # Convert inner ids of the neighbors raw-ids.
        user_neighbors = (algo_nbhd.trainset.to_raw_uid(inner_id) 
                            for inner_id in user_neighbors)
        nbhds[uid] = list(user_neighbors)

      nbhd_clusters[index] = nbhds
    
    return nbhd_clusters
