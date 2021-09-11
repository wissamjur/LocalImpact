from collections import defaultdict
import pandas as pd
import math

'''
    - The n parameter allows us to control the top-n ratings to be considered.
        This allows us to calculate NDCG@5 or NDCG@10 for instance
    - But, if we have neighborhoods in the neighborhood parameter, the we cannot consider NDCG@n.
        That would defeat the purpose of our neighborhood evaluation method
        It's better to consider NDCG of the whole user/user+neighborhood when considering neighborhood metric evaluation
'''

def compute_neighborhood_ndcg(predictions, neighbors, n=10000):
    neighborhood_ndcg = defaultdict(list)
    map_users = defaultdict(list)
    top_n = defaultdict(list)
    top_n_real = defaultdict(list)

    # map the predictions to each user (predictions of the user-item in the testset)
    for uid, iid, true_r, est, _ in predictions:
        map_users[uid].append((iid, true_r, est))

    # then sort the predictions for each user and retrieve the k highest ones
    # this will give us the detailed prediction data for every user already sorted in a top-n manner
    for uid, user_ratings in list(map_users.items()):

        # work on a copy of the user_ratings list rather to avoid overwriting it
        # the copy won't work unless we use list.copy() method
        user_neighbors_ratings = user_ratings.copy()

        # if neighborhood evaluation is to be considered (k>1), append the ratings of the nieghbors in user_ratings
        for neighbor in neighbors[uid]:
            neighbor_ratings = map_users[neighbor]
            user_neighbors_ratings.extend(neighbor_ratings)

        # sort by order of highest prediction
        user_neighbors_ratings.sort(key=lambda x: x[2], reverse=True)
        top_n[uid] = user_neighbors_ratings[:n]

        # sort by order of highest actual rating (real values)
        user_neighbors_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n_real[uid] = user_neighbors_ratings[:n]

    for uid, user_ratings in top_n.items():

        dcg = 0
        idcg = 0

        i = 1 # used to define the index (rank of the item in the list of recommended items)
        for (iid, true_r, est) in user_ratings:

            # DCG calculation
            utility = pow(2, true_r) - 1
            discount_factor = math.log2(i + 1)

            dcg += utility / discount_factor
            i += 1

            # IDCG calculation
                # an additional one is added (index + 1) since the index starts at 0 as the 1st position
            search_top_n_real = [index + 1 for index, v in enumerate(top_n_real[uid]) if v[0] == iid] # find the location of the item in the ground-truth rankings
            if not search_top_n_real:
                # print("item not found in the ground-truth rankings, using lowest values instead")
                search_top_n_real = n + 1
                utility = 1
                item_index_in_gt = search_top_n_real # item index in ground truth ratings
            else:
                item_index_in_gt = search_top_n_real[0]

            discount_factor_real = math.log2(item_index_in_gt + 1)
            idcg += utility / discount_factor_real

        neighborhood_ndcg[uid].extend((dcg/idcg, len(user_ratings)))

    neighborhood_ndcg_df = pd.DataFrame.from_dict(neighborhood_ndcg, orient='index') \
                            .reset_index() \
                            .sort_values(by=['index']) \
                            .rename({'index' : 'user_id', 0 : 'ndcg', 1 : 'neighborhood_size'}, axis=1)

    return neighborhood_ndcg_df