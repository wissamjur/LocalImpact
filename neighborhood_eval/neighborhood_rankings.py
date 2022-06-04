from collections import defaultdict
from scipy import stats
import pandas as pd


def get_critical_nbhds(neighborhoods, predictions_df, precisions_df, recalls_df, p_thresh=0.5):
    critical_nbhds_test_1 = defaultdict(list)
    critical_nbhds_test_2 = defaultdict(list)

    for uid, nbhd in neighborhoods.items():
        # calculate the precision in N
        prec_nbhd = precisions_df[precisions_df['user_id'].isin(nbhd)]
        prec_uid = precisions_df[precisions_df['user_id'] == uid]
        prec_n = pd.concat([prec_uid, prec_nbhd])
        prec_n_value = sum(prec_n.precision.to_list()) / len(prec_n)

        # calculat the precision in D', the equivalent of N
        prec_equiv = precisions_df[~precisions_df['user_id'].isin(prec_n.user_id.to_list())]
        prec_equiv_value = sum(prec_equiv.precision.to_list()) / len(prec_equiv)

        # if test-1 passes for a neighborhood N, proceed it to test-2
        if (prec_n_value - prec_equiv_value) < 0:
            # store all N that pass test-1
            critical_nbhds_test_1[uid] = nbhd

        # get the predictions of N and D' to apply t-test
        pred_nbhd = predictions_df[predictions_df['uid'].isin(nbhd)]
        pred_uid = predictions_df[predictions_df['uid'] == uid]
        pred_n = pd.concat([pred_nbhd, pred_uid])

        pred_equiv = predictions_df[~predictions_df['uid'].isin(list(set(pred_n.uid.to_list())))]

        # apply test-2 - Welch's t-test
        wtt = stats.ttest_ind(pred_nbhd.est.to_list(), pred_equiv.est.to_list(), equal_var=False)

        # if test-1 passes for a neighborhood N, report the final result and monitor metric performance
        # precision already calculated as part of the base t-test
        if wtt.pvalue > p_thresh:

            # calculate the recall in N and D'
            recall_nbhd = recalls_df[recalls_df['user_id'].isin(nbhd)]
            recall_uid = recalls_df[recalls_df['user_id'] == uid]
            recall_n = pd.concat([recall_uid, recall_nbhd])
            recall_n_value = sum(recall_n.recall.to_list()) / len(recall_n)

            recall_equiv = recalls_df[~recalls_df['user_id'].isin(recall_n.user_id.to_list())]
            recall_equiv_value = sum(recall_equiv.recall.to_list()) / len(recall_equiv)

            f1_n = (2 * prec_n_value * recall_n_value) / (prec_n_value + recall_n_value)
            f1_equiv = (2 * prec_equiv_value * recall_equiv_value) / (prec_equiv_value + recall_equiv_value)

            critical_nbhds_test_2[uid] = (
                (uid, nbhd, len(pred_nbhd), len(pred_equiv),
                prec_n_value, prec_equiv_value,
                recall_n_value, recall_equiv_value,
                f1_n, f1_equiv))

    # convert critical_nbhds after test2 to a df and return the result
    critical_nbhds_final_df = pd.DataFrame(critical_nbhds_test_2).T.rename(
        {
            0: 'uid',
            1: 'nbhd',
            2: 'nbhd_size',
            3: 'equiv_size',
            4: 'precision_nbhd',
            5: 'precision_equiv',
            6: 'recall_nbhd',
            7: 'recall_equiv',
            8: 'f1_nbhd',
            9: 'f1_equiv'
        }, axis=1).reset_index(drop=True)

    print('Percentage of critical neighborhoods:', round(len(critical_nbhds_final_df) / len(neighborhoods) * 100, 2))

    return critical_nbhds_final_df


def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """Return precision and recall at k metrics for each user"""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold)) for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls


def precision_recall_at_k_dfs(predictions, k=10, threshold=3.5):

    users = list(set(predictions.userID.to_list()))
    precisions = dict()
    recalls = dict()

    for uid in users:
        user_df = predictions[predictions['userID'] == uid]
        user_df_sorted = user_df.sort_values(by=['prediction'], ascending=False)
        user_predictions = user_df_sorted.prediction.to_list()
        user_real_ratings = user_df_sorted.rating.to_list()

        user_ratings = list(zip(user_predictions, user_real_ratings))

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold)) for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls
