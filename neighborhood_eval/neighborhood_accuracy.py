import numpy as np
import pandas as pd
from scipy import stats
from math import sqrt
from collections import defaultdict


# Function that returns the critical neighborhoods, suitable prediction-based algorithms
# p-threshold for the t-test, ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
def critical_nbhds_accuracy(neighborhoods, predictions_df, p_thresh=0.5):
    critical_nbhds_test_1 = defaultdict(list)  # mse positive
    critical_nbhds_test_2 = defaultdict(list)  # Whelch's t-test

    for uid, nbhd in list(neighborhoods.items()):
        # get N and D'
        pred_nbhd = predictions_df[predictions_df['uid'].isin(nbhd)]
        pred_nbhd_equiv = predictions_df[~(predictions_df['uid'].isin(nbhd))]

        # calculate the overall loss per neighborhood
        nbhd_loss_n = pred_nbhd.prediction_loss.to_list()
        nbhd_equiv_loss_n = pred_nbhd_equiv.prediction_loss.to_list()

        nbhd_loss = sum(nbhd_loss_n) / len(nbhd_loss_n)
        nbhd_equiv_loss = sum(nbhd_equiv_loss_n) / len(nbhd_equiv_loss_n)

        # if test-1 passes for a neighborhood N, proceed it to test-2
        if (nbhd_loss - nbhd_equiv_loss) > 0:
            # store all N that pass test-1
            critical_nbhds_test_1[uid] = nbhd

        # apply test-2 - Welch's t-test
        wtt = stats.ttest_ind(pred_nbhd.est.to_list(), pred_nbhd_equiv.est.to_list(), equal_var=False)

        # if test-1 passes for a neighborhood N, report the final result and monitor metric performance
        if wtt.pvalue > p_thresh:
            # variables used to calculate accuracy metrics
            y_true = np.array(pred_nbhd.r_ui.to_list())
            y_pred = np.array(pred_nbhd.est.to_list())
            y_true_equiv = np.array(pred_nbhd_equiv.r_ui.to_list())
            y_pred_equiv = np.array(pred_nbhd_equiv.est.to_list())

            # mse
            mse_subset = np.mean((y_true - y_pred)**2)
            mse_subset_equiv = np.mean((y_true_equiv - y_pred_equiv)**2)

            # mae
            mae_subset = np.mean(np.abs(y_true - y_pred))
            mae_subset_equiv = np.mean(np.abs(y_true_equiv - y_pred_equiv))

            # rmse
            rmse_subset = sqrt(np.mean((y_true - y_pred)**2))
            rmse_subset_equiv = sqrt(np.mean((y_true_equiv - y_pred_equiv)**2))

            critical_nbhds_test_2[uid] = (
                (nbhd, len(pred_nbhd), len(pred_nbhd_equiv),
                mse_subset, mse_subset_equiv,
                mae_subset, mae_subset_equiv,
                rmse_subset, rmse_subset_equiv))

    # convert critical_nbhds after test2 to a df and return the result
    critical_nbhds_final_df = pd.DataFrame(critical_nbhds_test_2).T.rename(
        {
            0: 'nbhd',
            1: 'nbhd_size',
            2: 'equiv_size',
            3: 'mse_nbhd',
            4: 'mse_equiv',
            5: 'mae_nbhd',
            6: 'mae_equiv',
            7: 'rmse_nbhd',
            8: 'rmse_equiv'
        }, axis=1).reset_index(drop=True)
    critical_nbhd_stats(critical_nbhds_test_1, critical_nbhds_final_df, neighborhoods)

    return critical_nbhds_final_df


def critical_nbhd_stats(critical_nbhds_test_1, critical_nbhds, nbhd_clusters):
    print('Clustering method used: PCC')
    print('total nbhds - test1:', len(critical_nbhds_test_1))
    print('total nbhds - test2:', len(critical_nbhds))

    critical_nbhds['mask-1'] = np.select([(critical_nbhds['mse_nbhd'] > critical_nbhds['mse_equiv'])], ['true'])
    print('total nbhds - test3 - MSE:', len(critical_nbhds[(critical_nbhds['mask-1'] == 'true')]))

    critical_nbhds['mask-2'] = np.select([(critical_nbhds['mae_nbhd'] > critical_nbhds['mae_equiv'])], ['true'])
    print('total nbhds - test3 - MAE:', len(critical_nbhds[(critical_nbhds['mask-2'] == 'true')]))

    critical_nbhds['mask-3'] = np.select([(critical_nbhds['rmse_nbhd'] > critical_nbhds['rmse_equiv'])], ['true'])
    print('total nbhds - test3 - RMSE:', len(critical_nbhds[(critical_nbhds['mask-3'] == 'true')]))
    print('Critical nbhd %', round(len(critical_nbhds) / len(nbhd_clusters) * 100, 2))
    print('\n')
