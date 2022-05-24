from collections import defaultdict
from scipy import stats

'''
 Function that returns the critical neighborhoods, suitable prediction-based algorithms
'''
def critical_nbhds_accuracy(neighborhoods, predictions_df):
  critical_nbhds_test_1 = defaultdict(list) # mse positive
  critical_nbhds_test_2 = defaultdict(list) # Whelch's t-test
  p_thresh = 0.5 # p-threshold for the t-test, ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html

  for uid, nbhd in list(neighborhoods.items()):
    # get N and D'
    pred_nbhd = predictions_df[predictions_df['uid'].isin(nbhd)]
    pred_nbhd_equiv = predictions_df[~(predictions_df['uid'].isin(nbhd))]

    # calculate the overall loss per neighborhood
    nbhd_loss_n = pred_nbhd.prediction_loss.to_list()
    nbhd_equiv_loss_n = pred_nbhd_equiv.prediction_loss.to_list()

    nbhd_loss = sum(nbhd_loss_n)/len(nbhd_loss_n)
    nbhd_equiv_loss = sum(nbhd_equiv_loss_n)/len(nbhd_equiv_loss_n)

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

  return (critical_nbhds_test_1, critical_nbhds_test_2)
