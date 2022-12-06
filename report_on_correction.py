import torch
import math

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sns


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from asses_batch_effect import test_batch_effect_fast


## Function to make a PCA plot and p-value histogram of the batch effect. Takes in a pandas df.
def make_report(data, n_batches, batch_size, train_idx, test_idx, prefix = "", suffix = ""):
    sns.set_style('whitegrid')
    sns.set_palette('Set2')

    y = torch.tensor(data.copy().values)
    row_names = data.index
    col_names = data.columns
    data = pd.DataFrame(StandardScaler().fit_transform(data))
    data.index = row_names
    data.columns = col_names
    data = data.transpose()
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(data)

    sample_colors = [i//batch_size for i in range(batch_size * n_batches)]
    sample_labels = [format(i + 1) for i in range(batch_size * n_batches)]

    pca_plot = sns.scatterplot(x = pca_components[:, 0], y = pca_components[:, 1], 
                               hue = sample_colors, palette = "Set2")
    
    plot_title = prefix + "PCA_plot_by_batch_epoch_" + suffix
    pca_plot.set_title(plot_title)
    pca_plot.set_xlabel('PC1')
    pca_plot.set_xlabel('PC2')
    plt.legend(bbox_to_anchor=(1.02, 1), loc = 2, borderaxespad = 0.)

    for i, label in enumerate(sample_labels):
        pca_plot.text(pca_components[i, 0], pca_components[i, 1], label, 
                      fontsize = 8)

    path = "./pca_plots/" + plot_title + ".png"
    plt.savefig(path, bbox_inches = 'tight')
    plt.clf()
    plt.close()

    p_values = test_batch_effect_fast(y, n_batches, batch_size)
    p_train = p_values[train_idx]
    p_test = p_values[test_idx]
    p_values = np.concatenate([p_train, p_test])
    plot_data = pd.DataFrame({'p_value' : p_values, 
                              'group' : ['train' for i in range(0, len(train_idx))] + ['test' for i in range(0, len(test_idx))]})

    pval_plot = sns.histplot(data = plot_data, x = 'p_value', hue = 'group', stat = 'count')
    # p_values_test = p_values.
    # plt.hist(p_values)
    plot_title = prefix + "pval_hist_epoch_" + suffix
    # plt.title(plot_title)
    pval_plot.set(title = plot_title, xlabel = 'p value', ylabel = 'Count')
    # plt.ylabel('Count')
    # plt.xlabel('p value')
    path = "./p_value_histograms/" + plot_title + ".png"
    # plt.savefig(path)
    pval_plot.get_figure().savefig(path) 

    plt.clf()
    plt.close()

    return "Saved plots"



def batch_density_plot(data, n_batches, batch_size, plot_title, *args):
    plot_title = plot_title + " batch means"
    batches = args

    xx = torch.tensor(data.values)
    xx = xx.reshape(len(xx), n_batches, batch_size).mean(2)
    batch_means = pd.DataFrame(xx.numpy())
    columns = batch_means.columns
    batch_means['feature'] = data.index
    batch_means = batch_means.melt(id_vars = ['feature'], value_vars = batches,
                                  var_name = "batch", value_name = "batch_mean")

    sns.kdeplot(data = batch_means, x = "batch_mean", hue = "batch", 
                cut = 0, fill = True, common_norm = False, alpha = 0.07,
                palette = "Set1").set(title = plot_title)



def correction_scatter(original_data, corrected_data, n_batches, batch_size, alpha = 0.07):
      plt.clf()
      data_tensor = torch.tensor(original_data.values)
      data_tensor = data_tensor.reshape(len(data_tensor), n_batches, batch_size)
      data_means_og = torch.mean(data_tensor, 2)

      data = original_data - corrected_data

      data_tensor = torch.tensor(data.values)
      data_tensor = data_tensor.reshape(len(data_tensor), n_batches, batch_size)
      corrections = torch.mean(data_tensor, 2)

      rows = math.floor(math.sqrt(n_batches))
      cols = math.ceil(n_batches / rows)
      fig, plots = plt.subplots(rows, cols, figsize = (15,10))
      fig.suptitle('Batch Effect scatter plots')

      for i in range(rows):
          for j in range(cols):
              plots[i, j].scatter(data_means_og[:, i*cols + j], corrections[:, i*cols + j], alpha = alpha)
              plots[i, j].set_ylim(-1.5, 1.5)
              plots[i, j].set_xlim(-1.5, 1.5)
              plots[i, j].set_xlabel('Uncorrected batch mean')
              plots[i, j].set_title("Batch " + format(i*cols + j) + " mean vs correction")
              if (j == 0):
                  plots[i, j].set_ylabel('Batch correction')
