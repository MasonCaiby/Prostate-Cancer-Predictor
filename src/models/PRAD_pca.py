import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import statsmodels.formula.api as smf

# plt.gray()

def scale_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)

def scree_plot(pca, title=None):
    num_components = pca.n_components_
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111)
    # ax.bar(ind, vals, 0.35,
    #        color=[(0.949, 0.718, 0.004)*3,
    #               (0.898, 0.49, 0.016)*47,
    #               (0.863, 0, 0.188),
    #               (0.694, 0, 0.345),
    #               (0.486, 0.216, 0.541),
    #               (0.204, 0.396, 0.667),
    #               (0.035, 0.635, 0.459),
    #               (0.486, 0.722, 0.329),
    #              ])

    rects1 = ax.bar(ind[0:8], vals[0:8], 0.35, color='green')
    rects2 = ax.bar(ind[8:], vals[8:], 0.35, color='black')


    for i in range(4):
        ax.annotate(r"%s%%" % ((str(vals[i]*100)[:4])), (ind[i]+0.2, vals[i]), va="bottom", ha="center", fontsize=12)

    # ax.set_xticklabels(ind, fontsize=12)
    # ax.xaxis.set_ticks(np.arange(0, len(ind), 1))
    ax.yaxis.set_ticks(np.arange(0.00, 0.09, 0.02))

    ax.set_ylim(0, max(vals)+0.02)
    ax.set_xlim(0-0.45, 70+0.45)

    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)

    ax.set_xlabel("Principal Components (10)", fontsize=12)
    ax.set_ylabel("Variance Explained (%)", fontsize=12)

    if title is not None:
        plt.title(title, fontsize=16)


def pca_fit_data(data, components):
    pca = PCA(n_components=components)
    pca.fit(data)
    return pca.transform(data)



if __name__ == '__main__':

    genes_df = pd.read_csv('/Users/meghan/DSI/capstone/Prostate-Cancer-Predictor/data/balanced_data/X_train_all.csv', index_col=0)
    # genes_y
    gene_array = genes_df.values

    scaled_genes = scale_data(gene_array)
    pca_genes = pca_fit_data(scaled_genes, 375)

    pca_gene = PCA(n_components=375)
    pca_gene.fit(scaled_genes)
    var = pca_gene.explained_variance_ratio_
    print (var)
    # scree_plot(pca_gene, title='all 38000 genes')
    # plt.show()
    # plt.close()


    # results = sm.logit(genes_y, cars_with_ones).fit()
    # print(results.summary())

    # pca_with_ones = sm.add_constant(pca_cars)
    # pca_results = sm.OLS(cars_y, pca_with_ones).fit()
    # print(pca_results.summar y())
