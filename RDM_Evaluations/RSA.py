import helper
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import os
from sklearn import linear_model
from scipy.stats import spearmanr


def spearman(rdm1, rdm2):
    """Calculate Spearman"""
    return spearmanr(rdm1, rdm2)[0]


def spearman_correlation(layer_rdm, fmri_rdms):
    """
    Caluclating the R correclation using Spearman correlation.
    mean([spearman correlation (for every fmri rdm and layer rdm)])
    :param layer_rdm: average RDM for a layer of the network shape (18x18)
    :param fmri_rdms: the fmri RDM of a brain area shape (20x18x18)
    :return: np.mean(corr) : mean of the correlations of the network layer rdm to all fmri RDMs of one brain region
    """
    corr = []
    layer_rdm = helper.remove_diagonal(helper.get_lower_triangular(layer_rdm))
    layer_rdm = layer_rdm.reshape(-1, 1)

    for fmri_rdm in fmri_rdms:
        fmri_rdm = (helper.remove_diagonal(helper.get_lower_triangular(fmri_rdm))).reshape(-1, 1)
        corr.append(spearman(layer_rdm,fmri_rdm))
    return np.mean(corr)

def squared_correlation(layer_rdm, fmri_rdms):
    """
    Calculating the squared R by using linear regression.
    mean(square([linear regression(For every fmri rdm and the layer rdm)]))
    :param layer_rdm: average RDM for a layer of the network shape (18x18)
    :param fmri_rdms: the fmri RDM of a brain area shape (20x18x18)
    :return: np.mean(corr) : mean of the squared linear regression correlation
    """
    corr = []
    layer_rdm = helper.remove_diagonal(helper.get_lower_triangular(layer_rdm))
    layer_rdm = layer_rdm.reshape(-1, 1)

    for fmri_rdm in fmri_rdms:
        lr = linear_model.LinearRegression()
        fmri_rdm = (helper.remove_diagonal(helper.get_lower_triangular(fmri_rdm))).reshape(-1, 1)
        lr.fit(layer_rdm, fmri_rdm)
        corr.append(lr.score(layer_rdm, fmri_rdm))

    corr_squared = np.square(corr)
    return np.mean(corr)


def visualize_rsa_matrix(result_dict,layer_list, option,correlation):
    """
    Visualizing the RSA results as a heatmap.
    :param result_dict: Result of the RSA as a dictionary. Keys are the brain regions and values are the RSA float results
    representing a layer that was compared to the brain region: "IPS12" : [conv1,conv2,...]
    :param layer_list: List of the layer names ["conv1","conv2", ...]
    :param option: String representing the task : "taskBoth" / "taskNum" / "taskSize"
    :param correlation: integer 1-2 : 1 = R^2;  2 = Spearman R
    :return: heatmap : heatmap that was generated
    """
    rsa_matrix = []
    brain_regions = ["IPS15", "V3ABV7", "V13", "IPS345", "IPS12", "IPS0", "V3AB", "V3", "V2", "V1"]

    for brain_region in brain_regions:
        rsa_matrix.append(result_dict[brain_region])
    rsa_matrix = np.array(rsa_matrix)
    # Visualize
    layers = layer_list

    plt.title("brain - network RDM similarity: " + option, fontsize = 15)
    if correlation == 1:
        cbar = {'label': 'R2 Score from linear regression'}
    if correlation == 2:
        cbar = {'label': 'R Score from Spearman correlation'}
    heatmap = sns.heatmap(rsa_matrix, xticklabels=layers, yticklabels=brain_regions, cmap="inferno"
        ,cbar_kws = cbar)   #,cbar_kws={'label': 'Spearman correlation coefficient'}) #cbar_kws={'label': 'R2 Score from linear regression'})#
    heatmap.figure.axes[-1].yaxis.label.set_size(13)

    return heatmap


def create_rsa_matrix(option, result_path):
    """
    Heatmap calculation
    :param option: Integer 0,1,2 representing the different options for the tasks taskBoth, taskNum , taskSize
    :param result_path: Path of the directory where the network results are stored e.g. "Alexnet pretrained results"
    :return: Nothing , creates a heatmap of the RSA in resultpath / results / (correlation method)
    """

    # Creating a dictionary for the RDMs of the brain regions depending on the option

    brain_rdms = helper.get_brain_regions_npz(option)
    # Creating a dictionary for the RDMs of the layers from the network
    sep = helper.check_platform()
    path = result_path + sep + "sub04"
    layer_list = helper.get_layers_ncondns(path)[1]

    # Calculating an result dictionary

    result_dict_squared_regression = {}
    result_dict_spearman_correlation = {}
    for brain_region in brain_rdms:
        brain_rdms[brain_region] = brain_rdms[brain_region].f.arr_0
        brain_rdm = brain_rdms[brain_region]
        result_dict_squared_regression[brain_region] = []
        result_dict_spearman_correlation[brain_region] = []

        for layer in layer_list:
            network_path = result_path + sep + "average_rdms" + sep + layer + ".npz"
            network_rdm = helper.loadnpz(network_path)
            network_rdm = network_rdm.f.arr_0
            rsa_result_spearman = spearman_correlation(network_rdm,brain_rdm)
            rsa_result_squared = squared_correlation(network_rdm , brain_rdm)
            result_dict_spearman_correlation[brain_region].append(rsa_result_spearman)
            result_dict_squared_regression[brain_region].append(rsa_result_squared)

    sep = helper.check_platform()

    directory = result_path + sep + "RDM_Evaluation_Results" + sep +"R2_squared_linear_regression"
    if not os.path.exists(directory):
        os.makedirs(directory)

    save_path = (result_path + sep + "RDM_Evaluation_Results" + sep +"R2_squared_linear_regression" + sep + "RSA" + "_" +
                 ["taskBoth", "taskNum", "taskSize"][option])
    visualize_rsa_matrix(result_dict_squared_regression, layer_list, ["taskBoth", "taskNum", "taskSize"][option],1)

    plt.savefig(save_path,bbox_inches='tight')
    plt.close()

    directory = result_path + sep + "RDM_Evaluation_Results" + sep + "R_Spearman"
    if not os.path.exists(directory):
        os.makedirs(directory)
    save_path = (result_path + sep + "RDM_Evaluation_Results" + sep + "R_Spearman" + sep + "RSA" + "_" + ["taskBoth", "taskNum", "taskSize"][option])
    visualize_rsa_matrix(result_dict_spearman_correlation, layer_list, ["taskBoth", "taskNum", "taskSize"][option],2)

    plt.savefig(save_path,bbox_inches='tight')
    plt.close()