from itertools import cycle, islice
import numpy as np
from sklearn import linear_model
import helper
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy import stats
import pandas as pd
from scipy.stats import spearmanr


def evaluate_fmri_squared_correlation(layer_rdm, fmri_rdms):
    corr = []
    layer_rdm = layer_rdm.reshape(-1, 1)

    for fmri_rdm in fmri_rdms:
        lr = linear_model.LinearRegression()
        fmri_rdm = (helper.remove_diagonal(helper.get_lower_triangular(fmri_rdm))).reshape(-1, 1)
        lr.fit(layer_rdm, fmri_rdm)
        corr.append(lr.score(layer_rdm, fmri_rdm))

    corr_squared = np.square(corr)
    return np.mean(corr), stats.ttest_1samp(corr, 0)[1]


def spearman(rdm1, rdm2):
    """Calculate Spearman"""
    return spearmanr(rdm1, rdm2)[0]


def evaluate_fmri_spearman_correlation(layer_rdm, fmri_rdms):

    corr = []
    for fmri_rdm in fmri_rdms:
        fmri_rdm = helper.remove_diagonal(helper.get_lower_triangular(fmri_rdm))
        corr.append(spearman(layer_rdm, fmri_rdm))

    return np.mean(corr), 0


def get_lowernoiseceiling_squared_correlation(rdm):
    num_subs = rdm.shape[0]
    lnc = []

    for i in range(num_subs):
        sub_rdm = rdm[i,:,:]
        rdm_sub_removed = np.delete(rdm, i, axis=0)
        mean_sub_rdm = np.mean(rdm_sub_removed,axis=0)
        sub_rdm = (helper.remove_diagonal(helper.get_lower_triangular(sub_rdm))).reshape(-1,1)
        mean_sub_rdm = helper.remove_diagonal(helper.get_lower_triangular(mean_sub_rdm)).reshape(-1,1)
        lr = linear_model.LinearRegression()
        lr.fit(sub_rdm,mean_sub_rdm)
        lnc.append(lr.score(sub_rdm,mean_sub_rdm))

    lnc = np.mean(lnc)

    return lnc


def get_lowernoiseceiling_spearman_correlation(rdm):

    # Take the lower noise ceiling
    # 1. Extracting one subject from the overall rdm
    # 2. Take the mean of all the other RDMs
    # 3. Take spearman correlation of subject RDM and mean subject RDM
    # 4. We do this for all the subjects and then calculate the average
    # => Can we predict person 15 from the rest of the subjects?
    # => Low Noise-Ceiling means we need better data
    num_subs = rdm.shape[0]
    lnc = 0.0
    for i in range(num_subs):
        sub_rdm = rdm[i, :, :]
        rdm_sub_removed = np.delete(rdm, i, axis=0)  # remove one person
        mean_sub_rdm = np.mean(rdm_sub_removed, axis=0)  # take mean of other RDMs
        mean_sub_rdm = helper.remove_diagonal(helper.get_lower_triangular(mean_sub_rdm))
        sub_rdm = helper.remove_diagonal(helper.get_lower_triangular(sub_rdm))
        lnc += spearman(sub_rdm, mean_sub_rdm)  # take spearman

    lnc = lnc / num_subs  # average it
    return lnc

def get_uppernoiseceiling_squared_correlation(rdm):
    num_subs = rdm.shape[0]
    unc = []
    for i in range(num_subs):
        sub_rdm = rdm[i,:,:]
        mean_sub_rdm = np.mean(rdm,axis=0)
        sub_rdm = (helper.remove_diagonal(helper.get_lower_triangular(sub_rdm))).reshape(-1,1)
        mean_sub_rdm = (helper.remove_diagonal(helper.get_lower_triangular(mean_sub_rdm))).reshape(-1,1)
        lr = linear_model.LinearRegression()
        lr.fit(sub_rdm,mean_sub_rdm)
        unc.append(lr.score(sub_rdm,mean_sub_rdm))
    unc = np.mean(unc)
    return unc

def get_uppernoiseceiling_spearman_correlation(rdm):
    #Calculate upper noise ceiling
    #1. Take the mean of all RDMs without removing subjects
    #2. Spearman of subject and average RDMs
    #3. Average all of that
    #=> How good are the RDMs generalized
    num_subs = rdm.shape[0]
    unc = 0.0
    for i in range(num_subs):
        sub_rdm = rdm[i, :, :]
        mean_sub_rdm = np.mean(rdm, axis=0)  # take mean
        sub_rdm = helper.remove_diagonal(helper.get_lower_triangular(sub_rdm))
        mean_sub_rdm = helper.remove_diagonal(helper.get_lower_triangular(mean_sub_rdm))
        unc += spearman(sub_rdm, mean_sub_rdm)  # calculate spearman
    unc = unc / num_subs
    return unc

def get_noise_ceiling_fmri_spearman_correlation(target):
    "Function to calculate noise ceilings for fmri scans"

    key_list = []
    for keys, values in target.items():
        key_list.append(keys)

    # lower nc and upper nc
    lnc = get_lowernoiseceiling_spearman_correlation(target[key_list[0]])
    unc = get_uppernoiseceiling_spearman_correlation(target[key_list[0]])

    noise_ceilings = {"lnc": lnc, "unc" : unc}
    return noise_ceilings


def get_noise_ceiling_fmri_squared_correlation(target):
    "Function to calculate noise ceilings for fmri scans"

    key_list = []
    for keys, values in target.items():
        key_list.append(keys)

    # lower nc and upper nc
    lnc = get_lowernoiseceiling_squared_correlation(target[key_list[0]])
    unc = get_uppernoiseceiling_squared_correlation(target[key_list[0]])

    noise_ceilings = {"lnc": lnc, "unc" : unc}
    return noise_ceilings

def scan_result(out,noise_ceiling,squared):
    """Returns final dictionary with r², noiseceiling and sig and the noise ceilings"""

    # Get the noise ceilings
    lnc = noise_ceiling["lnc"]
    unc = noise_ceiling["unc"]

    # Get the keys
    # Its only one key but this way it does not matter what the keys name is
    key_list = []
    for keys, values in out.items():
        key_list.append(keys)
    area_name = key_list[0]

    # Calculate the important values
    # {'IT_RDMs': (0.0012347533900088506, 0.009814439270876047)}
    r2_value = out[area_name][0]
    area_percentNC = ((r2_value) / (lnc)) * 100.  # evc percent of noise ceiling
    if squared:
        sig_value = out[area_name][1]  # Are the results I produce random? Could I get my results with a random RDM?



    # return all the values in a dict
    # {'EVC_RDMs': [0.0007705639149007719, 0.001534065990921806, 4.9209348450876575], [lnc,unc]}
    if squared:
        output_dict = {area_name: [r2_value, area_percentNC, sig_value, [lnc, unc]]}
    if not squared:
        output_dict = {area_name: [r2_value, area_percentNC, 0, [lnc,unc]]}

    return output_dict

def save_as_xlsx(result_dict, layer_names,save_path,squared=False):

    brain_regions = ["IPS15", "V3ABV7", "V13", "IPS345", "IPS12", "IPS0", "V3AB", "V3", "V2", "V1"]
    brain_regions.reverse()
    data = []

    for brain_region in brain_regions:
        for layer in layer_names:
            values = result_dict[layer][brain_region][layer]
            new_data = [brain_region, layer, values[0], values[1], values[2], values[3][0], values[3][1]]
            data.append(new_data)
    df = pd.DataFrame(data, columns=["ROI","network layer"," R²", "noise ceilling %", "significance", "lower noise ceilling", "upper noise ceilling"])

    sep = helper.check_platform()
    save_path_excel = ""
    if squared:
        save_path_excel = (save_path + sep + "RDM_Evaluation_Results" + sep + "R2_squared_linear_regression" + sep +
        "R² and noise ceilling results.xlsx")
    else:
        save_path_excel = (save_path + sep + "RDM_Evaluation_Results" + sep + "R_spearman" + sep +
        "R and noise ceilling results.xlsx")
    df.to_excel(save_path_excel,sheet_name="results", index=False)


def visualize_noise_graph(result_dict,brain_region_noise_dict, save_path, squared):
    """
    Function to visualize the noise graphs.
    :param result_dict:
    :param brain_region_noise_dict:
    :param save_path:
    :param squared:
    :return:
    """
    #Get layernames
    sep = helper.check_platform()
    layer_path = save_path + sep + "sub04"
    layer_names = helper.get_layers_ncondns(layer_path)[1]
    save_path = save_path + sep + "RDM_Evaluation_Results"
    if squared:
        save_path = save_path + sep + "R2_squared_linear_regression"
    elif not squared:
        save_path = save_path + sep + "R_Spearman"

    colours = ['#fbb4ae',
               '#b3cde3',
               '#ccebc5',
               '#decbe4',
               '#fed9a6',
               '#ffffcc',
               '#e5d8bd',
               '#fddaec',
               '#f2f2f2',
               '#b3e2cd',
               '#fdcdac',
               '#cbd5e8',
               '#f4cae4',
               '#e6f5c9',
               '#fff2ae',
               '#f1e2cc',
               '#cccccc']

    colours = list(islice(cycle(colours), 100)) # cicle through color scheme

    # 1.) Visualize the lower and upper noise ceilling of our fmri scans
    noise = brain_region_noise_dict
    x_label = ["IPS15", "V3ABV7", "V13", "IPS345", "IPS12", "IPS0", "V3AB", "V3", "V2", "V1"]
    x_label.reverse()

    lnc_y_values = [noise[x]["lnc"] for x in x_label]
    unc_y_values = [noise[x]["unc"] for x in x_label]
    mpl.style.use("seaborn-paper")
    fig, axs = plt.subplots(1,2)
    fig.set_figheight(5)
    fig.set_figwidth(20)
    axs[0].set_title("Lower noise ceiling")
    axs[0].set(xlabel= "Brain ROIs")
    if squared:
        axs[0].set(ylabel="R² from linear regression")
    elif not squared:
        axs[0].set(ylabel = "Spearman correlation coefficient")
    axs[1].set(xlabel="Brain ROIs")


    axs[1].set_title("Upper noise ceiling")
    for x in range(len(x_label)):
        lnc_plot = axs[0].bar(x_label[x], lnc_y_values[x],color=colours[x])
        unc_plot = axs[1].bar(x_label[x], unc_y_values[x],color=colours[x])

    save_path_noise = save_path + sep + "Lower and upper noise ceilling"
    plt.savefig(save_path_noise)
    plt.close()



    for i, roi in enumerate(x_label):
        plt.clf()

        x_label = layer_names
        lnc = lnc_y_values[i]
        unc = unc_y_values[i]
        r2 = [result_dict[layer][roi][layer][0] for layer in layer_names]
        noise_percentage = [result_dict[layer][roi][layer][1] for layer in layer_names]

        save_path_roi = save_path + sep + "NoiseceillingGraph_" + roi
        fig, axs = plt.subplots(1, 2)
        if squared:
            fig.suptitle("Brainregion: " + roi + "| R² linear regression and Noise ceilling in % ")
        elif not squared:
            fig.suptitle("Brainregion: " + roi + "| R Spearman and Noise ceilling in % ")
        #fig.suptitle("Brainregion: " + roi + "| R and Noise ceilling in % ")
        fig.set_figheight(5)
        fig.set_figwidth(20)
        #axs[0].set_title("R from Spearman")
        if squared:
            axs[0].set_title("R² score from linear regression")
        elif not squared:
            axs[0].set_title("R spearman correlation coefficient")
        axs[0].set(xlabel="Network Layers")
        if squared:
            axs[0].set(ylabel="Colored: R² , Grey: Noiseceilling")
        elif not squared:
            axs[0].set(ylabel="Colored: R , Grey: Noiseceilling")

        axs[1].set(xlabel="Network Layers")
        #axs[1].set(ylabel="Nois")
        axs[0].fill_between
        axs[1].set_title("Noise ceiling in percent")
        nc_plots = []
        for x in range(len(x_label)):
            r2_plot = axs[0].bar(x_label[x], r2[x], color=colours[x])
            nc_plot = axs[1].bar(x_label[x], noise_percentage[x], color=colours[x])
            nc_plots.append(nc_plot)
            width = 0.44
            axs[0].fill_between((x - width / 2 - 0.05, x + width / 2 + 0.05), lnc,
                                unc, color='gray', alpha=0.3)

        axs[0].legend(nc_plots, x_label, ncol=4, bbox_to_anchor=(.6, -0.4, 1., .102),
                        loc='upper center')

        plt.savefig(save_path_roi)

        plt.close()


def noise_ceiling_main(option,network_save_path):
    """

    :param option: Integer from 0-2 representing the different task options : 0 - taskBoth, 1- taskNum , 2- taskSize
    :param network_save_path: direcotrie of the network e.g. "Alexnet pretrained results"
    :return: Nothing , creates multiple graphs into the directories network_save_path/RDM_Evaluation_Results/(Correlation method)
    """

    # Brain regions
    brain_regions = ["IPS15", "V3ABV7", "V13", "IPS345", "IPS12", "IPS0", "V3AB", "V3", "V2", "V1"]

    #Get layernames
    sep = helper.check_platform()
    layer_path = network_save_path + sep + "sub04"
    layer_names = helper.get_layers_ncondns(layer_path)[1]

    #3.) Get noise ceilling for every brain region
    # dictionary to store the results of noise ceilling
    brain_region_noise_dict_squared_correlation = helper.get_brain_regions_npz(option)
    brain_region_noise_dict_spearman_correlation = helper.get_brain_regions_npz(option)
    for brain_region in brain_regions:
        noise = get_noise_ceiling_fmri_squared_correlation(brain_region_noise_dict_squared_correlation[brain_region])
        brain_region_noise_dict_squared_correlation[brain_region] = noise
        noise = get_noise_ceiling_fmri_spearman_correlation(brain_region_noise_dict_spearman_correlation[brain_region])
        brain_region_noise_dict_spearman_correlation[brain_region] = noise


    brain_npz_files = helper.get_brain_regions_npz(option)  #get all
    # final results being saved into as dictonary of dictonaries
    result_dict_squared_correlation = {}    # { layer1 : { brain_region1 : results , ..., brain_region2 : } , layer2 : ... }
    result_dict_spearman_correlation = {}
    for layer in layer_names:
        result_dict_squared_correlation[layer] = {}
        result_dict_spearman_correlation[layer] = {}

        for brain_region in brain_regions:
            layer_path = network_save_path + sep + "average_rdms" + sep + layer + ".npz"
            layer_rdm = helper.loadnpz(layer_path)
            layer_rdm = layer_rdm.f.arr_0
            layer_rdm = helper.remove_diagonal(helper.get_lower_triangular(layer_rdm))
            brain_rdms = brain_npz_files[brain_region].f.arr_0
            result_for_layer_squared = {layer : evaluate_fmri_squared_correlation(layer_rdm,brain_rdms)}
            result_for_layer_spearman = {layer : evaluate_fmri_spearman_correlation(layer_rdm,brain_rdms)}
            result_dict_squared_correlation[layer][brain_region] = scan_result(result_for_layer_squared,
                                                            brain_region_noise_dict_squared_correlation[brain_region],
                                                                               True)
            result_dict_spearman_correlation[layer][brain_region] = scan_result(result_for_layer_spearman,
                                                    brain_region_noise_dict_spearman_correlation[brain_region], False)

    save_as_xlsx(result_dict_squared_correlation,layer_names,network_save_path,squared=True)
    visualize_noise_graph(result_dict_squared_correlation,brain_region_noise_dict_squared_correlation,
                          network_save_path, True)
    save_as_xlsx(result_dict_spearman_correlation,layer_names,network_save_path,squared=False)
    visualize_noise_graph(result_dict_spearman_correlation, brain_region_noise_dict_spearman_correlation,
                          network_save_path, False)