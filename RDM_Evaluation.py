from itertools import cycle, islice
import numpy as np
import glob
import os
from scipy.stats import spearmanr
from sklearn import linear_model
import helper
from scipy.spatial.distance import squareform
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

def remove_diagonal(rdm_lower_triangle):
    """Removes the diagonal of zeros and one value before the zero from the lower triangle of the rdm"""

    index = 1
    counter = 2
    rdm_lower_triangle = np.delete(rdm_lower_triangle, 0)
    loops = 1

    while loops != 18:
        rdm_lower_triangle = np.delete(rdm_lower_triangle, index)
        rdm_lower_triangle = np.delete(rdm_lower_triangle, index)
        index += counter
        counter += 1
        loops += 1

    not_double_please = []


    return rdm_lower_triangle


def RSA_spearman(rdm1, rdm2):
    """Calculate Spearman"""
    return spearmanr(rdm1, rdm2)[0]


def get_lowertriangular(rdm):
    """Get lower triangle of a RDM"""
    num_conditions = rdm.shape[0]
    return rdm[np.tril_indices(num_conditions, 1)]


def compare_rdms(network_rdm, brain_rdms):
    """Compares two RDM using only the lower triangle and Spearman-correlation"""

    
    # Spearman Correlation group averaged
    # Load npz files from given path
    if type(network_rdm) != np.ndarray:
        loadednpz1 = helper.loadnpz(network_rdm)
        loadednpz2 = helper.loadnpz(brain_rdms)
        rdm1 = loadednpz1.f.arr_0
        rdm2 = loadednpz2.f.arr_0
    else:
        rdm1 = network_rdm
        rdm2 = brain_rdms

    # Get lower triangle of rdm
    rdm1 = get_lowertriangular(rdm1)
    rdm2 = get_lowertriangular(rdm2)

    # Remove diagonal and values above the diagonal
    rdm1 = remove_diagonal(rdm1)
    rdm2 = remove_diagonal(rdm2)

    result = RSA_spearman(rdm1, rdm2)

    #Spearman Correlation not group averaged 
    network_rdm = remove_diagonal(get_lowertriangular(network_rdm))
    brain_rdms = [remove_diagonal(get_lowertriangular(brain_rdm)) for brain_rdm in brain_rdms]
    result = [RSA_spearman(network_rdm, brain_rdm) for brain_rdm in brain_rdms]
    result = np.mean(result)

    """

    # Option linear regression R^2
    corr = []
    layer_rdm = (remove_diagonal(get_lowertriangular(network_rdm))).reshape(-1, 1)
    for fmri_rdm in brain_rdms:
        fmri_rdm = remove_diagonal(get_lowertriangular(fmri_rdm)).reshape(-1,1)
        lr = linear_model.LinearRegression()
        lr.fit(layer_rdm, fmri_rdm)
        corr.append(lr.score(layer_rdm,fmri_rdm))

    corr_squared = np.square(corr)
    result = np.mean(corr_squared)

    return result
    """

def z_transform_matrix(lower_triangle_matrix):
    """Z-transform every value inside array"""
    mean_value = np.mean(lower_triangle_matrix)
    standard_deviation = np.std(lower_triangle_matrix)

    new_array = []
    for value in lower_triangle_matrix:
        new_value = (value - mean_value) / standard_deviation
        new_array.append(new_value)

    new_array = np.asarray(new_array)

    return new_array


def multiple_regression(rdm):
    """Perform multiple regression to yield beta weights"""
    # Load npz file from given path
    loaded_npz = rdm

    # Load predictor matrices
    sep = helper.check_platform()
    predictor_npz_1 = helper.loadnpz("RSA_matrices" + sep  + "all_RDM_predictors.npz")

    # Extract array (rdm) from npz
    if isinstance(rdm,np.lib.npyio.NpzFile):
        rdm = loaded_npz.f.arr_0

    predictor_rdms = predictor_npz_1.f.arr_0
    predictor1 = predictor_rdms[0]
    predictor2 = predictor_rdms[1]
    predictor3 = predictor_rdms[2]
    predictor4 = predictor_rdms[3]
    predictor5 = predictor_rdms[4]

    # Step one get lower triangular and remove diagonal
    rdm = get_lowertriangular(rdm)
    rdm = remove_diagonal(rdm)

    predictor1 = get_lowertriangular(predictor1)
    predictor1 = remove_diagonal(predictor1)

    predictor2 = get_lowertriangular(predictor2)
    predictor2 = remove_diagonal(predictor2)

    predictor3 = get_lowertriangular(predictor3)
    predictor3 = remove_diagonal(predictor3)

    predictor4 = get_lowertriangular(predictor4)
    predictor4 = remove_diagonal(predictor4)

    predictor5 = get_lowertriangular(predictor5)
    predictor5 = remove_diagonal(predictor5)

    # Z-transform
    rdm = z_transform_matrix(rdm)
    predictor1 = z_transform_matrix(predictor1)
    predictor2 = z_transform_matrix(predictor2)
    predictor3 = z_transform_matrix(predictor3)
    predictor4 = z_transform_matrix(predictor4)
    predictor5 = z_transform_matrix(predictor5)

    x = np.array([predictor1, predictor2, predictor3, predictor4, predictor5]).transpose()
    clf = linear_model.LinearRegression().fit(x,rdm)

    return clf.coef_


def read_in_npz_files(path):
    """This function reads in all npz files from a path into an dictionary"""
    #npz_files = glob.glob(path + "/**/*.npz", recursive=True)
    npz_files = glob.glob(path + "/*.npz")

    loaded_npz_dict = {}
    for npz_file in npz_files:
        sep = helper.check_platform()
        filename = npz_file.split(sep)[-1].split(".")[0]
        loaded_npz = helper.loadnpz(npz_file)
        loaded_npz_dict[filename] = loaded_npz

    return loaded_npz_dict


def multiple_regression_average_results(results_path):
    """Function to perform multiple regression on all values inside of the dictionary results_dict"""
    results_dict = read_in_npz_files(results_path)
    sorted_dict = {}
    sep = helper.check_platform()
    layer_path = results_path.split(sep)[0] + sep + "sub04"
    layer_names = helper.get_layers_ncondns(layer_path)[1]
    for key in layer_names:
        value = results_dict[key]
        results_dict[key] = multiple_regression(value)
        sorted_dict[key] = results_dict[key]
    visualize_multiple_regression(sorted_dict, results_path, "from averaged RDMs (Option1)", [])


def visualize_multiple_regression(results_dict, save_path, option, standard_error_dict):
    """Visualize and save multiple regression results"""
    x_coordinates = []
    y_coordinates = [[], [], [], [], []]  # Results for 5 predictors

    for key in results_dict:
        x_coordinates.append(key)
        for index in range(0,5):
            y_coordinates[index].append(results_dict[key][index])

    plt.plot(x_coordinates, y_coordinates[0], color="black", marker="^", label="Number")
    plt.plot(x_coordinates, y_coordinates[1], color="grey", marker="o", label="Average Item Size")
    plt.plot(x_coordinates, y_coordinates[2], color="green", marker="D", label="Total field Area")
    plt.plot(x_coordinates, y_coordinates[3], color="grey" , marker="s", label="Total surface")
    plt.plot(x_coordinates, y_coordinates[4], color="red",  marker="v", label="Density")

    sep = helper.check_platform()
    save_path = save_path + sep + "Multiple_regression_" + option

    plt.xlabel("layer")
    plt.ylabel("beta weights")
    plt.title("Multiple regression on layer RDMs")
    plt.legend( bbox_to_anchor = (1.05, 1), loc = 'upper left')


    if standard_error_dict != []:
        list_of_errors = []
        for layer in standard_error_dict:
            list_of_errors.append(standard_error_dict[layer])
    #print(standard_error_dict)

    plt.savefig(save_path,bbox_inches='tight')
    plt.close()


def get_standard_error(standard_error_dict):
    """
    This function gets the standard error for the solo multiple regression function
    Dictionary has the format:
    {
    layer1 : [[beta weights sub 1], [beta weights sub 2], ... ],
    layer2:  [[beta weights sub 1], [beta weights sub 2], ... ],
    ...
     }
    """
    # 1.) Format dictionary to dictionary with format:
    # { layer1 : [[predictor1 from sub 1 , p1 f s2 , p1 f3,...] , [predictor2 from sub1, p2 f s2,...],...], layer2: ...}
    predictor_dict = {}
    for layer in standard_error_dict:
        predictor_dict[layer] = [[],[],[],[],[]]
        for sub in standard_error_dict[layer]:
            index = 0
            for predictor in sub:
                predictor_dict[layer][index].append(predictor)
                index += 1

    standard_error_result_dict = {}
    # Remove first value from each
    for layer in predictor_dict:
        standard_error_result_dict[layer] = []
        index = 0
        for array in predictor_dict[layer]:

            x = array.remove(array[0])
            standard_error_result_dict[layer].append(np.std(array))
            index += 1

    return standard_error_result_dict

def multiple_regression_solo_averaged(path):
    """This function performs multiple regression for every npz and then calculates the average and visualizes it"""
    list_of_subs = helper.get_sub_list()
    sep = helper.check_platform()
    layer_path = path + sep + "sub04"
    layer_names = helper.get_layers_ncondns(layer_path)[1]
    save_path = path + sep + "average_rdms"
    result_dict = {}
    standard_error = {}
    for layer_name in layer_names:
        result_dict[layer_name] = []
        standard_error[layer_name] = []
    for sub in list_of_subs:
        sub_path = path + sep + sub + "_rdms"
        for layer_name in layer_names:
            npz_path = sub_path + sep + layer_name + ".npz"
            loaded_rdm = helper.loadnpz(npz_path)
            mr_of_rdm = multiple_regression(loaded_rdm)
            if (type(result_dict[layer_name])) == list:
                result_dict[layer_name] = mr_of_rdm
            else:
                result_dict[layer_name] += mr_of_rdm
            standard_error[layer_name].append(mr_of_rdm)

    standard_error_dict = (get_standard_error(standard_error))

    # Divide to get average
    for key in result_dict:
        result_dict[key] = result_dict[key] / 20

    visualize_multiple_regression(result_dict, save_path, "from every subject (Option2)", standard_error_dict)

def create_brain_region_rdm_dict(option):
    """
    preprocessing brain RDMs for RSA : comparing brain RDM to Network rdm
    Creates a dictionary with averaged brain region rdms inside {ips0: [rdm], ips12: [rdm], ... }
    option 1 = taskBoth
    option 2 = taskNum
    option 3 = taskSize
    """

    # Get option as filter to search for correct npz files
    sep = helper.check_platform()
    option = ["taskBoth", "taskNum", "taskSize"][option]

    cwd = os.getcwd()
    search_path = os.getcwd() + sep + "RSA_matrices"
    all_rsa_files = glob.glob(search_path + "/*" + ".npz")

    filtered_files = []
    brain_rdms_dictionary = {}
    # Filter them on the correct option
    for file in all_rsa_files:
        if (option in file):
            filtered_files.append(file)

    for filtered_file in filtered_files:
        average_rdm = []
        loadednpz = helper.loadnpz(filtered_file)
        filename = filtered_file.split(helper.check_platform())[-1]
        brain_region = filename.split("_")
        brain_region = brain_region[-1].split(".")[0]
        # loaded npz is a 20x18x18 array
        if (np.shape(loadednpz.f.arr_0)) == (20, 18, 18):
            # needs to be averaged
            for rdm in loadednpz.f.arr_0:
                if type(average_rdm) == list:
                    average_rdm = rdm
                else:
                    average_rdm += rdm
            average_rdm = average_rdm / 20

        brain_rdms_dictionary[brain_region] = average_rdm

    return brain_rdms_dictionary


def sq(x):
    return squareform(x, force = "tovector", checks=False)


def visualize_rsa_matrix(result_dict,layer_list, option):
    rsa_matrix = []
    brain_regions = ["IPS15", "V3ABV7", "V13", "IPS345", "IPS12", "IPS0", "V3AB", "V3", "V2", "V1"]

    for brain_region in brain_regions:
        rsa_matrix.append(result_dict[brain_region])
    rsa_matrix = np.array(rsa_matrix)
    # Visualize
    layers = layer_list

    plt.title("brain - network RDM similarity: " + option, fontsize = 15)
    heatmap = sns.heatmap(rsa_matrix, xticklabels=layers, yticklabels=brain_regions, cmap="inferno", vmin=0  #vmax=1
        ,cbar_kws = {'label': 'R2 Score from linear regression'})   #,cbar_kws={'label': 'Spearman correlation coefficient'}) #cbar_kws={'label': 'R2 Score from linear regression'})#
    heatmap.figure.axes[-1].yaxis.label.set_size(13)

    return heatmap


def create_rsa_matrix(option, result_path):

    #Creates matrix comparing with rsa : [ x axis : layer of network , y axis : brain regions from RSA_Matrices ]
    #option 1 = taskBoth
    #option 2 = taskNum
    #option 3 = taskSize

    # Creating a dictionary for the RDMs of the brain regions depending on the option
    # Old version
    #brain_rdms = create_brain_region_rdm_dict(option)
    brain_rdms = get_brain_regions_npz(option)
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
            network_rdm = remove_diagonal(get_lowertriangular(network_rdm))
            #rsa_result_spearman = compare_rdms(network_rdm,brain_rdm)
            rsa_result_squared = evaluate_fmri(network_rdm , brain_rdm)[0]
            #result_dict_spearman_correlation[brain_region].append(rsa_result_spearman)
            result_dict_squared_regression[brain_region].append(rsa_result_squared)

    sep = helper.check_platform()
    save_path = (result_path + sep + "average_rdms" + sep + "RSA" + "_" + ["taskBoth", "taskNum", "taskSize"][option])
    visualize_rsa_matrix(result_dict_squared_regression, layer_list, ["taskBoth", "taskNum", "taskSize"][option])
    plt.savefig(save_path)
    plt.close()



def get_uppernoiseceiling(rdm):
    num_subs = rdm.shape[0]
    unc = []
    for i in range(num_subs):
        sub_rdm = rdm[i,:,:]
        mean_sub_rdm = np.mean(rdm,axis=0)
        sub_rdm = (remove_diagonal(get_lowertriangular(sub_rdm))).reshape(-1,1)
        mean_sub_rdm = (remove_diagonal(get_lowertriangular(mean_sub_rdm))).reshape(-1,1)
        lr = linear_model.LinearRegression()
        lr.fit(sub_rdm,mean_sub_rdm)
        unc.append(lr.score(sub_rdm,mean_sub_rdm))
    unc = np.mean(unc)
    return unc

"""
def get_uppernoiseceiling(rdm):
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
        sub_rdm = remove_diagonal(get_lowertriangular(sub_rdm))
        mean_sub_rdm = remove_diagonal(get_lowertriangular(mean_sub_rdm))
        unc += RSA_spearman(sub_rdm, mean_sub_rdm)  # calculate spearman
    unc = unc / num_subs
    return unc
"""

def get_lowernoiseceiling(rdm):
    num_subs = rdm.shape[0]
    lnc = []

    for i in range(num_subs):
        sub_rdm = rdm[i,:,:]
        rdm_sub_removed = np.delete(rdm, i, axis=0)
        mean_sub_rdm = np.mean(rdm_sub_removed,axis=0)
        sub_rdm = (remove_diagonal(get_lowertriangular(sub_rdm))).reshape(-1,1)
        mean_sub_rdm = remove_diagonal(get_lowertriangular(mean_sub_rdm)).reshape(-1,1)
        lr = linear_model.LinearRegression()
        lr.fit(sub_rdm,mean_sub_rdm)
        lnc.append(lr.score(sub_rdm,mean_sub_rdm))

    lnc = np.mean(lnc)

    return lnc

"""
def get_lowernoiseceiling(rdm):
    
    #Take the lower noise ceiling
    #1. Extracting one subject from the overall rdm
    #2. Take the mean of all the other RDMs
    #3. Take spearman correlation of subject RDM and mean subject RDM
    #4. We do this for all the subjects and then calculate the average
    #=> Can we predict person 15 from the rest of the subjects?
    #=> Low Noise-Ceiling means we need better data
    
    num_subs = rdm.shape[0]
    lnc = 0.0
    for i in range(num_subs):
        sub_rdm = rdm[i, :, :]
        rdm_sub_removed = np.delete(rdm, i, axis=0)  # remove one person
        mean_sub_rdm = np.mean(rdm_sub_removed, axis=0)  # take mean of other RDMs
        mean_sub_rdm = remove_diagonal(get_lowertriangular(mean_sub_rdm))
        sub_rdm = remove_diagonal(get_lowertriangular(sub_rdm))
        lnc += RSA_spearman(sub_rdm, mean_sub_rdm)  # take spearman
    lnc = lnc / num_subs  # average it
    return lnc
"""

def get_noise_ceiling_fmri(target):
    "Function to calculate noise ceilings for fmri scans"

    key_list = []
    for keys, values in target.items():
        key_list.append(keys)

    # lower nc and upper nc
    lnc = get_lowernoiseceiling(target[key_list[0]])
    unc = get_uppernoiseceiling(target[key_list[0]])

    noise_ceilings = {"lnc": lnc, "unc" : unc}
    return noise_ceilings

def get_brain_regions_npz(option):
    # Get option as filter to search for correct npz files
    sep = helper.check_platform()
    option = ["taskBoth", "taskNum", "taskSize"][option]
    # All rsa_files as npz
    #all_rsa_files  = glob.glob("RSA_Matrices" + "/*.npz")
    #network_rdms = glob.glob(average_results + "/*" + ".npz", recursive=True)
    #average_results = result_path + sep + "average_results"
    cwd = os.getcwd()
    search_path = os.getcwd() + sep + "RSA_matrices"
    all_rsa_files = glob.glob(search_path + "/*" + ".npz")
    brain_region_dict = {}
    filtered_files = []
    # Filter them on the correct option
    for file in all_rsa_files:
        if (option in file):
            filtered_files.append(file)

    for filtered_file in filtered_files:
        average_rdm = []
        loadednpz = helper.loadnpz(filtered_file)
        filename = filtered_file.split(helper.check_platform())[-1]
        brain_region = filename.split("_")
        brain_region = brain_region[-1].split(".")[0]
        brain_region_dict[brain_region] = loadednpz

    return brain_region_dict


def scan_result(out,noise_ceiling):
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
    sig_value = out[area_name][1]  # Are the results I produce random? Could I get my results with a random RDM?



    # return all the values in a dict
    # {'EVC_RDMs': [0.0007705639149007719, 0.001534065990921806, 4.9209348450876575], [lnc,unc]}
    output_dict = {area_name: [r2_value, area_percentNC, sig_value, [lnc, unc]]}

    return output_dict


def evaluate_fmri(layer_rdm,fmri_rdms):

    corr = []
    layer_rdm = layer_rdm.reshape(-1,1)
    
    for fmri_rdm in fmri_rdms:
        lr = linear_model.LinearRegression()
        fmri_rdm = (remove_diagonal(get_lowertriangular(fmri_rdm))).reshape(-1,1)
        lr.fit(layer_rdm, fmri_rdm)
        corr.append(lr.score(layer_rdm,fmri_rdm))

    corr_squared = np.square(corr)
    return np.mean(corr), stats.ttest_1samp(corr, 0)[1]


def noise_ceiling_main(option,network_save_path):

    brain_regions = ["IPS15", "V3ABV7", "V13", "IPS345", "IPS12", "IPS0", "V3AB", "V3", "V2", "V1"]

    #Get layernames
    sep = helper.check_platform()
    layer_path = network_save_path + sep + "sub04"
    layer_names = helper.get_layers_ncondns(layer_path)[1]

    #3.) Get noise ceilling for every brain region
    brain_region_noise_dict = get_brain_regions_npz(option)

    for brain_region in brain_regions:
        noise = get_noise_ceiling_fmri(brain_region_noise_dict[brain_region])
        brain_region_noise_dict[brain_region] = noise


    brain_npz_files = get_brain_regions_npz(option)
    result_dict = {}
    for layer in layer_names:
        result_dict[layer] = {}
        for brain_region in brain_regions:
            layer_path = network_save_path + sep + "average_rdms" + sep + layer + ".npz"
            layer_rdm = helper.loadnpz(layer_path)
            layer_rdm = layer_rdm.f.arr_0
            layer_rdm = remove_diagonal(get_lowertriangular(layer_rdm))
            brain_rdms = brain_npz_files[brain_region].f.arr_0
            result_for_layer = {layer : evaluate_fmri(layer_rdm,brain_rdms)}
            result_dict[layer][brain_region] = scan_result(result_for_layer,brain_region_noise_dict[brain_region])

    save_as_xlsx(result_dict,layer_names,network_save_path)
    visualize_noise_graph(result_dict,brain_region_noise_dict, network_save_path)


def save_as_xlsx(result_dict, layer_names,save_path):

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
    save_path_excel = save_path + sep + "average_rdms" + sep + "R² and noise ceilling results.xlsx"
    df.to_excel(save_path_excel,sheet_name="results", index=False)


def visualize_noise_graph(result_dict,brain_region_noise_dict, save_path):

    #Get layernames
    sep = helper.check_platform()
    layer_path = save_path + sep + "sub04"
    layer_names = helper.get_layers_ncondns(layer_path)[1]
    save_path = save_path + sep + "average_rdms"
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
    colours = list(islice(cycle(colours), 100)) #cicle through color scheme
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
    #axs[0].set(ylabel="Spearman correlation coefficient")
    axs[1].set(xlabel="Brain ROIs")
    #axs[1].set(ylabel="Spearman correlation coefficient")

    axs[1].set_title("Upper noise ceiling")
    for x in range(len(x_label)):
        lnc_plot = axs[0].bar(x_label[x], lnc_y_values[x],color=colours[x])
        unc_plot = axs[1].bar(x_label[x], unc_y_values[x],color=colours[x])

    save_path_noise = save_path + sep + "Lower and upper noise ceilling"
    plt.savefig(save_path_noise)
    #plt.show()
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
        fig.suptitle("Brainregion: " + roi + "| R² and Noise ceilling in % ")
        #fig.suptitle("Brainregion: " + roi + "| R and Noise ceilling in % ")
        fig.set_figheight(5)
        fig.set_figwidth(20)
        #axs[0].set_title("R from Spearman")
        axs[0].set_title("R² score from linear regression")
        axs[0].set(xlabel="Network Layers")
        axs[0].set(ylabel="Colored: R² , Grey: Noiseceilling")
        #axs[0].set(ylabel="Colored: R , Grey: Noiseceilling")

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

#noise_ceiling_main(1,"Alexnet pretrained results")
#create_rsa_matrix(0,"Alexnet pretrained results")
#create_rsa_matrix(1,"Alexnet pretrained results")
#create_rsa_matrix(2,"Alexnet pretrained results")
