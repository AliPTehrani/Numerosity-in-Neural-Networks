import numpy as np
import glob
import os
from scipy.stats import spearmanr
from sklearn import linear_model
import helper
from matplotlib import pyplot as plt
import seaborn as sns


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

    return rdm_lower_triangle


def RSA_spearman(rdm1, rdm2):
    """Calculate Spearman"""
    return spearmanr(rdm1, rdm2)[0]


def get_lowertriangular(rdm):
    """Get lower triangle of a RDM"""
    num_conditions = rdm.shape[0]
    return rdm[np.tril_indices(num_conditions, 1)]


def compare_rdms(rdm_path1,rdm_path2):
    """Compares two RDM using only the lower triangle and Spearman-correlation"""

    # Load npz files from given path
    if type(rdm_path1) == np.ndarray:
        rdm1 = rdm_path1
        rdm2 = rdm_path2
    else:
        loadednpz1 = helper.loadnpz(rdm_path1)
        loadednpz2 = helper.loadnpz(rdm_path2)
        rdm1 = loadednpz1.f.arr_0
        rdm2 = loadednpz2.f.arr_0

    # Extract array (rdm) from npz


    # Get lower triangle of rdm
    rdm1 = get_lowertriangular(rdm1)
    rdm2 = get_lowertriangular(rdm2)

    # Remove diagonal and values above the diagonal
    rdm1 = remove_diagonal(rdm1)
    rdm2 = remove_diagonal(rdm2)

    result = RSA_spearman(rdm1, rdm2)

    return result


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
    predictor_npz_1 = helper.loadnpz("RSA_matrices\\all_RDM_predictors\\all_RDM_predictors.npz")

    # Extract array (rdm) from npz
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
    npz_files = glob.glob(path + "/**/*.npz", recursive=True)
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

    for key in results_dict:
        value = results_dict[key]
        results_dict[key] = multiple_regression(value)

    visualize_multiple_regression(results_dict, results_path, "from averaged RDMs (Option1)", [])


def visualize_multiple_regression(results_dict, save_path, option, standard_error_dict):
    """Visualize and save multiple regression results"""
    x_coordinates = []
    y_coordinates = [[], [], [], [], []]  # Results for 5 predictors

    for key in results_dict:
        x_coordinates.append(key)
        for index in range(0,5):
            y_coordinates[index].append(results_dict[key][index])

    plt.plot(x_coordinates, y_coordinates[0], color = "black", marker = "^", label = "Number")
    plt.plot(x_coordinates, y_coordinates[1], color = "grey", marker = "o", label = "Average Item Size")
    plt.plot(x_coordinates, y_coordinates[2], color = "green", marker = "D", label = "Total field Area")
    plt.plot(x_coordinates, y_coordinates[3], color = "grey" , marker = "s", label = "Total surface")
    plt.plot(x_coordinates, y_coordinates[4], color = "red",  marker = "v", label = "Density")

    save_path = os.path.join(save_path, "beta_weights " + option)

    plt.xlabel("Layer")
    plt.ylabel("Weight")
    plt.title("Beta weights")
    plt.legend()

    if standard_error_dict != []:
        list_of_errors = []
        for layer in standard_error_dict:
            list_of_errors.append(standard_error_dict[key])
    print(list_of_errors)


    plt.savefig(save_path)
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
    save_path = os.path.join(path, "average_results")
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
    option = ["taskBoth", "taskNum", "taskSize"][option - 1]
    # All rsa_files as npz
    all_rsa_files = images = glob.glob("RSA_Matrices" + "/**/*.npz", recursive=True)
    filtered_files = []
    brain_rdms_dictionary = {}
    # Filter them on the correct option
    for file in all_rsa_files:
        if (option in file):
            filtered_files.append(file)

    for filtered_file in filtered_files:
        average_rdm = []
        loadednpz = helper.loadnpz(filtered_file)
        filename = filtered_file.split(helper.check_platform())[1]
        brain_region = (filename.split("_")[3])
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


def create_network_layer_rdm_dict(result_path):
    """Returns dictionary with {layer : rdm, ... } for the average rdms of the network"""

    average_results = os.path.join(result_path, "average_results")
    network_rdms = glob.glob(average_results + "/**/*.npz",recursive=True)
    network_rdms_dictionary = {}
    for network_rdm in network_rdms:
        layer_name = network_rdm.split(helper.check_platform())[-1].split(".")[0]
        loaded_npz = helper.loadnpz(network_rdm)
        rdm = loaded_npz.f.arr_0
        network_rdms_dictionary[layer_name] = rdm

    return network_rdms_dictionary


def visualize_rsa_matrix(result_dict,layer_list):
    rsa_matrix = []
    # Need dictionary reversed for visual matrix
    reverse_list = []
    for brain_region in result_dict:
        reverse_list.append(brain_region)
    reverse_list.reverse()
    for brain_region in reverse_list:
        rsa_matrix.append(result_dict[brain_region])
    rsa_matrix = np.array(rsa_matrix)
    # Visualize
    layers = layer_list
    brain_regions = reverse_list

    heatmap = sns.heatmap(rsa_matrix, xticklabels=layers, yticklabels=brain_regions,cmap="inferno",vmin = 0, vmax=1)
    return heatmap


def create_rsa_matrix(option, result_path):
    """
    Creates matrix comparing with rsa : [ x axis : layer of network , y axis : brain regions from RSA_Matrices ]
    option 1 = taskBoth
    option 2 = taskNum
    option 3 = taskSize
    """
    # Creating a dictionary for the RDMs of the brain regions depending on the option
    brain_rdms = create_brain_region_rdm_dict(option)
    # Creating a dictionary for the RDMs of the layers from the network
    network_rdms = create_network_layer_rdm_dict(result_path)

    # Comparing average brain rdms to average network rdms (brain region vs network layer)
    rsa_dict = {}

    # Calculating an result dictionary
    layer_list = []

    result_dict = {}
    for brain_region in brain_rdms:
        brain_rdm = brain_rdms[brain_region]
        result_dict[brain_region] = []
        for layer in network_rdms:
            if layer not in layer_list:
                layer_list.append(layer)
            network_rdm = network_rdms[layer]
            rsa_result = compare_rdms(network_rdm,brain_rdm)
            result_dict[brain_region].append(rsa_result)

    save_path = os.path.join(result_path, "average_results", "RSA" + "_" + ["taskBoth", "taskNum", "taskSize"][option])
    visualize_rsa_matrix(result_dict, layer_list)
    plt.savefig(save_path)
    plt.close()



#OPTION 2:
#result_dict = multiple_regression_solo_averaged("Alexnet pretrained results")

#OPTION 1:
#multiple_regression_average_results("Alexnet results\\average_rdms")

#RSA
#create_rsa_matrix(1, "Alexnet results")

#x = compare_rdms("Alexnet random results_1\sub04\sess1_tr01_N3_S3_TFA1_J1.npz","Alexnet random results\sub04\sess1_tr01_N3_S3_TFA1_J1.npz")
#print(x)