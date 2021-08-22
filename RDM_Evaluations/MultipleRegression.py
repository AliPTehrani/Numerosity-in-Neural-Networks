import helper
import numpy as np
from sklearn import linear_model
from matplotlib import pyplot as plt
import glob
import os


'''
Multiple regression: Perform multiple linear regression to yield beta weights for the different parameters 
in our testcases. The RDMs that were created for the network will be used to perform the multiple regression against
five predictor matrices from the experiment.

There are two different options for multiple regression : 
    Option 1 : Multiple regression on the averaged RMDs . So the RDMs for all 20 subjects are averaged first and then
    the multiple regression will be performed on the average rdms
    
    Option 2: Multiple regression on every subject. The multiple regression will be performed for every RDM for every 
    subject. Afterwards the results from the multiple regression will be averaged.
'''


def z_transform_matrix(lower_triangle_matrix):
    """
    Will take the lower triangle that was extracted from a RDM (without the middle diagonal)
    and z_transform every value inside that array.

    for every value in array:
        new_value = z_transform(value) = value - (mean_value of array) / (standard_deviation of array)

    :param lower_triangle_matrix: extracted lower triangle values from an RDM as an 1d np array
    :return: new_array : the same array where every value inside is z transformed now
    """

    mean_value = np.mean(lower_triangle_matrix)
    standard_deviation = np.std(lower_triangle_matrix)

    new_array = []
    for value in lower_triangle_matrix:
        new_value = (value - mean_value) / standard_deviation
        new_array.append(new_value)

    new_array = np.asarray(new_array)

    return new_array


def multiple_regression(rdm):
    """
    This function is used to actually perform multiple regression on an given RDM.
    Multiple regression will be performed for the RDM against 5 predictor matrices that represent the different parameter
    settings of the stimuli

    :param rdm: Either an loaded npz file, storing the 18x18 RDM or the RDM can be extracted already into an np.array of
    the shape 18x18
    :return: clf.coef_ : These are the 5 beta weights resulting from the multiple regression
    """

    # Extract array (rdm) from npz if it is not already done
    # Given the loaded npz
    loaded_npz = rdm
    if isinstance(rdm, np.lib.npyio.NpzFile):
        rdm = loaded_npz.f.arr_0

    # Load predictor matrices
    sep = helper.check_platform()
    # all five predictor matrices are stored in the "all_RDM_predictors.npz"
    predictor_npz = helper.loadnpz("RSA_matrices" + sep + "all_RDM_predictors.npz")
    predictor_rdms = predictor_npz.f.arr_0
    predictor1 = predictor_rdms[0]
    predictor2 = predictor_rdms[1]
    predictor3 = predictor_rdms[2]
    predictor4 = predictor_rdms[3]
    predictor5 = predictor_rdms[4]

    # get lower triangular and remove diagonal for rdm and all five predictor matrices
    rdm = helper.get_lower_triangular(rdm)
    rdm = helper.remove_diagonal(rdm)

    predictor1 = helper.get_lower_triangular(predictor1)
    predictor1 = helper.remove_diagonal(predictor1)

    predictor2 = helper.get_lower_triangular(predictor2)
    predictor2 = helper.remove_diagonal(predictor2)

    predictor3 = helper.get_lower_triangular(predictor3)
    predictor3 = helper.remove_diagonal(predictor3)

    predictor4 = helper.get_lower_triangular(predictor4)
    predictor4 = helper.remove_diagonal(predictor4)

    predictor5 = helper.get_lower_triangular(predictor5)
    predictor5 = helper.remove_diagonal(predictor5)

    # Z-transform
    rdm = z_transform_matrix(rdm)
    predictor1 = z_transform_matrix(predictor1)
    predictor2 = z_transform_matrix(predictor2)
    predictor3 = z_transform_matrix(predictor3)
    predictor4 = z_transform_matrix(predictor4)
    predictor5 = z_transform_matrix(predictor5)

    # perform the Linear regression with x as the five predictor matrices and our given RDM
    x = np.array([predictor1, predictor2, predictor3, predictor4, predictor5]).transpose()
    clf = linear_model.LinearRegression().fit(x, rdm)

    return clf.coef_


def read_in_npz_files(path):
    """
    Loads all npz files that are inside an directory.

    :param path: Path of the directory where the .npz files are stored
    :return: loaded_npz_dict : dictionary that has the layer names as keys and the loaded npz as value
    """
    npz_files = glob.glob(path + "/*.npz")

    loaded_npz_dict = {}
    for npz_file in npz_files:
        sep = helper.check_platform()
        filename = npz_file.split(sep)[-1].split(".")[0]
        loaded_npz = helper.loadnpz(npz_file)
        loaded_npz_dict[filename] = loaded_npz

    return loaded_npz_dict

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


def visualize_multiple_regression(results_dict, save_path, option, standard_error_dict=[]):
    """
    Visualize and save multiple regression results

    :param results_dict:  results from the multiple regression as a dictionary. Keys are the layer names and values are
    the arrays with the five beta weights.
    :param save_path: Path of the directory where the plot should be saved.
    :param option: Just a String to clarify if it is the plFot for option 1 or option 2
    :param standard_error_dict: Optional list for option 2. It contains the results for every subject, so the standard
    error can be calculated among all subjects.
    :return: Nothing , saves plot of the multiple regression into the given directory
    """

    #
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

    plt.xticks(rotation=45)
    if standard_error_dict != []:
        list_of_errors = []
        for layer in standard_error_dict:
            list_of_errors.append(standard_error_dict[layer])
    #print(standard_error_dict)



    plt.savefig(save_path,bbox_inches='tight')
    plt.close()

def multiple_regression_average_results(path):
    """
    Option 1 : Perform multiple regression on the average RDMs of all subject.
    :param results_path: Path of the average RDMs
    :return: Nothing, saves the Plot in the same directory as the RDMs
    """
    """Function to perform multiple regression on all values inside of the dictionary results_dict"""
    sep = helper.check_platform()
    save_path = path + sep + "RDM_Evaluation_Results"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    results_path = path + sep + "average_rdms"
    results_dict = read_in_npz_files(results_path)
    sorted_dict = {}
    sep = helper.check_platform()
    layer_path = results_path.split(sep)[0] + sep + "sub04"
    layer_names = helper.get_layers_ncondns(layer_path)[1]
    for key in layer_names:
        value = results_dict[key]
        results_dict[key] = multiple_regression(value)
        sorted_dict[key] = results_dict[key]
    visualize_multiple_regression(sorted_dict, save_path, "from averaged RDMs (Option1)", [])


def multiple_regression_solo_averaged(path):
    """
    Option 2 : Performs multiple regression on all different subject x layer RDMs. Averages the results afterwards.
    :param path: Path were all the directories for the subjects can be found.
    :return: Nothing , saves the plot in the directory path + average_rdms
    """
    """This function performs multiple regression for every npz and then calculates the average and visualizes it"""
    list_of_subs = helper.get_sub_list()
    sep = helper.check_platform()
    layer_path = path + sep + "sub04"
    layer_names = helper.get_layers_ncondns(layer_path)[1]
    save_path = path + sep + "RDM_Evaluation_Results"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

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