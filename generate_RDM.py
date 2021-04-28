import numpy as np
import os
import matplotlib.pyplot as plt
import helper
import seaborn as sns


def sum_tensors(all_tensors, x):
    """
    Summing two tensors
    Cases :
    1.) Empty list + Tensor
    2.) List + Tensor
    3.) Tensor + Tensor
    """

    new_tensor = []

    # Case 1: Tensor all_tensors is empty
    if all_tensors == []:
        all_tensors = x
        return all_tensors

    # Case 2: Tensor all_tensors is not empty and type is list
    elif type(all_tensors) == list:
        for i, tensor in enumerate(x):
            # Get correct arrays from both list and tensor
            # all tensors is a list [i]
            layer = all_tensors[i]
            # x is a tensor x[tensor]s
            layer2 = x[tensor]
            result = layer.__add__(layer2)
            new_tensor.append(result)

    # Case 3: Tensor all_tensors is an nparray just like tensor x
    else:
        for i, tensor in enumerate(x):
            # Get correct arrays from both tensors
            layer = all_tensors[tensor]
            layer2 = x[tensor]

            # Add the two arrays
            result = layer.__add__(layer2)
            new_tensor.append(result)

    return new_tensor


def divide_tensors(all_tensors_dict):
    """
    Function to divide a tensor dictionary
    Used for averaging across all test cases
    Format : { test case : [array , frequency], test case 2 : [array,frequency], ...}
    for every test case:
        dictionary[test case] = array/ frequency

    Returns dictionary {test case : list(averaged array),...}
    """
    # for every test case
    for test_case in all_tensors_dict:
        new_array = []
        # get frequency
        frequency = all_tensors_dict[test_case][1]
        # get array
        array = all_tensors_dict[test_case][0]
        # for every element in array: create new array with average values
        for element in array:
            new_array_element = element / frequency
            new_array.append(new_array_element)
        all_tensors_dict[test_case] = new_array
    return all_tensors_dict


def average_all_npz(result_path, sub):
    """
    This function is used to calculate the average for every test case of all npz files that were generated for one sub
    Uses a dictionary to calculate
    Returns: dictionary of average layer values
    """
    sep = helper.check_platform()
    path = result_path + sep + sub

    # Array with all test cases

    position_and_frequency = helper.get_test_cases()

    for file_name in os.listdir(path):

        # Assign npz to correct test case
        test_case_of_npz = (file_name.split("_")[2] + "_" + file_name.split("_")[3] + "_" + file_name.split("_")[4])

        # Fill dictionary {test case 1 : [all arrays summed, frequency] , testcase2 : ... , ...}
        # If frequency == 0: sum tensors on empty list
        if position_and_frequency[test_case_of_npz][1] == 0:
            position_and_frequency[test_case_of_npz][0] = sum_tensors([],
                                                               helper.loadnpz(path+ sep + file_name))
        # If frequency /= 0: sum tensors
        else:
            position_and_frequency[test_case_of_npz][0] = sum_tensors(position_and_frequency[test_case_of_npz][0],
                                                               helper.loadnpz(path + sep + file_name))

        # Count frequency
        position_and_frequency[test_case_of_npz][1] += 1

    # Average all test cases
    layer_names = helper.get_layers_ncondns(path)[1]
    averaged_test_cases_dict = divide_tensors(position_and_frequency)
    # Convert them to an dictionary with layers as keys
    array_dict = convert_arrays_to_layer(averaged_test_cases_dict, layer_names)

    return array_dict


def convert_arrays_to_layer(averaged_array_dict, layers):
    """
    This function converts the dictionary with averaged arrays (for every test case) into a dictionary for the layers
    """

    old_dict = averaged_array_dict
    new_dict = {}
    for i, key in enumerate(layers):
        layer_list = []
        for second_key in old_dict:
            value = old_dict[second_key]
            correct_layer = value[i]
            layer_list.append(correct_layer)
        new_dict[key] = layer_list
    return new_dict


def get_average_of_voxels(averaged_array_dict):
    """This function calculates the average for every unit of the "voxel" (2d part of tensor) """
    #average_list = [[], [], [], [], [], [], [], []]
    average_list = []
    for layer in averaged_array_dict:
        average_list.append([])
    # Get mean for every unit of voxel
    layers_dict = averaged_array_dict

    layer_number = 0
    # for every layer
    for layer_name in layers_dict:
        # for every test case
        for test_case in layers_dict[layer_name]:    # 0 ... 17
            voxel_averages = []
            for first_dimension in test_case:   #(1,64,55,55) -> (64,55,55)
                for second_dimension in first_dimension:  #(64,55,55) -> (55,55)
                    mean_of_voxel = np.mean(second_dimension)
                    voxel_averages.append(mean_of_voxel)

            voxel_array = np.asarray(voxel_averages)
            if average_list[layer_number] == []:
                average_list[layer_number] = voxel_array
            else:
                average_list[layer_number] = average_list[layer_number] + voxel_array

        layer_number += 1

    for index, item in enumerate(average_list):
        average_list[index] = item / 18

    return average_list


def scale_arrays(averaged_array_dict):
    """This function scales the averaged array dictionary using the average list from get_average_of_voxels"""
    average_list = get_average_of_voxels(averaged_array_dict)
    layers_dict = averaged_array_dict
    new_dictionary = {}

    first_index = 0    # conv1, conv2 ...
    # for every layer conv1, conv2 ...
    for layer_name in layers_dict:
        # Test cases , N1S02T03
        test_cases = layers_dict[layer_name]
        new_dictionary[layer_name] = []
        # for every test case in test cases 0, 1, ... ,17
        for test_case in test_cases:
            new_dictionary[layer_name].append([])
            second_index = 0
            for first_dimension in test_case:
                for second_dimension in first_dimension:
                    second_dimension = second_dimension - average_list[first_index][second_index]
                    new_dictionary[layer_name][-1].append(second_dimension)
                    second_index += 1
        first_index += 1

    return new_dictionary


def pearson_coefficient(scaled_layers):
    """This function finally creates the rdm using 1 - pearson coefficient """
    flattend_activations = []
    arrays = np.asarray(scaled_layers)
    # Flatten the arrays to calculate pearson coefficient
    for array in arrays:
        flattend_activations.append(array.ravel())

    flattend_activations = np.asarray(flattend_activations)
    rdm = 1 - np.corrcoef(flattend_activations)

    return rdm


def create_average_rdm(result_path):
    """This function calculates the average RDMs for all subjects"""
    list_of_subs = helper.get_sub_list()

    # Get the names for all layers of the network
    path = result_path + helper.check_platform() + "sub04"
    layer_names = helper.get_layers_ncondns(path)[1]
    sep = helper.check_platform()
    for layer in layer_names:
        result_array = 0
        for sub in list_of_subs:
            layer_path = result_path + sep + sub+"_rdms" + sep + layer + ".npz"
            loaded_npz = helper.loadnpz(layer_path)
            result_array += loaded_npz.f.arr_0

        result_array = result_array / len(list_of_subs)
        save_path = os.path.join(result_path, "average_results")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.savez((save_path + sep + layer + ".npz"), result_array)


def create_rdms(result_path, sub):
    """
    Main function to create rdms
    Structrue:
    1.) Average all npz
    2.) Scale
    3.) Calculate pearson coefficient
    """
    sep = helper.check_platform()
    layers_dict = average_all_npz(result_path, sub)
    scaled_layers = scale_arrays(layers_dict)
    net_save_dir = result_path + sep + sub + "_rdms"

    if not os.path.exists(net_save_dir):
        os.makedirs(net_save_dir)

    for layer in layers_dict:
        rdm = pearson_coefficient(scaled_layers[layer])
        save_path = result_path + sep + sub + "_rdms" + sep + layer
        np.savez(save_path, rdm)


def visualize_rdms(result_path):
    """This function visualizes every RDM"""
    list_of_subs = helper.get_sub_list()
    list_of_subs.append("average_results")
    # Get names of the layers
    path = result_path + helper.check_platform() + "sub04"
    layer_names = helper.get_layers_ncondns(path)[1]

    # For every subject
    for sub in list_of_subs:
        # For every layer
        for layer in layer_names:
            # Visualize RDM
            if sub == "average_results":
                path = os.path.join(result_path, sub, layer + ".npz")
            else:
                path = os.path.join(result_path, sub + "_rdms", layer + ".npz")

            rdm = np.load(path)
            rdm = rdm.f.arr_0

            test_cases_dict = helper.get_test_cases()
            test_cases = []
            for test_case in test_cases_dict:
                test_cases.append(test_case)

            plt.title("RDM of layer: " + layer)
            plt.figure(figsize=(12, 10), dpi=80)
            save_path = path[:-4]
            plt.title("RDM of layer: " + layer, fontsize=20)
            heatmap = sns.heatmap(rdm, xticklabels=test_cases, yticklabels=test_cases, cmap="inferno", vmin=0, vmax=2,
                        cbar_kws={'label': '1-Pearson r'})
            heatmap.figure.axes[-1].yaxis.label.set_size(16)
            plt.savefig(save_path)
            plt.close()


