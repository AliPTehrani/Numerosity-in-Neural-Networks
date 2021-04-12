import generate_features as gf
import generate_RDM as RDMs
import RDM_Evaluation
import os
from networks.alexnet import *
from networks.resnet import *
import helper
'''
This file is used to run the main function.
Structure:
1.) Choose Model
2.) Generate features
3.) Generate RDMs 
4.) Multiple regression
'''


def main():
    """Main function"""
    # some needed variables
    list_of_subs = helper.get_sub_list()

    # 1.)Ask for model
    print("Please choose the Network.")
    print("Enter 1 for AlexNet")
    print("Enter 2 for Resnet")
    model = input("Enter number for model:")
    random = input("Random weights (otherwise pretrained)? Enter 1 for random:")

    # Basing on Model get the reults path
    if model == "1":

        result_path = "Alexnet pretrained results"
        model = AlexNet(is_pretrained=True)
        if random == "1":
            model = AlexNet(is_pretrained=False)
            result_path = "Alexnet random results"

    if model == "2":
        result_path = "Resnet pretrained results"
        model = resnet18(pretrained=True)
        if random == "1":
            model = resnet18(pretrained=False)
            result_path = "Resnet random results"

    # 2.) Generate features if not existing
    generate_features = input("Generate features ? Enter 1 for yes:")
    if generate_features == "1":
        print("Features for all subjects and stimuli will be generated.")
        print("This might take a while please be patient.")
        gf.run_torchvision_model(model, result_path)


    # 3.) Generate RDMs
    generate_rdms = input("Generate RDMs ? Enter 1 for yes:")
    if generate_rdms == "1":
        for sub in list_of_subs:
            RDMs.create_rdms(result_path, sub)
        RDMs.create_average_rdm(result_path)
        RDMs.visualize_rdms(result_path)
    # 3.) Multiple regression
    create_multiple_regression_graph = input("Generate beta weights Graph (Multiple Regression)? Enter 1 for yes:")

    if create_multiple_regression_graph == "1":
        if not(generate_rdms == "1"):
            RDMs.create_average_rdm(result_path)

        result_path_2 = os.path.join(result_path ,"average_results")
        RDM_Evaluation.multiple_regression_average_results(result_path_2)

    # 4.) RSA
    create_RSA_heatmap = input("Create RSA (Brain RDMs vs Network RDMs? Enter 1 for yes:")
    if create_RSA_heatmap == "1":

        if (not(create_multiple_regression_graph == "1")) and (not(generate_rdms == "1")):
            RDMs.create_average_rdm(result_path)

        finished = 0
        while finished == 0:
            print("Choose option for brain rdms.")
            print("Enter 1 for TaskBoth")
            print("Enter 2 for TaskNum")
            print("Enter 3 for TaskSize")
            choose_option = input("Enter the option (1,2,3):")
            choose_option = int(choose_option) - 1
            RDM_Evaluation.create_rsa_matrix(choose_option, result_path)
            print("Heatmap was created in:" + " " + result_path + "\\" + "average_results")
            finished_check = input("Would you like to create more heatmaps? Enter 1 for yes:")
            if int(finished_check) != 1:
                finished = 1

    # 5.) Delete npz to save memory
    delete_files_bool = input("Delete npz files to save memory? Enter 1 for yes:")
    if delete_files_bool == "1":
        helper.delete_files(result_path)


main()
