from prettytable import PrettyTable
from prettytable import ALL as ALL
import time
from networks.alexnet import *
from networks.resnet import *
import os
import generate_features as gf
import helper as helper
import generate_RDM as RDMs
import RDM_Evaluation
from tqdm import tqdm
'''
This file is used to run the main function.
Structure:
1.) Choose Model
2.) Generate features
3.) Generate RDMs
4.) Multiple regression
'''


class Clear:
    def __call__(self):
        if os.name==('ce','nt','dos'): os.system('cls')
        elif os.name=='posix': os.system('clear')
        else: print('\n'*120)
        def __neg__(self): self()
        def __repr__(self):
            self();return ''


def represents_int(s):
    "Check if string s is representing an integer"
    try:
        int(s)
        return True
    except ValueError:
        return False


def create_pretty_table_networks():
    p = PrettyTable(hrules=ALL)

    p.field_names = ["Configuration_Number", "Network", "Pretrained / Random Weights"]
    p.add_row(["1", "Alexnet", "Pretrained"])
    p.add_row(["2", "Alexnet", "Random Weights"])
    p.add_row(["3", "cornet_rt", "Pretrained"])
    p.add_row(["4", "cornet_rt", "Random Weights"])
    p.add_row(["5", "cornet_s", "Pretrained"])
    p.add_row(["6", "cornet_s", "Random Weights"])
    p.add_row(["7", "resnet", "Pretrained"])
    p.add_row(["8", "resnet", "Random Weights"])

    last_row = 0
    for row in p:
        row.border = False
        row.header = False
        last_row = (row.get_string(fields=["Configuration_Number"]).strip())
    last_row = (int(last_row))
    return [p, last_row]


def get_model(config_number):

    if config_number == "1":
        model = AlexNet(is_pretrained=True)
    elif config_number == "2":
        model = AlexNet(is_pretrained=False)

    elif config_number == "7":
        model = resnet18(is_pretrained=True)
    elif config_number == "8":
        model = resnet18(is_pretrained=False)

    return model


def get_save_path(config_number):

    if config_number == "1":
        save_path = "Alexnet pretrained results"
    elif config_number == "2":
        save_path = "Alexnet random results"
    elif config_number == "7":
        save_path = "resnet18 pretrained results"
    elif config_number == "8":
        save_path = "resnet18 random results"

    return save_path


def choose_model_main():

    network_table = create_pretty_table_networks()
    last_row = network_table[1]
    network_table = network_table[0]

    correct_input = False
    while not correct_input:
        clear = Clear()
        print(network_table)
        config_number = input("Please enter the Configuration_Number from the tabel above:")

        if not represents_int(config_number):
            print("ERROR: Please enter an number!")
            time.sleep(5)
        elif (int(config_number) < 1) or (int(config_number) > last_row):
            print("ERROR: Please enter an number between 1 and " + str(last_row) + " !")
            time.sleep(5)
        else:
            correct_input = True
            model = get_model(config_number)
            save_path = get_save_path(config_number)

    return [model, save_path]


def generate_features_main(model, save_path):

    clear = Clear()
    ask_generate_features = input("Generate features ? Enter 1 for yes:")

    if ask_generate_features == "1":
        print("Features for all subjects and stimuli will be generated.")
        print("This might take a while please be patient.")
        gf.run_torchvision_model(model, save_path)
    else:
        print("WARNING: Features will not be generated!")
        time.sleep(5)


def generate_rdms_main(save_path):

    clear = Clear()
    list_of_subs = helper.get_sub_list()
    generate_rdms = input("Generate RDMs ? Enter 1 for yes:")
    if generate_rdms == "1":
        print("RDMs will be created in: " + save_path )
        for sub in tqdm(list_of_subs):
            RDMs.create_rdms(save_path, sub)
        RDMs.create_average_rdm(save_path)
        RDMs.visualize_rdms(save_path)

    else:
        print("WARNING: RDMs will not be generated!")
        time.sleep(5)


def multiple_regression_main(save_path):

    clear = Clear()
    create_multiple_regression_graph = input("Generate beta weights Graph (Multiple Regression)? Enter 1 for yes:")
    if create_multiple_regression_graph == "1":
        print("Option 1 : Yield beta weights from the averaged RDMs.")
        print("Option 2 : Perform multiple regression for every subject layer. Average after.")
        option = input("Please choose option (1/2):")

        if option == "1":
            save_path_2 = os.path.join(save_path, "average_results")
            RDM_Evaluation.multiple_regression_average_results(save_path_2)
            print("Multiple regression was performed on average RDMs. Graph is created in: " + save_path_2)
            option2 = input("Would you like to also create option 2? Enter 1 for yes:")
            if option2 == "1":
                RDM_Evaluation.multiple_regression_solo_averaged(save_path)
                print(
                    "Multiple regression was performed on every subject. Averaged Graph is created in: " + save_path_2)

        if option == "2":
            save_path_2 = os.path.join(save_path, "average_results")
            RDM_Evaluation.multiple_regression_solo_averaged(save_path)
            print("Multiple regression was performed on every subject. Averaged Graph is created in: " + save_path_2)
            option1 = input("Would you like to also create option 1? Enter 1 for yes:")
            if option1 == "1":
                RDM_Evaluation.multiple_regression_average_results(save_path_2)
                print("Multiple regression was performed on average RDMs. Graph is created in: " + save_path_2)


def rsa_heatmap_main(save_path):

    clear = Clear()
    create_rsa_heatmap = input("Create RSA (Brain RDMs vs Network RDMs)? Enter 1 for yes:")

    if create_rsa_heatmap == "1":

        finished = 0
        while finished == 0:
            clear = Clear()
            print("Choose option for brain rdms.")
            print("Enter 1 for TaskBoth")
            print("Enter 2 for TaskNum")
            print("Enter 3 for TaskSize")
            choose_option = input("Enter the option (1,2,3):")
            choose_option = int(choose_option) - 1
            RDM_Evaluation.create_rsa_matrix(choose_option, save_path)
            print("Heatmap was created in:" + " " + save_path + "\\" + "average_results")
            finished_check = input("Would you like to create more heatmaps? Enter 1 for yes:")
            if int(finished_check) != 1:
                finished = 1


def main_ui():

    # 1.) Choose Model
    model_info = choose_model_main()
    model = model_info[0]
    save_path = model_info[1]

    # 2.) Generate features:
    generate_features_main(model, save_path)

    # 3.) Generate RDMs
    try:
        generate_rdms_main(save_path)
    except FileNotFoundError:
        print("WARNING: RDMS could not be created!")
        print("Please make sure that features were created before.")

    # 4.) Multiple Regression
    try:
        multiple_regression_main(save_path)
    except FileNotFoundError:
        print("WARNING: Multiple regression could not be performed!")
        print("Please make sure that RDMs were created before")

    # 5.) Create RSA heatmap
    try:
        rsa_heatmap_main(save_path)
    except FileNotFoundError:
        print("WARNING: RSA could not be performed!")
        print("Please make sure that RDMs were generated before.")

    # 6.) Delete activation files
    delete_files_bool = input("Delete npz files to save memory? Enter 1 for yes:")
    if delete_files_bool == "1":
        print("NPZ files will be deleted!")
        helper.delete_files(save_path)
    else:
        print("NPZ files will not be deleted!")


main_ui()