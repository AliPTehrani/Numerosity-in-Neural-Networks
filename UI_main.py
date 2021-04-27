from prettytable import PrettyTable
from prettytable import ALL as ALL
import time
from networks.alexnet import *
from networks.cornet_rt import *
from networks.cornet_s import *
from networks.cornet_z import *
from networks.resnet import *
from networks.vgg11 import *
from networks.vgg11_bn import *
from networks.vgg13 import *
from networks.vgg13_bn import *
from networks.vgg16_bn import *
from networks.vgg16 import *
from networks.vgg19 import *
from networks.vgg19_bn import *
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
    print("Network Table:")
    p = PrettyTable(hrules=ALL)

    p.field_names = ["ID", "Network"]
    p.add_row(["1", "AlexNet"])
    p.add_row(["2", "CorNet"])
    p.add_row(["3", "ResNet"])
    p.add_row(["4", "VGG"])


    last_row = 0
    for row in p:
        row.border = False
        row.header = False
        last_row = (row.get_string(fields=["ID"]).strip())
    last_row = (int(last_row))
    return [p, last_row]

def create_pretty_table_network_configuration(ID):
    p = PrettyTable(hrules=ALL)
    p.field_names = ["ID", "Architecture", "Pretrained/Random"]
    # ID 1 : Alexnet
    if ID == "1":
        print("Architecture Table for Alexnet:")
        p.add_row(["1", "AlexNet", "Pretrained"])
        p.add_row(["2", "AlexNet", "Random"])

    if ID == "2":
        print("Architecture Table for Cornet:")
        p.add_row(["1", "Cornet_rt", "Pretrained"])
        p.add_row(["2", "Cornet_rt", "Random"])
        p.add_row(["3", "Cornet_s", "Pretrained"])
        p.add_row(["4", "Cornet_s", "Random"])
        p.add_row(["5", "Cornet_z", "Pretrained"])
        p.add_row(["6", "Cornet_z", "Random"])

    if ID == "3":
        print("Architecture Table for Resnet:")
        p.add_row(["1", "resnet18", "Pretrained"])
        p.add_row(["2", "resnet18", "Random"])
        p.add_row(["3", "resnet34", "Pretrained"])
        p.add_row(["4", "resnet34", "Random"])
        p.add_row(["5", "resnet50_load", "Pretrained"])
        p.add_row(["6", "resnet50_load", "Random"])
        p.add_row(["7", "resnet101", "Pretrained"])
        p.add_row(["8", "resnet101", "Random"])
        p.add_row(["9", "resnet152", "Pretrained"])
        p.add_row(["10", "resnet152", "Random"])

    if ID == "4":
        print("Architecture Table for VGG:")
        p.add_row(["1", "vgg11", "Pretrained"])
        p.add_row(["2", "vgg11", "Random"])
        p.add_row(["3", "vgg11_bn", "Pretrained"])
        p.add_row(["4", "vgg11_bn", "Random"])
        p.add_row(["5", "vgg13", "Pretrained"])
        p.add_row(["6", "vgg13", "Random"])
        p.add_row(["7", "vgg13_bn", "Pretrained"])
        p.add_row(["8", "vgg13_bn", "Random"])
        p.add_row(["9", "vgg16", "Pretrained"])
        p.add_row(["10", "vgg16", "Random"])
        p.add_row(["11", "vgg16_bn", "Pretrained"])
        p.add_row(["12", "vgg16_bn", "Random"])
        p.add_row(["13", "vgg19", "Pretrained"])
        p.add_row(["14", "vgg19", "Random"])
        p.add_row(["15", "vgg19_bn", "Pretrained"])
        p.add_row(["16", "vgg19_bn", "Random"])

    last_row = 0
    for row in p:
        row.border = False
        row.header = False
        last_row = (row.get_string(fields=["ID"]).strip())
    last_row = (int(last_row))
    return [p, last_row]


def get_model_and_save_path(model_id,setting_id):
    model_index = int(model_id) - 1
    setting_index = int(setting_id) - 1

    result = 0
    if model_index == 0:
        alexnet_setting = [[AlexNet(is_pretrained=True), "Alexnet pretrained results"],
                           [AlexNet(is_pretrained=False), "Alexnet random results"]
                           ]

        result = alexnet_setting[setting_index]

    elif model_index == 1:

        if setting_index == 0:
            result = [cornet_rt(pretrained=True), "Cornet_rt pretrained results"]
        elif setting_index == 1:
            result = [cornet_rt(pretrained=False),"Cornet_rt random results"]
        elif setting_index == 2:
            result = [cornet_s(pretrained=True),"Cornet_s pretrained results"]
        elif setting_index == 3:
            result = [cornet_s(pretrained=False),"Cornet_s random results"]
        elif setting_index == 4:
            result = [cornet_z(pretrained=True),"Cornet_z pretrained results"]
        elif setting_index == 5:
            result = [cornet_z(pretrained=False), "Cornet_z random results"]

    elif model_index == 2:

        if setting_index == 0:
            result = [resnet18(pretrained=True), "Resnet18 pretrained results"]
        elif setting_index == 1:
            result = [resnet18(pretrained=False), "Resnet18 random results"],
        elif setting_index == 2:
            result = [resnet34(pretrained=True), "Resnet34 pretrained results"]
        elif setting_index == 3:
            result = [resnet34(pretrained=False), "Resnet34 random results"]
        elif setting_index == 4:
            result = [resnet50_load(pretrained=True), "Resnet50 pretrained results"]
        elif setting_index == 5:
            result = [resnet50_load(pretrained=False), "Resnet50 random results"]
        elif setting_index == 6:
            result = [resnet101(pretrained=True), "Resnet101 pretrained results"]
        elif setting_index == 7:
            result = [resnet101(pretrained=False), "Resnet101 random results"]
        elif setting_index == 8:
            result = [resnet152(pretrained=True), "Resnet152 pretrained results"]
        elif setting_index == 9:
            result = [resnet152(pretrained=False), "Resnet152 random results"]

    elif model_index == 3:

        if setting_index == 0:
            result = [VGG11Net(is_pretrained=True), "VGG11 pretrained results"]
        elif setting_index == 1:
            result = [VGG11Net(is_pretrained=False), "VGG11 random results"],
        elif setting_index == 2:
            result = [VGG11_bnNet(is_pretrained=True), "VGG11_bn pretrained results"]
        elif setting_index == 3:
            result = [VGG11_bnNet(is_pretrained=False), "VGG11_bn random results"]
        elif setting_index == 4:
            result = [VGG13Net(is_pretrained=True), "VGG13 pretrained results"]
        elif setting_index == 5:
            result = [VGG13Net(is_pretrained=False), "VGG13 random results"]
        elif setting_index == 6:
            result = [VGG13_bnNet(is_pretrained=True), "VGG13_bn pretrained results"]
        elif setting_index == 7:
            result = [VGG13_bnNet(is_pretrained=False), "VGG13_bn random results"]
        elif setting_index == 8:
            result = [VGG16Net(is_pretrained=True), "VGG16 pretrained results"]
        elif setting_index == 9:
            result = [VGG16Net(is_pretrained=False), "VGG16 random results"]
        elif setting_index == 10:
            result = [VGG16_bnNet(is_pretrained=True), "VGG16_bn pretrained results"]
        elif setting_index == 11:
            result = [VGG16_bnNet(is_pretrained=False), "VGG16_bn random results"]
        elif setting_index == 12:
            result = [VGG19Net(is_pretrained=True), "VGG19 pretrained results"]
        elif setting_index == 13:
            result = [VGG19Net(is_pretrained=False), "VGG19 random results"]
        elif setting_index == 14:
            result = [VGG19_bnNet(is_pretrained=True), "VGG19_bn pretrained results"]
        elif setting_index == 15:
            result = [VGG19_bnNet(is_pretrained=False), "VGG19_bn random results"]

    if result == 0:
        print("Warning: No network selected! UI will restart in 5 seconds!")
        time.sleep(5)
        main_ui()

    return result


def choose_model_main():

    network_table = create_pretty_table_networks()
    last_row = network_table[1]
    network_table = network_table[0]

    correct_input = False
    while not correct_input:
        clear = Clear()
        print(network_table)
        config_number = input("Please enter the ID of the Architecture from the table above:")

        if not represents_int(config_number):
            print("ERROR: Please enter an number!")
            time.sleep(5)
        elif (int(config_number) < 1) or (int(config_number) > last_row):
            print("ERROR: Please enter an number between 1 and " + str(last_row) + " !")
            time.sleep(5)
        else:
            correct_input = True

    model_id = config_number

    setting_table = create_pretty_table_network_configuration(model_id)
    last_row = setting_table[1]
    setting_table = setting_table[0]
    correct_input_2 = False

    while not correct_input_2:
        clear = Clear()

        print(setting_table)
        setting_number = input("Please enter the ID of the Architecture-Setting from the table above:")

        if not represents_int(setting_number):
            print("ERROR: Please enter an number!")
            time.sleep(5)
        elif (int(setting_number) < 1) or (int(setting_number) > last_row):
            print("ERROR: Please enter an number between 1 and " + str(last_row) + " !")
            time.sleep(5)
        else:
            correct_input_2 = True
            result = get_model_and_save_path(model_id,setting_number)
            model = result[0]
            save_path = result[1]

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
            if not represents_int(finished_check):
                finished = 1
            elif int(finished_check) != 1:
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