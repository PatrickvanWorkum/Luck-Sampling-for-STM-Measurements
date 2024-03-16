import h5py
import numpy as np
import os
from tqdm.auto import tqdm

directory_path = r"C:\Users\User\Documents\2024\project"
os.chdir(directory_path)


def read_batch(file_path, start_cut, end_cut):
    with open(file_path, "r") as file:
        txt_lines = file.readlines()[start_cut:end_cut]

    return txt_lines


def extract_coeff(data):  # is good, but would be better if faster
    coeff_index = list(range(1, 32))
    coeff = [[] for _ in coeff_index]

    for i, line in tqdm(list(enumerate(data))):
        if line.startswith("# "):
            for j, index in list(enumerate(coeff_index, start=1)):
                values = [float(num) for num in data[i + index + 1].split()[:2]]
                coeff[j - 1].append(values)

    if len(coeff) == 31:
        print(
            f"Successful separated coefficients. The number of coefficient separated into both their real and imaginary is {len(coeff)}"
        )

    if len(coeff[1]) == 50 * 50 * 160:
        print(
            f"Successful separated coefficients. The number of values for coefficient one for both real and imaginary is {len(coeff[0])}"
        )

    return coeff


def data_processing(data):  # works

    modulus_data = []

    for sublist in data:
        modulus_sublist = [np.sqrt((i[0]) ** 2 + (i[1]) ** 2) for i in sublist]
        modulus_data.append(modulus_sublist)

    print("Finished modulus calculations")
    return modulus_data


def saving_h5py(modulus_data, output_file):

    with h5py.File(output_file, "w") as hf:
        for i, moduli in enumerate(modulus_data):
            grp = hf.create_group(f"Cofficient_index_{i+1}")
            grp.create_dataset(f"Moduli_coeff_{i+1}", data=moduli)

            coeff_t = 30
            if i == coeff_t:
                print(f"Saved all coefficient {i+1}")


def read_h5py(input_file):

    with h5py.File(input_file, "r") as hf:
        for key in hf.keys():
            print(f"Dataset: {key}")
            dataset = hf[key]
            for subkey in dataset.keys():
                data = dataset[subkey][:]
                print(f"First 5 values of {subkey}: {data[:5]}")


def create_data():
    file = "./Data/Measurement_of_2021-06-18_1825.txt"
    start_line, end_line = 14, 14000014
    txt_lines = read_batch(file, start_line, end_line)

    coeff = extract_coeff(txt_lines)

    modulus_data = data_processing(coeff)

    output_directory = "./Data"
    output_file = os.path.join(output_directory, "Measurement_of_2021-06-18_1825.h5")
    saving_h5py(modulus_data, output_file)
    return output_file


def ask_user_action():
    while True:
        user_input = input(
            "Enter 'read' to read data or 'create' to create data: "
        ).lower()
        if user_input in ["R", "r", "read"]:
            file = input("Enter what is the h5 file that you would like to read: ")
            read_h5py(file)
        elif user_input in ["create", "C", "c"]:
            file = input("Enter what txt file would you like to convert to a h5 file: ")
            output_file = create_data()
            print(f"A new h5 file has be create called {output_file}")
        elif user_input in ["E", "end", "e"]:
            print("Ending program")
            break
        else:
            print("Error: Invalid input. Please enter 'read' or 'create'.")


if __name__ == "__main__":
    ask_user_action()
