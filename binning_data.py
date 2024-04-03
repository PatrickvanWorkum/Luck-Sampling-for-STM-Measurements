# This has been moved to new class set up.


import os
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson
import numpy as np
from tqdm.auto import tqdm
import h5py

directory_path = [
    r"C:\Users\User\Documents\2024\project",
    r"C:\Users\Patrick Van Workum\Documents\Python scripts",
]


for file in directory_path:
    try:
        os.chdir(file)
    except FileNotFoundError:
        print("Not Valid file path")


def import_h5py(input_file, no_coeff):
    moduli = []
    real = []
    complex = []
    with h5py.File(input_file, "r") as hf:
        for i in range(no_coeff):
            moduli.append(hf[f"/Coefficient_index_{i+1}/Modulus_coeff_{i+1}"][()])
            real.append(hf[f"/Coefficient_index_{i+1}/Real_coeff_{i+1}"][()])
            complex.append(hf[f"/Coefficient_index_{i+1}/Complex_coeff_{i+1}"][()])

    data = [moduli] + [real] + [complex]
    data = np.array(data)
    return data


def extract_per_pixel(data, pixels_co, sampling_rate=160):
    multi = False
    if len(pixels_co[0]) != 1:
        multi = True

    row_no, col_no = pixels_co
    idx_nos = []
    for idx_x, x in enumerate(row_no):
        for idx_y, y in enumerate(col_no):
            if idx_y == idx_x:
                idx_no = (((x - 1) * PIXEL_DIM) + (y - 1)) * sampling_rate
                idx_nos.append(idx_no)

    data_filtered = []
    for coeff in data:
        pix_val = []
        for idx, _ in enumerate(coeff):
            if idx in idx_nos:
                for offset in range(sampling_rate):
                    pix_val.append(coeff[idx + offset])
        data_filtered.append(pix_val)

    return data_filtered, multi, idx_nos


def plot_binned_data(data, MULTI_PIXEL_CHECK, saving_path):
    BIN_NO = 20

    if MULTI_PIXEL_CHECK == False:
        plt.figure(figsize=(8, 6))

        for i, coeff in tqdm(enumerate(data)):
            bin_edges = np.linspace(min(coeff), max(coeff), BIN_NO)
            mean, variance = norm.fit(coeff)

            plt.subplot(1, 3, i % 3 + 1)
            plt.hist(coeff, bins=bin_edges, edgecolor="black", alpha=0.75)

            xmin, xmax = plt.xlim(mean - 5 * variance, mean + 5 * variance)

            x = np.linspace(xmin, xmax, 100)
            npdf = norm.pdf(x, mean, variance) * len(coeff) * np.diff(bin_edges)[0]

            plt.plot(x, npdf, "r-", lw=2, label="Gaussian fit", alpha=0.5)

            plt.axvline(mean, lw=2, label="Mean", color="grey", linestyle="--")

            plt.xlabel("Binned snapshot values")
            plt.ylabel("Frequency")
            plt.title(f"Coefficient_{i+1}")
            plt.tight_layout()
            plt.legend()

            if (i + 1) % 3 == 0:
                plt.show()
                file_path = os.path.join(
                    saving_path,
                    f"coefficient_{i-1}&{i}&{i+1}_histogram_pixel_{PIXEL[0]}{PIXEL[1]}_bin{BIN_NO}_complex.png",
                )
                if os.path.exists(file_path):
                    os.remove(file_path)

                plt.savefig(file_path)
                plt.clf()
                # implement multi pixel plotting
                pass

            if i == 30:
                plt.show()
                file_path = os.path.join(
                    saving_path,
                    f"coefficient_{i+1}_histogram_pixel_{PIXEL[0]}{PIXEL[1]}_bin{BIN_NO}_complex.png",
                )
                if os.path.exists(file_path):
                    os.remove(file_path)

                plt.savefig(file_path)

                plt.clf()
    else:
        print("Need to implement this functionality")


if __name__ == "__main__":

    file = "./Data/Measurement_of_2021-06-18_1825.h5"
    path = r"C:\Users\User\Documents\2024\project\figures"
    PIXEL = [[2], [1]]
    PIXEL_DIM = 50
    NO_COEFF = 31

    data = import_h5py(input_file=file, no_coeff=NO_COEFF)

    real_f_data = extract_per_pixel(data[1], PIXEL)
    print(real_f_data[2])
    complex_f_data = extract_per_pixel(data[2], PIXEL)

    # plot_binned_data(real_f_data[0], real_f_data[1], path)
    plot_binned_data(complex_f_data[0], complex_f_data[1], path)
