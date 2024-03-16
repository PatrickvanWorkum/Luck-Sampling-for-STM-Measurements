import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import h5py
import re

directory_path = r"C:\Users\User\Documents\2024\project"
os.chdir(directory_path)

pixel_dim = 50


def import_h5py(input_file):
    data = []
    with h5py.File(input_file, "r") as hf:
        for key in hf.keys():
            dataset = hf[key]
            for subkey in dataset.keys():
                data.append(dataset[subkey][:])

    return data


def o_mean(data, sampling_rate=160):
    mean_data = []
    num_coeff = 31
    count = 0
    sums = 0

    for sublist in tqdm(data):
        coeff_mean = []
        for i in sublist:
            sums += i
            count += 1

            if count == sampling_rate:
                coeff_mean.append(sums / count)
                count, sums = 0, 0
        mean_data.append(coeff_mean)

    if len(mean_data) == num_coeff:
        print("Have all coefficient means calculated for each pixel")
    else:
        print("Mean data is missing")

    return mean_data


def plotting_mean_specturms(data, saving_path):

    pixel_dim = 50
    ticks = (1, 10, 20, 30, 40, 50)

    r_data = [np.reshape(sublist, (pixel_dim, pixel_dim)) for sublist in data]

    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    plt.figure(figsize=(8, 6))
    for i, values in tqdm(enumerate(r_data)):

        min_parameter = 0
        max_parameter = np.median(values) + 3 * np.std(values)

        plt.subplot(1, 2, i % 2 + 1)
        plt.imshow(
            values[::-1],
            vmin=min_parameter,
            vmax=max_parameter,
            cmap="gray",
            extent=(0.5, pixel_dim + 0.5, 0.5, pixel_dim + 0.5),
        )
        plt.colorbar()
        plt.title(f"Coefficients {i+1}")
        plt.xlabel("Pixels")
        plt.ylabel("Pixels")
        plt.xticks(ticks, ticks)
        plt.xlim(1, pixel_dim)
        plt.yticks(ticks, ticks)
        plt.ylim(1, pixel_dim)
        plt.tight_layout()
        plt.grid(visible=False)

        if i % 2 == 1:
            plt.show()
            file_path = os.path.join(
                saving_path, f"coefficient_{i}&{i+1}_median_values_harmonics_r.png"
            )
            if os.path.exists(file_path):
                os.remove(file_path)

            plt.savefig(file_path)
            plt.clf()
        if i == 30:
            file_path = os.path.join(
                saving_path, f"coefficient_{i+1}_median_values_harmonics_r.png"
            )
            if os.path.exists(file_path):
                os.remove(file_path)

            plt.savefig(file_path)


if __name__ == "__main__":

    file = "./Data/Measurement_of_2021-06-18_1825.h5"
    path = r"C:\Users\User\Documents\2024\project\figures"

    data = import_h5py(file)
    mean = o_mean(data)
    plotting_mean_specturms(mean, path)


"""
Notes for project;
160 sample per pixels
need to group together all real and imaginary coefficients # Q: what is the difference again of the real and imaginary parts
plot first imaginary => should look like fish tooth
histogram/binning of the individuals to see distribution and statistic testing to see shapes
binning only one pixel 
cumaltive histogram plotting 
FT coefficients
"""

"""
    How to generalize this way more
    # # IMPLEMENT classes for different stats to plot colourmap with; mean, std, median, mode, range

    # def color_range(data):
    #     min_input, max_input = input(
    #         "Please enter what parameters would you like to the min and max of the colour: "
    #     )

    #     min_in = re.findall(r"\d+", min_input)

    #     if min_input in ["mean", "Mean"]:
    #         min_limit = np.mean(data)
    #     elif min_input in ["Mean+3sd", "Mean-3sd"]:
    #         min_limit = np.mean(data) - 3 * np.std(data)
    #     elif min_input in ["Mean+2sd", "Mean-2sd"]:
    #         min_limit = np.mean(data) - 3 * np.std(data)

    #     return min_limit, max_limit
"""
