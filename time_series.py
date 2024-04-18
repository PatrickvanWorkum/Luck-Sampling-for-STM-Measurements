import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson
from tqdm.auto import tqdm

directory_path = [
    r"C:\Users\User\Documents\2024\project",
    r"C:\Users\Patrick Van Workum\Documents\Python scripts",
]


for file in directory_path:
    try:
        os.chdir(file)
    except FileNotFoundError:
        print("Not Valid file path")


class original_plotting:
    def __init__(self, file_name, no_coeff, pixel_dim, saving_path):
        self.file_name = file_name
        self.no_coeff = no_coeff
        self.pixels_dim = pixel_dim
        self.data = self.import_h5py()
        self.saving_path = saving_path
        self.mean_data = self.o_mean()

    def import_h5py(self):
        moduli = []
        real = []
        complex = []
        with h5py.File(self.file_name, "r") as hf:
            for i in range(self.no_coeff):
                moduli.append(hf[f"/Coefficient_index_{i+1}/Modulus_coeff_{i+1}"][()])
                real.append(hf[f"/Coefficient_index_{i+1}/Real_coeff_{i+1}"][()])
                complex.append(hf[f"/Coefficient_index_{i+1}/Complex_coeff_{i+1}"][()])

        data = [moduli] + [real] + [complex]
        data = np.array(data)

        return data

    def o_mean(self):
        mean_data = []
        count = 0
        sums = 0
        for data_types in tqdm(self.data):
            data_type = []
            for coeff_data in data_types:
                coeff_mean = []
                for value in coeff_data:
                    sums += value
                    count += 1
                    if count == sampling_rate:
                        coeff_mean.append(sums / count)
                        count, sums = 0, 0

                data_type.append(coeff_mean)

            mean_data.append(data_type)

        mean_data = np.array(mean_data)
        return mean_data

    def plotting_mean_specturms(self):

        ticks = (1, 10, 20, 30, 40, 50)

        for count, data_type in enumerate(self.mean_data[1:, :, :]):

            if count == 0:
                type = "real"
            elif count == 1:
                type = "complex"

            r_data = [
                np.reshape(sublist, (self.pixels_dim, self.pixels_dim))
                for sublist in data_type
            ]

            if not os.path.exists(self.saving_path):
                os.makedirs(self.saving_path)

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
                    extent=(0.5, self.pixels_dim + 0.5, 0.5, self.pixels_dim + 0.5),
                )
                plt.colorbar()
                plt.title(f"Coefficients {i+1}")
                plt.xlabel("Pixels")
                plt.ylabel("Pixels")
                plt.xticks(ticks, ticks)
                plt.xlim(1, self.pixels_dim)
                plt.yticks(ticks, ticks)
                plt.ylim(1, self.pixels_dim)
                plt.tight_layout()
                plt.grid(visible=False)

                if i % 2 == 1:
                    file_path = os.path.join(
                        self.saving_path,
                        f"coefficient_{i}&{i+1}_median_values_harmonics_r_{type}.png",
                    )
                    if os.path.exists(file_path):
                        os.remove(file_path)

                    plt.savefig(file_path)
                    plt.clf()
                if i == 30:
                    file_path = os.path.join(
                        self.saving_path,
                        f"coefficient_{i+1}_median_values_harmonics_r_{type}.png",
                    )
                    if os.path.exists(file_path):
                        os.remove(file_path)

                    plt.savefig(file_path)


class Dataset_analysis:
    def __init__(self, file_name, no_coeff, pixels, pixel_dim, saving_path):
        self.file_name = file_name
        self.no_coeff = no_coeff
        self.pixel = pixels
        self.row_no, self.col_no = self.pixel
        self.pixels_dim = pixel_dim
        self.pixels = self.pixel_check()
        self.data = self.import_h5py()
        self.saving_path = saving_path

    def import_h5py(self):
        moduli = []
        real = []
        complex = []
        with h5py.File(self.file_name, "r") as hf:
            for i in range(self.no_coeff):
                moduli.append(hf[f"/Coefficient_index_{i+1}/Modulus_coeff_{i+1}"][()])
                real.append(hf[f"/Coefficient_index_{i+1}/Real_coeff_{i+1}"][()])
                complex.append(hf[f"/Coefficient_index_{i+1}/Complex_coeff_{i+1}"][()])

        data = [moduli] + [real] + [complex]
        data = np.array(data)

        pixel_slices = []
        for pixel in self.pixels:
            slices = []
            for i in range(data.shape[1]):
                sliced_data = data[
                    1:, i, pixel : pixel + 160
                ]  # change to including moduli
                slices.append(sliced_data)

            pixel_slices.append(slices)

        data = np.array(pixel_slices)
        data = np.transpose(data, (0, 2, 1, 3))

        return data

    def pixel_check(self):
        idx_nos = []
        for idx_x, x in enumerate(self.row_no):
            for idx_y, y in enumerate(self.col_no):
                if idx_y == idx_x:
                    idx_no = (((x - 1) * self.pixels_dim) + (y - 1)) * sampling_rate
                    idx_nos.append(idx_no)

        return idx_nos

    def time_series_plot(self):
        for idx, pixel in enumerate(self.data):
            for j, data_type in enumerate(pixel):

                if j == 0:
                    type = "real"
                elif j == 1:
                    type = "complex"

                for i, coeff in enumerate(data_type):

                    plt.subplot(2, 2, i % 4 + 1)

                    x = np.linspace(start=0, stop=len(coeff), num=len(coeff))
                    plt.plot(x, coeff)
                    plt.xlabel("Index")
                    plt.ylabel("Values")
                    plt.title(
                        f"Coef {i+1}, Pixel: row {self.row_no[idx]}, col {self.col_no[idx]}"
                    )

                    if (i + 1) % 4 == 0:
                        plt.tight_layout()
                        file_path = os.path.join(
                            self.saving_path,
                            f"coefficient_{i-2}&{i-1}&{i}&{i+1}_{type}_pixel{self.row_no[idx]}{self.col_no[idx]}.png",
                        )
                        if os.path.exists(file_path):
                            os.remove(file_path)

                        plt.savefig(file_path)
                        plt.clf()

                    if i == 30:
                        plt.tight_layout()
                        file_path = os.path.join(
                            self.saving_path,
                            f"coefficient_{i-2}&{i-1}&{i}&{i+1}_{type}_pixel{self.row_no[idx]}{self.col_no[idx]}.png",
                        )
                        if os.path.exists(file_path):
                            os.remove(file_path)

                        plt.savefig(file_path)
                        plt.clf()

                    # if i == 3:
                    #     plt.tight_layout()
                    #     file_path = os.path.join(
                    #         self.saving_path,
                    #         f"coefficient_{i-2}&{i-1}&{i}&{i+1}_{type}_pixel{self.row_no[idx]}{self.col_no[idx]}.png",
                    #     )
                    #     if os.path.exists(file_path):
                    #         os.remove(file_path)

                    #     plt.savefig(file_path)
                    #     plt.clf()
                    #     break

    def plot_binned_data(self):
        BIN_NO = 20

        for idx, pixel in enumerate(self.data):
            for j, pixel_data in enumerate(pixel):

                if j == 0:
                    type = "real"
                elif j == 1:
                    type = "complex"

                plt.figure(figsize=(8, 6))

                for i, coeff in tqdm(enumerate(pixel_data)):
                    bin_edges = np.linspace(min(coeff), max(coeff), BIN_NO)
                    mean, variance = norm.fit(coeff)

                    plt.subplot(1, 3, i % 3 + 1)
                    plt.hist(coeff, bins=bin_edges, edgecolor="black", alpha=0.75)

                    xmin, xmax = plt.xlim(mean - 5 * variance, mean + 5 * variance)

                    x = np.linspace(xmin, xmax, 100)
                    npdf = (
                        norm.pdf(x, mean, variance) * len(coeff) * np.diff(bin_edges)[0]
                    )

                    plt.plot(x, npdf, "r-", lw=2, label="Gaussian fit", alpha=0.5)

                    plt.axvline(mean, lw=2, label="Mean", color="grey", linestyle="--")

                    plt.xlabel("Binned snapshot values")
                    plt.ylabel("Frequency")
                    plt.title(f"Coefficient_{i+1}")
                    plt.tight_layout()
                    plt.legend()

                    if (i + 1) % 3 == 0:
                        file_path = os.path.join(
                            self.saving_path,
                            f"coefficient_{i-1}&{i}&{i+1}_{type}_histogram_pixel_{self.row_no[idx]}{self.col_no[idx]}.png",
                        )
                        if os.path.exists(file_path):
                            os.remove(file_path)

                        plt.savefig(file_path)
                        plt.clf()

                    if i == 30:
                        file_path = os.path.join(
                            self.saving_path,
                            f"coefficient_{i+1}_{type}_histogram_pixel_{self.row_no[idx]}{self.col_no[idx]}.png",
                        )
                        if os.path.exists(file_path):
                            os.remove(file_path)

                        plt.savefig(file_path)

                        plt.clf()

    def correlation(self):
        for index_pixel, pixel in enumerate(self.data):

            for index_type, datatype in enumerate(pixel):

                if index_type == 0:
                    type = "Real"
                if index_type == 1:
                    type = "Complex"

                COEFFICIENT_1 = datatype[0]
                count_max = len(datatype[1]) + 1
                counts = [i for i in range(1, count_max)]

                for index, coefficient in enumerate(datatype):
                    correlation = coefficient - COEFFICIENT_1
                    if index == 0:
                        pass
                    else:
                        plt.plot(counts, correlation, label=f"Coefficient_{index+1}")

                        if index % 5 == 0:
                            plt.xlabel("Index")
                            plt.ylabel("Correlation")
                            plt.legend(fontsize="x-small")
                            plt.title(
                                f"Correlation between coefficent 1 and {index-3}-{index+1}"
                            )
                            file_path = os.path.join(
                                self.saving_path,
                                f"Correlation_1_&_{index-3}-{index+1}_{self.row_no[index_pixel]}{self.col_no[index_pixel]}_{type}.png",
                            )
                            if os.path.exists(file_path):
                                os.remove(file_path)

                            plt.savefig(file_path)
                            plt.clf()

        print("All correlations are calculation.")


if __name__ == "__main__":

    file = "./Data/Measurement_of_2021-06-18_1825.h5"
    path = r"C:\Users\User\Documents\2024\project\test_figures"
    sampling_rate = 160
    PIXEL_DIM = 50
    NO_COEFF = 31

    PIXEL = [[2, 2], [1, 2]]
    PIXEL2 = [[3], [4]]
    PIXELR = [np.random.randint(2, 49, size=13).tolist() for _ in range(2)]

    for i in range(len(PIXELR[0])):
        folder_name = f"Pixel_{PIXELR[0][i]},{PIXELR[1][i]}"
        directory_name = os.path.join(path, folder_name)
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)

        dataset = Dataset_analysis(
            file_name=file,
            pixels=[[PIXELR[0][i]], [PIXELR[1][i]]],
            no_coeff=NO_COEFF,
            pixel_dim=PIXEL_DIM,
            saving_path=directory_name,
        )

        dataset.correlation()
        dataset.time_series_plot()
