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


class Dataset:
    def __init__(self, file_name, no_coeff, pixels, pixel_dim, saving_path):
        self.file_name = file_name
        self.no_coeff = no_coeff
        self.pixel = pixels
        self.row_no, self.col_no = self.pixel
        self.pixels_dim = pixel_dim
        self.data = self.import_h5py()
        self.pixels = self.pixel_check()
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
        pixel_slices = []
        for pixel in self.pixels:
            slices = []
            for i in range(self.data.shape[1]):
                sliced_data = self.data[1:, i, pixel : pixel + 160]
                slices.append(sliced_data)

            pixel_slices.append(slices)

        pixel_time_series = np.array(pixel_slices)
        pixel_time_series = np.transpose(pixel_time_series, (0, 2, 1, 3))

        for idx, pixel in enumerate(pixel_time_series):
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
                    plt.title(f"Coefficient_{i+1}")

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

    def plot_binned_data(self):
        BIN_NO = 20

        pixel_slices = []
        for pixel in self.pixels:
            slices = []
            for i in range(self.data.shape[1]):
                sliced_data = self.data[1:, i, pixel : pixel + 160]
                slices.append(sliced_data)

            pixel_slices.append(slices)

        pixel_time_series = np.array(pixel_slices)
        pixel_time_series = np.transpose(pixel_time_series, (0, 2, 1, 3))

        for idx, pixel in enumerate(pixel_time_series):
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
                            f"coefficient_{i-1}&{i}&{i+1}_{type}_histogram_pixel_{self.row_no[idx]}{self.col_no[idx]}_complex.png",
                        )
                        if os.path.exists(file_path):
                            os.remove(file_path)

                        plt.savefig(file_path)
                        plt.clf()

                    if i == 30:
                        file_path = os.path.join(
                            self.saving_path,
                            f"coefficient_{i+1}_{type}_histogram_pixel_{self.row_no[idx]}{self.col_no[idx]}_complex.png",
                        )
                        if os.path.exists(file_path):
                            os.remove(file_path)

                        plt.savefig(file_path)

                        plt.clf()


if __name__ == "__main__":

    file = "./Data/Measurement_of_2021-06-18_1825.h5"
    path = r"C:\Users\User\Documents\2024\project\test_figures"
    PIXEL = [[2, 2, 3], [1, 2, 3]]
    PIXEL2 = [[2], [1]]
    sampling_rate = 160
    PIXEL_DIM = 50
    NO_COEFF = 31

    dataset = Dataset(
        file_name=file,
        pixels=PIXEL,
        no_coeff=NO_COEFF,
        pixel_dim=PIXEL_DIM,
        saving_path=path,
    )

    dataset.plot_binned_data()
