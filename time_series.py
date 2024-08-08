import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.signal import correlate, find_peaks
from tqdm.auto import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.fft import fft, ifft, fftshift, fftfreq  # FT stuff

mpl.use("pdf")
plt.style.use("dqc")

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
    def __init__(
        self, file_name, no_coeff, pixel_dim, saving_path, preprocessed_data=None
    ):
        self.file_name = file_name
        self.no_coeff = no_coeff
        self.pixels_dim = pixel_dim
        self.saving_path = saving_path
        self.preprocessed_data = (
            preprocessed_data  # New parameter for preprocessed data
        )

        if preprocessed_data is not None:
            self.data = preprocessed_data  # Use preprocessed data if provided
            self.mean_data = self.calculate_mean_from_preprocessed()
        else:
            self.data = self.import_h5py()  # Original data loading
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

    def calculate_mean_from_preprocessed(self):
        mean_data = []
        for data_types in tqdm(self.preprocessed_data):
            data_type = []
            for coeff_data in data_types:
                coeff_mean = []
                for value in coeff_data:
                    coeff_mean.append(np.mean(value))  # Simplified mean calculation
                data_type.append(coeff_mean)
            mean_data.append(data_type)

        mean_data = np.array(mean_data)
        return mean_data

    def plotting_mean_specturms(self, data):

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
                        f"test_coefficient_{i}&{i+1}_median_values_harmonics_r_{type}.pdf",
                    )
                    if os.path.exists(file_path):
                        os.remove(file_path)

                    plt.savefig(file_path)
                    plt.clf()
                    exit()
                if i == 30:
                    file_path = os.path.join(
                        self.saving_path,
                        f"coefficient_{i+1}_median_values_harmonics_r_{type}.pdf",
                    )
                    if os.path.exists(file_path):
                        os.remove(file_path)

                    plt.savefig(file_path)


class Dataset_analysis:
    def __init__(
        self, file_name, no_coeff, pixels, pixel_dim, saving_path, allpixel=False
    ):
        self.file_name = file_name
        self.no_coeff = no_coeff
        self.pixel = pixels
        self.row_no, self.col_no = self.pixel
        self.pixels_dim = pixel_dim
        self.pixels = self.pixel_check(pixelcheck=allpixel)
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

    def pixel_check(self, pixelcheck):
        if pixelcheck == True:
            idx_nos = []
            for idx_x, x in enumerate(self.row_no):
                for idx_y, y in enumerate(self.col_no):
                    idx_no = ((x * self.pixels_dim) + y) * sampling_rate
                    idx_nos.append(idx_no)
            return idx_nos

        else:
            idx_nos = []
            for idx_x, x in enumerate(self.row_no):
                for idx_y, y in enumerate(self.col_no):
                    if idx_y == idx_x:
                        idx_no = ((x * self.pixels_dim) + y) * sampling_rate
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

    @staticmethod
    def sin_fitting(x, y):
        from scipy import optimize

        def sin_function(x, a, b, c):
            sin = a * np.sin(b * np.array(x) + c)
            return sin

        amplitude_guess = (np.max(y) - np.min(y)) / 2
        major_freq = 220  # TODO: parameter to be passed in
        frequency_guess = major_freq * 2 * np.pi
        phase_guess = np.pi / 2

        initial_guesses = [amplitude_guess, frequency_guess, phase_guess]

        limits = (
            [0, frequency_guess * 0.5, -np.pi],  # Lower bounds
            [np.inf, frequency_guess * 1.5, np.pi],  # Upper bounds
        )

        try:
            params, _ = optimize.curve_fit(
                sin_function, x, y, p0=initial_guesses, bounds=limits
            )
            fitted_sin = sin_function(x, *params)
            return fitted_sin

        except ValueError as e:
            print(f"ValueError: {e}")
            print("Initial guesses or bounds might be incorrect.")
            exit()

    def fouriertransform_plot(
        self, data, index, index_pixel, _type, m, domain_limit=790
    ):
        if index != 0:
            fig, axs = plt.subplots(2, 2, figsize=(12, 8))

            n = len(data)
            Ts = 1 / 1600
            total_t = Ts * n
            t = np.arange(0, total_t, Ts)

            # First subplot: Original time-domain data
            axs[0, 0].plot(t, data)
            axs[0, 0].set_xlabel("Time (seconds)")
            axs[0, 0].set_ylabel("Amplitude")
            axs[0, 0].set_title("Original Time-Domain Signal")

            dataff = fft(data)
            dataf = abs(dataff)

            fs = 1 / Ts

            ft = fftfreq(n, Ts)
            ft = fftshift(ft)

            domain_mask = (ft >= -domain_limit) & (ft <= domain_limit)

            def select_top_m_indices(data, m):
                sorted_indices = np.argsort(data)[::-1]  # Descending order
                selected_indices = []
                for idx in sorted_indices:
                    if len(selected_indices) == m:
                        break
                    if all(abs(idx - sel_idx) > 1 for sel_idx in selected_indices):
                        selected_indices.append(idx)
                return np.array(selected_indices)

            # Apply mask to filter frequency domain
            filtered_dataf = dataf[domain_mask]
            filtered_ft = ft[domain_mask]

            half = len(filtered_ft) // 2

            neg_top_mag_i = select_top_m_indices(filtered_dataf[:half], m)
            pos_top_mag_i = select_top_m_indices(filtered_dataf[half:], m)

            pos_top_freq = filtered_ft[pos_top_mag_i + half]
            neg_top_freq = filtered_ft[neg_top_mag_i]

            # Second subplot: Magnitude spectrum
            axs[0, 1].plot(filtered_ft, filtered_dataf)
            axs[0, 1].set_xlabel("Frequency (Hz)")
            axs[0, 1].set_ylabel("Magnitude")
            axs[0, 1].set_title("Magnitude Spectrum")

            for freq in pos_top_freq:
                axs[0, 1].axvline(
                    x=freq, color="C0", linestyle="--", label=f"Pos Freq: {freq:.2f} Hz"
                )
            for freq in neg_top_freq:
                axs[0, 1].axvline(
                    x=freq, color="C1", linestyle="--", label=f"Neg Freq: {freq:.2f} Hz"
                )

            half = len(ft) // 2
            new_dataff = np.zeros_like(dataff)
            domain_diff = new_dataff.shape[0] - filtered_dataf.shape[0]

            new_dataff[domain_diff // 2 + 1 + neg_top_mag_i] = dataff[
                domain_diff // 2 + 1 + neg_top_mag_i
            ]
            new_dataff[0] = dataff[0]

            new_dataff[pos_top_mag_i + half] = dataff[pos_top_mag_i + half]

            # Third subplot: Major frequencies
            axs[1, 0].plot(ft[1:], abs(new_dataff[1:]))
            axs[1, 0].set_xlabel("Frequency (Hz)")
            axs[1, 0].set_ylabel("Magnitude")
            axs[1, 0].set_title("Major Frequencies")

            restored_time_data = ifft(new_dataff)
            restored_time_data = np.real(restored_time_data)

            # Fourth subplot: Restored time-domain data
            axs[1, 1].plot(t[1:], restored_time_data[1:])
            axs[1, 1].set_xlabel("Time (seconds)")
            axs[1, 1].set_ylabel("Amplitude")
            axs[1, 1].set_title("Restored Time-Domain Signal")

            fig.suptitle(f"FT Analysis for Correlation between 1 and {index + 1}")

            file_path = os.path.join(
                self.saving_path,
                f"Correlation_1&{index + 1}_{self.row_no[index_pixel]}{self.col_no[index_pixel]}_{_type}FTA_m={m}_domainlimit{domain_limit}.pdf",
            )
            if os.path.exists(file_path):
                os.remove(file_path)

            plt.tight_layout()
            plt.savefig(file_path)
            plt.clf()

    def fouriertransform(self, data, index, m, domain_limit=790):
        # m: Hyperparameter, number of frequencies to extract
        # domain_limit: Hyperparamter the range that FT is done #BUG: domain limiting not working

        if index != 0:
            n = len(data)
            Ts = 1 / 1600
            total_t = Ts * n
            t = np.arange(0, total_t, Ts)

            # Original time-domain data
            dataff = fft(data)
            dataf = abs(dataff)

            fs = 1 / Ts

            ft = fftfreq(n, Ts)
            ft = fftshift(ft)

            domain_mask = (ft >= -domain_limit) & (ft <= domain_limit)

            def select_top_m_indices(data, m):
                sorted_indices = np.argsort(data)[::-1]  # Descending order
                selected_indices = []
                for idx in sorted_indices:
                    if len(selected_indices) == m:
                        break
                    if all(abs(idx - sel_idx) > 1 for sel_idx in selected_indices):
                        selected_indices.append(idx)
                return np.array(selected_indices)

            # Apply mask to filter frequency domain
            filtered_dataf = dataf[domain_mask]
            filtered_ft = ft[domain_mask]
            filtered_dataff = dataff[domain_mask]

            # Magnitude spectrum

            half = len(filtered_ft) // 2

            neg_top_mag_i = select_top_m_indices(filtered_dataf[:half], m)
            pos_top_mag_i = select_top_m_indices(filtered_dataf[half:], m)

            pos_top_freq = filtered_ft[pos_top_mag_i + half]
            # neg_top_freq = filtered_ft[neg_top_mag_i]

            # Major frequencies

            half = len(ft) // 2
            new_dataff = np.zeros_like(dataff)
            domain_diff = new_dataff.shape[0] - filtered_dataf.shape[0]

            new_dataff[domain_diff // 2 + 1 + neg_top_mag_i] = dataff[
                domain_diff // 2 + 1 + neg_top_mag_i
            ]
            new_dataff[0] = dataff[0]

            new_dataff[pos_top_mag_i + half] = dataff[pos_top_mag_i + half]

            # Restored time-domain data
            restored_time_data = ifft(new_dataff)
            restored_time_data = np.real(restored_time_data)

            return restored_time_data

    def extract_peak_and_neighbors(self, coefficient, peaks, radius=1):

        peak_data = []
        length = len(coefficient)

        for i in peaks:
            start_index = max(0, i - radius)
            end_index = min(length, i + radius + 1)
            peak_data.append(coefficient[start_index:end_index])

        return peak_data

    def cross_correlation(self, plotting=False):
        # sliced_data = []
        row_number = 0
        col_number = 0
        for index_pixel, pixel in enumerate(self.data):

            if index_pixel % self.pixels_dim == 0 and index_pixel != 0:
                row_number += 1
                col_number = 0

            col_number += 1

            indexdata = []
            for index_type, datatype in enumerate(pixel):

                if index_type == 0:
                    _type = "Real"
                if index_type == 1:
                    _type = "Complex"

                COEFFICIENT_1 = datatype[0]
                count_max = len(datatype[1]) + 1
                counts = [i for i in range(1, count_max)]
                PLOT_INDEX = 1

                _, axs = plt.subplots(2, 3, figsize=(10, 8))

                for index, coefficient in enumerate(datatype):

                    correlation_values = correlate(
                        COEFFICIENT_1, coefficient, mode="same"
                    )

                    correlation_values = correlation_values - COEFFICIENT_1

                    if index != 0:
                        peaks = 1

                        FT_values = self.fouriertransform(
                            correlation_values, index, m=peaks
                        )
                        # TODO: SAVE_POINT = FT_values

                        wave_peaks, _ = find_peaks(FT_values)
                        peak_diffs = np.diff(wave_peaks)

                        if plotting:
                            axs[(PLOT_INDEX - 1) // 3, (PLOT_INDEX - 1) % 3].plot(
                                counts,
                                FT_values - np.mean(FT_values),
                                label=f"Coefficient_{index + 1}",
                            )
                            axs[(PLOT_INDEX - 1) // 3, (PLOT_INDEX - 1) % 3].plot(
                                counts,
                                correlation_values,
                                label=f"Coefficient_{index + 1}FT",
                            )
                            axs[(PLOT_INDEX - 1) // 3, (PLOT_INDEX - 1) % 3].plot(
                                counts,
                                coefficient,  # FT_values
                                label=f"Coefficient_{index + 1}FT",
                            )

                            for i in wave_peaks:
                                axs[
                                    (PLOT_INDEX - 1) // 3, (PLOT_INDEX - 1) % 3
                                ].axvline(
                                    x=i,
                                    color="r",
                                    linestyle="--",
                                    label="Peak" if i == wave_peaks[0] else "",
                                )

                            axs[(PLOT_INDEX - 1) // 3, (PLOT_INDEX - 1) % 3].set_xlabel(
                                "Index"
                            )
                            axs[(PLOT_INDEX - 1) // 3, (PLOT_INDEX - 1) % 3].set_ylabel(
                                f"Correlation with Coefficient {index + 1} from FT"
                            )
                            PLOT_INDEX += 1

                            if index % 6 == 0:
                                plt.suptitle(
                                    f"Correlation between coefficient 1 and {index-4}-{index+1}"
                                )
                                file_path = os.path.join(
                                    self.saving_path,
                                    f"Peak_extraction_{index+1}_{self.row_no[index_pixel]}{self.col_no[index_pixel]}_{_type}_comparsion_{peaks}peaks.pdf",
                                )
                                if os.path.exists(file_path):
                                    os.remove(file_path)

                                plt.tight_layout()
                                plt.savefig(file_path)
                                PLOT_INDEX = 1
                                _, axs = plt.subplots(
                                    2, 3, figsize=(15, 10)
                                )  # Reset figure and axes

                        datap = []
                        if len(wave_peaks) > 0:
                            radius = 1  # Hyperparameter for number of indices for slice
                            datap = self.extract_peak_and_neighbors(
                                coefficient, wave_peaks, radius
                            )
                        datap = [item for sublist in datap for item in sublist]
                        indexdata.append(datap)

                    hdf5_file_path = os.path.join(self.saving_path, f"Sliced_data_FTpeaks{peaks}.h5")
                    with h5py.File(hdf5_file_path, "w") as hdf5_file:
                        hdf5_file.create_dataset(
                            f"Pixel_{row_number}{col_number}",
                            data=np.array(indexdata),
                        )

        #         sliced_data.append(indexdata)

        # sliced_data = np.array(sliced_data)
        # print(sliced_data.shape)

        # return sliced_data


if __name__ == "__main__":

    file = "./Data/Measurement_of_2021-06-18_1825.h5"
    path = r"C:\Users\User\Documents\2024\project\test_figures"
    sampling_rate = 160
    PIXEL_DIM = 50
    NO_COEFF = 31

    PIXEL = [[2], [1]]
    PIXEL2 = [[3], [4]]
    PIXELR = [np.random.randint(2, 49, size=13).tolist() for _ in range(2)]
    idx = np.arange(PIXEL_DIM)
    ALLPIXEL = [idx.tolist(), idx.tolist()]

    print(ALLPIXEL)

    # for i in range(len(PIXELR[0])):
    #     folder_name = f"Pixel_{PIXELR[0][i]},{PIXELR[1][i]}"
    #     directory_name = os.path.join(path, folder_name)
    #     if not os.path.exists(directory_name):
    #         os.makedirs(directory_name)

    #     dataset = Dataset_analysis(
    #         file_name=file,
    #         pixels=[[PIXELR[0][i]], [PIXELR[1][i]]],
    #         no_coeff=NO_COEFF,
    #         pixel_dim=PIXEL_DIM,
    #         saving_path=directory_name,
    #     )

    #     dataset.cross_correlation()
    #     # dataset.time_series_plot()

    folder_name = f"Pixel_{PIXEL[0][0]},{PIXEL[1][0]}"
    directory_name = os.path.join(path, folder_name)
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    dataset = Dataset_analysis(
        file_name=file,
        pixels=ALLPIXEL,
        no_coeff=NO_COEFF,
        pixel_dim=PIXEL_DIM,
        saving_path=directory_name,
        allpixel=True,
    )

    slice_data = (
        dataset.cross_correlation()
    )  # TODO: take sliced data for all pixels and plot each spectrum.
