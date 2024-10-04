import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.signal import correlate, find_peaks
from tqdm.auto import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.fft import fft, ifft, fftfreq, fft2, fftshift  # FT stuff

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
        self,
        file_name,
        no_coeff,
        pixel_dim,
        saving_path,
        preprocessed_data=None,
    ):
        self.file_name = file_name
        self.no_coeff = no_coeff
        self.pixels_dim = pixel_dim
        self.saving_path = saving_path
        self.preprocessed_data = (
            preprocessed_data  # New parameter for preprocessed data
        )

        if preprocessed_data == None:
            self.data = preprocessed_data  # Use preprocessed data if provided
            self.measure_data = self.calculate_mean_from_preprocessed()
        elif preprocessed_data == "2FT":
            self.data = self.import_h5py()  # Original data loading
            self.measure_data = self.FT2D()
        elif preprocessed_data == "2FT_DICE":
            self.data = self.import_h5py()  # Original data loading
            self.measure_data = self.FT2D_DICE_value()
        else:
            self.data = self.import_h5py()  # Original data loading
            self.measure_data = self.measure_cal()

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

    @staticmethod
    def reject_outliers(data, m=3):
        return data[abs(data - np.mean(data)) < m * np.std(data)]

    @staticmethod
    def dice_coefficient(A, B):
        A = np.asarray(A, dtype=float)  # predicted
        B = np.asarray(B, dtype=float)  # ground truth

        sum_A = np.sum(A)
        sum_B = np.sum(B)

        overlap = np.sum(np.minimum(A, B))

        if sum_A + sum_B == 0:
            return 1.0

        dice = 2 * overlap / (sum_A + sum_B)
        return dice

    def measure_cal(self):
        measure_data = []
        for data_types in self.data:
            data_type = []
            for coeff_data in data_types:
                coeff_measure = []
                pixeldata = [
                    coeff_data[i : i + sampling_rate]
                    for i in range(0, len(coeff_data), sampling_rate)
                ]
                for values in pixeldata:

                    # measure = np.mean(values) # mean
                    # measure = np.median(values) # median
                    # measure = np.sqrt(sum(x**2 for x in values) / len(values))  # RMS
                    rm_values = original_plotting.reject_outliers(values)
                    measure = np.mean(rm_values)  # RMOutlier

                    coeff_measure.append(measure)
                data_type.append(coeff_measure)

            measure_data.append(data_type)
        measure_data = np.array(measure_data)
        return measure_data

    def FT2D(self):
        FTdata = []
        for data_types in self.data:
            data_type = []
            for coeff_data in data_types:
                coeff_measure = []
                pixeldata = [
                    coeff_data[i : i + sampling_rate]
                    for i in range(0, len(coeff_data), sampling_rate)
                ]
                pixeldata = np.array(pixeldata)
                pixeldata = pixeldata[:, :slicefactor]
                for values in pixeldata:
                    measure = np.mean(values)
                    coeff_measure.append(measure)
                coeff_measure = np.array(coeff_measure)
                coeff_2FT = fft2(
                    coeff_measure.reshape(self.pixels_dim, self.pixels_dim)
                )

                coeff_2FT = fftshift(coeff_2FT)
                coeff_2FT = abs(coeff_2FT)
                data_type.append(coeff_2FT)

            FTdata.append(data_type)
        FTdata = np.array(FTdata)
        return FTdata

    def FT2D_DICE_value(self):
        plt.figure(figsize=(10, 6))
        dice_values = []
        for count, data_types in enumerate(self.data):
            if count == 0:
                type = "real"
            elif count == 1:
                type = "complex"
            else:
                break

            for coeff_data in data_types:
                pixeldata = [
                    coeff_data[i : i + sampling_rate]
                    for i in range(0, len(coeff_data), sampling_rate)
                ]
                pixeldata = np.array(pixeldata)
                pixeldata_slice = pixeldata[:, slicefactor:]

                coeff_measure = []
                for values in pixeldata:
                    measure = np.mean(values)
                    coeff_measure.append(measure)
                coeff_measure = np.array(coeff_measure)
                coeff_2FT = fft2(
                    coeff_measure.reshape(self.pixels_dim, self.pixels_dim)
                )

                coeff_2FT = fftshift(coeff_2FT)
                coeff_2FT = abs(coeff_2FT)

                Ground_True = np.array(coeff_2FT)

                coeff_measure_slice = []
                for values in pixeldata_slice:
                    measure_slice = np.mean(values)
                    coeff_measure_slice.append(measure_slice)
                coeff_measure_slice = np.array(coeff_measure_slice)
                coeff_2FT_slice = fft2(
                    coeff_measure_slice.reshape(self.pixels_dim, self.pixels_dim)
                )

                coeff_2FT_slice = fftshift(coeff_2FT_slice)
                coeff_2FT_slice = abs(coeff_2FT_slice)

                Predicted = np.array(coeff_2FT_slice)

                DICE_value = original_plotting.dice_coefficient(
                    A=Predicted, B=Ground_True
                )
                dice_values.append(DICE_value)

        dice_values = np.array(dice_values)
        dice_values = dice_values.reshape(2, len(dice_values) // 2)

        plt.plot(
            range(1, 1 + len(dice_values[0])),
            dice_values[0],
            marker="o",
            color="blue",
            label="real",
        )
        plt.plot(
            range(1, 1 + len(dice_values[1])),
            dice_values[1],
            marker="o",
            color="green",
            label="complex",
        )
        plt.title(
            f"DICE Coefficients for Data Sliced up to {slicefactor*1/sampling_rate} seconds"
        )
        plt.xlabel("Sample Index")
        plt.ylabel("DICE Value")
        plt.grid(True)
        plt.legend()

        file_path = os.path.join(
            self.saving_path,
            f"DICE_slicedata_backslice{slicefactor}.pdf",
        )
        if os.path.exists(file_path):
            os.remove(file_path)
        plt.tight_layout()

        plt.savefig(file_path)
        print(f"Plotted coefficient DICE.")
        plt.clf()

    def calculate_mean_from_preprocessed(self):
        measure_data = []
        for data_types in tqdm(self.preprocessed_data):
            data_type = []
            for coeff_data in data_types:
                coeff_measure = []
                for value in coeff_data:
                    coeff_measure.append(np.mean(value))  # Simplified mean calculation
                data_type.append(coeff_measure)
            measure_data.append(data_type)

        measure_data = np.array(measure_data)
        return measure_data

    def plotting_spectrums(self):
        ticks = (1, 10, 20, 30, 40, 50)
        num_coefficients = 31  # Total number of coefficients to plot

        if not os.path.exists(self.saving_path):
            os.makedirs(self.saving_path)

        rows, cols = 6, 6

        for count, data_type in enumerate(self.measure_data[1:, :, :]):

            if count == 0:
                type = "real"
            elif count == 1:
                type = "complex"

            r_data = [
                np.reshape(sublist, (self.pixels_dim, self.pixels_dim))
                for sublist in data_type
            ]
            r_data = np.array(r_data)
            r_data = r_data[:, 3:, :]  # slicing

            fig, axs = plt.subplots(
                rows,
                cols,
                figsize=(cols * 3, rows * 3),
                sharex=True,
                sharey=True,
            )
            axs = axs.flatten()

            for i, values in tqdm(enumerate(r_data)):
                values = np.array(values)
                values = values[::-1, :]

                # RMS boundries
                # f_values = values.flatten()

                # min_parameter = np.sqrt(
                #     sum(x**2 for x in f_values) / len(f_values)
                # ) - 3 * np.std(f_values)
                # max_parameter = np.sqrt(
                #     sum(x**2 for x in f_values) / len(f_values)
                # ) + 3 * np.std(f_values)

                min_parameter = np.mean(values) - 3 * np.std(values)  # mean or median
                max_parameter = np.mean(values) + 3 * np.std(values)

                subplot_index = i

                if subplot_index < len(axs):
                    ax = axs[subplot_index]
                    im = ax.imshow(
                        values,
                        vmin=min_parameter,
                        vmax=max_parameter,
                        cmap="gray",
                        interpolation="nearest",
                        origin="lower",
                    )
                    ax.set_ylabel("Pixels")
                    ax.set_yticks(ticks)
                    ax.set_yticklabels(ticks)
                    ax.set_xlabel("Pixels")
                    ax.set_xticks(ticks)
                    ax.set_xticklabels(ticks)
                    ax.set_title(f"Coefficient {i+1}")
                    ax.set_xlim(1, self.pixels_dim)
                    ax.set_ylim(1, self.pixels_dim)
                    ax.grid(visible=False)

            file_path = os.path.join(
                self.saving_path,
                f"all_coefficients_FT2D_values_harmonics_{type}_backslice{slicefactor}.pdf",
            )
            if os.path.exists(file_path):
                os.remove(file_path)
            plt.tight_layout()
            plt.savefig(file_path)
            print(f"Plotted all {type} coefficients of original data.")
            plt.clf()


class Dataset_analysis:
    def __init__(
        self,
        file_name,
        no_coeff,
        pixels,
        pixel_dim,
        saving_path,
        allpixel=False,
        plotting_type: str = None,
    ):
        self.file_name = file_name
        self.no_coeff = no_coeff
        self.pixel = pixels
        self.row_no, self.col_no = self.pixel
        self.pixels_dim = pixel_dim
        self.pixels = self.pixel_check(pixelcheck=allpixel)
        self.data = self.import_h5py()
        self.saving_path = saving_path

        valid_plotting_types = ["peak_extraction", "peak_diff", "single_FT_plotting"]

        # Check if the provided plotting_type is valid
        if plotting_type is not None and plotting_type not in valid_plotting_types:
            raise ValueError(
                f"Invalid plotting_type '{plotting_type}'. Choose from {valid_plotting_types}"
            )

        self.plotting = plotting_type

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

                        if i % 4 == 0:
                            plt.figure(figsize=(12, 8))

                        plt.subplot(2, 2, i % 4 + 1)

                        x = np.linspace(start=0, stop=len(coeff), num=len(coeff))
                        plt.plot(x * (0.1 / 160), coeff)
                        plt.xlabel("Time (seconds)")
                        plt.ylabel("Values")
                        plt.title(
                            f"Coef {i+1}, Pixel: row {self.row_no[idx]}, col {self.col_no[idx]}"
                        )

                        if (i + 1) % 4 == 0:
                            plt.tight_layout()
                            file_path = os.path.join(
                                self.saving_path,
                                f"coefficient_{i-2}&{i-1}&{i}&{i+1}_{type}_pixel{self.row_no[idx]}{self.col_no[idx]}.pdf",
                            )
                            if os.path.exists(file_path):
                                os.remove(file_path)

                            plt.savefig(file_path)
                            plt.clf()
                            exit()

                        if i == 30:
                            plt.tight_layout()
                            file_path = os.path.join(
                                self.saving_path,
                                f"coefficient_{i-2}&{i-1}&{i}&{i+1}_{type}_pixel{self.row_no[idx]}{self.col_no[idx]}.pdf",
                            )
                            if os.path.exists(file_path):
                                os.remove(file_path)

                            plt.savefig(file_path)
                            plt.clf()

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
        major_freq = 220
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

    def fouriertransform_plot(self, data, index, index_pixel, _type):
        if index != 0:
            n = len(data)
            Ts = 1 / 1600
            total_t = Ts * n
            t = np.arange(0, total_t, Ts)

            dataff = fft(data)
            orgpeak, _ = find_peaks(data, distance=3, prominence=0.02)
            meanpeak = np.mean(np.diff(t[orgpeak]))
            stdpeak = np.std(np.diff(t[orgpeak]))
            diff = np.diff(t[orgpeak])

            dataff[0] = 0
            dataff_n = np.array(
                [
                    (value - min(dataff)) / (max(dataff) - min(dataff))
                    for value in dataff
                ]
            )

            fs = 1 / Ts

            ft = fftfreq(n, Ts)
            testft = ft

            # Slicing to resort list
            slice_half = len(testft) // 2

            plot_dataf = np.array([*dataff_n[slice_half:], *dataff_n[:slice_half]])
            testft = np.array([*testft[slice_half:], *testft[:slice_half]])

            height_threshold = (abs(np.mean(dataff_n) + np.std(dataff_n)), 1)

            # Major frequncies search
            plot_peak, _ = find_peaks(abs(plot_dataf), height=height_threshold)
            idx_peaks = testft[plot_peak]

            peak, _ = find_peaks(
                abs(dataff_n),
                height=height_threshold,
            )

            plot_new_dataff = np.zeros_like(plot_dataf)
            plot_new_dataff[plot_peak] = plot_dataf[plot_peak]
            freqs = ft[peak]
            freqf = [i for i in freqs if i >= 0]

            new_dataff = np.zeros_like(dataff)
            new_dataff[peak] = dataff[peak]

            restored_time_data = ifft(new_dataff)
            restored_time_data = np.real(restored_time_data)
            slice_half = len(restored_time_data) // 2

            # restored_time_data = np.array(
            #     [*restored_time_data[slice_half:], *restored_time_data[:slice_half]]
            # )
            if self.plotting == "single_FT_plotting":
                fig, axs = plt.subplots(2, 2, figsize=(12, 8))

                # First subplot: Original time-domain data
                axs[0, 0].plot(t, data + (0 - np.mean(data)))
                axs[0, 0].set_xlabel("Time (seconds)")
                axs[0, 0].set_ylabel("Amplitude")
                axs[0, 0].set_title("Original Time-Domain Signal")

                for freq in orgpeak:
                    axs[0, 0].axvline(
                        x=t[freq],
                        color="C2",
                        linestyle="--",
                        label=f"Freq: {freq:.2f} Hz",
                    )

                # Second subplot: Magnitude spectrum
                # axs[0, 1].plot(filtered_ft, filtered_dataf)
                axs[0, 1].plot(testft, abs(plot_dataf))
                axs[0, 1].axhline(y=height_threshold[0], color="red", linestyle="--")
                axs[0, 1].set_xlabel("Frequency (Hz)")
                axs[0, 1].set_ylabel("Magnitude (Normalized)")
                axs[0, 1].set_title("Magnitude Spectrum")

                for freq in idx_peaks:
                    axs[0, 1].axvline(
                        x=freq, color="C1", linestyle="--", label=f"Freq: {freq:.2f} Hz"
                    )

                # Third subplot: Major frequencies
                axs[1, 0].plot(testft, abs(plot_new_dataff))
                axs[1, 0].set_xlabel("Frequency (Hz)")
                axs[1, 0].set_ylabel("Magnitude (Normalized)")
                axs[1, 0].set_title("Major Frequencies")

                # Fourth subplot: Restored time-domain data
                axs[0, 0].plot(t, restored_time_data)
                axs[1, 1].plot(t, restored_time_data)
                axs[1, 1].set_xlabel("Time (seconds)")
                axs[1, 1].set_ylabel("Amplitude")
                axs[1, 1].set_title("Restored Time-Domain Signal")

                fig.suptitle(f"FT Analysis for Correlation between 1 and {index + 1}")

                file_path = os.path.join(
                    self.saving_path,
                    f"Correlation_1&{index + 1}_{self.row_no[index_pixel]}{self.col_no[index_pixel]}_{_type}FTA_{len(idx_peaks)}.pdf",
                )
                if os.path.exists(file_path):
                    os.remove(file_path)

                plt.tight_layout()
                plt.savefig(file_path)
                plt.clf()

            return meanpeak, stdpeak, diff, freqf

    def fouriertransform(self, data, index, ran_type=None):

        if index != 0:
            n = len(data)
            Ts = 1 / 1600
            total_t = Ts * n
            t = np.arange(0, total_t, Ts)

            # Original time-domain data
            dataff = fft(data)
            dataff[0] = 0
            dataff_n = np.array(
                [
                    (value - min(dataff)) / (max(dataff) - min(dataff))
                    for value in dataff
                ]
            )

            fs = 1 / Ts

            ft = fftfreq(n, Ts)

            if ran_type == None:
                threshold = abs(np.mean(dataff_n) + np.std(dataff_n))
            elif ran_type <= 1:
                threshold = abs(np.mean(dataff_n) + ran_type * np.std(dataff_n))

            # Major frequncies search
            peak, _ = find_peaks(
                abs(dataff_n),
                height=threshold,
            )

            new_dataff = np.zeros_like(dataff)
            new_dataff[peak] = dataff[peak]
            freqs = ft[peak]
            freqs = [i for i in freqs if i > 0]

            restored_time_data = ifft(new_dataff)
            restored_time_data = np.real(restored_time_data)
            slice_half = len(restored_time_data) // 2

            return restored_time_data, freqs

    def extract_peak_and_neighbors(self, coefficient, peaks, radius=1):

        peak_data = []
        length = len(coefficient)

        for i in peaks:
            start_index = max(0, i - radius)
            end_index = min(length, i + radius + 1)
            peak_data.append(coefficient[start_index:end_index])

        return peak_data

    def plottingpeak(self, meanpeak, stdpeak, diffpeaks, freqf, index, plot_dim):

        fig, axs = plot_dim[0], plot_dim[1]

        x0 = np.arange(0, len(diffpeaks), 1)
        axs[0].plot(x0, diffpeaks, label=f"Coefficient {index}")
        axs[0].set_xlabel("Number of peaks")
        axs[0].set_ylabel("Difference between peaks")
        axs[0].set_title("Difference between peak of correlated peaks")
        axs[0].legend()

        timesf = [1 / i for i in freqf]
        index_list = [index for i in range(len(timesf))]

        axs[1].scatter(
            index_list,
            timesf,
            label=f"Estimated difference: Coeff {index}",
        )
        axs[1].errorbar(
            index,
            meanpeak,
            yerr=stdpeak,
            fmt="o",
            ecolor="black",
            capsize=3,
            alpha=0.7,
        )
        axs[1].set_xlabel("Coefficient No")
        axs[1].set_ylabel("Values")
        axs[1].set_title(f"Mean difference: Coeff {index}")
        axs[1].legend()

        if index == 6:
            fig.suptitle(
                f"Check of FT Anaylsis and peridically coefficients between 1 and {index + 1}"
            )

            file_path = os.path.join(
                self.saving_path,
                f"Peak_difference_{self.row_no[index]}{self.col_no[index]}_for_coefficients_1-{index + 1}.pdf",
            )
            if os.path.exists(file_path):
                os.remove(file_path)

            plt.tight_layout()
            plt.savefig(file_path)
            exit()

    def cross_correlation(self):
        radius = 1  # Hyperparameter for number of indices taken either side of slice
        hdf5_file_path = os.path.join(
            self.saving_path,
            f"Sliced_data_FTpeaks_sliceN{radius}_meanvalues.hdf5",
        )
        with h5py.File(hdf5_file_path, "w") as hdf5_file:
            row_number = 0
            col_number = 0
            for index_pixel, pixel in enumerate(self.data):

                if index_pixel % self.pixels_dim == 0 and index_pixel != 0:
                    print(f"Saved row {row_number} fully, from column 0-{col_number}")
                    row_number += 1
                    col_number = 0

                for index_type, datatype in enumerate(pixel):

                    if index_type == 0:
                        _type = "Real"
                    if index_type == 1:
                        _type = "Complex"

                    COEFFICIENT_1 = datatype[0]
                    count_max = len(datatype[1]) + 1
                    counts = [i for i in range(1, count_max)]
                    PLOT_INDEX = 1

                    if self.plotting == "peak_extraction":
                        _, axs = plt.subplots(2, 3, figsize=(15, 10))

                    if self.plotting == "peak_diff":
                        fig, axs = plt.subplots(1, 2, figsize=(12, 8))

                    final_data = []
                    for index, coefficient in enumerate(datatype):

                        correlation_values = correlate(
                            COEFFICIENT_1, coefficient, mode="same"
                        )

                        correlation_values = correlation_values - COEFFICIENT_1
                        if index != 0:

                            # plotting FT plot
                            if self.plotting == "single_FT_plotting" or "peak_diff":

                                meanpeak, stdpeak, diffpeaks, freqf = (
                                    self.fouriertransform_plot(
                                        correlation_values,
                                        index,
                                        index_pixel=index_pixel,
                                        _type=_type,
                                    )
                                )

                                if index == 2 and self.plotting == "single_FT_plotting":
                                    break

                                if self.plotting == "peak_diff":
                                    self.plottingpeak(
                                        meanpeak,
                                        stdpeak,
                                        diffpeaks,
                                        freqf,
                                        index,
                                        plot_dim=(fig, axs),
                                    )

                            FT_values, FT_fre = self.fouriertransform(
                                correlation_values, index
                            )
                            if len(FT_fre) == 0:
                                print(
                                    "FT analysis is not working, rerunning the analysis will lower threshold"
                                )
                                for i in np.arange(0.9, 0, -0.1):

                                    FT_values, FT_fre = self.fouriertransform(
                                        correlation_values, index, ran_type=i
                                    )

                            cap = (
                                max(FT_fre)
                                - min(FT_fre)
                                + 0.341 * (min(FT_fre) + max(FT_fre))
                            )
                            max_gap = round((1 / cap) / (0.1 / 160))

                            wave_peaks, _ = find_peaks(FT_values, distance=max_gap)

                            if self.plotting == "peak_extraction":
                                axs[(PLOT_INDEX - 1) // 3, (PLOT_INDEX - 1) % 3].plot(
                                    counts,
                                    FT_values,
                                    label=f"FT function",
                                )
                                axs[(PLOT_INDEX - 1) // 3, (PLOT_INDEX - 1) % 3].plot(
                                    counts,
                                    correlation_values
                                    + (0 - np.mean(correlation_values)),
                                    label=f"Correlation values",
                                )

                                for i in wave_peaks:
                                    axs[
                                        (PLOT_INDEX - 1) // 3, (PLOT_INDEX - 1) % 3
                                    ].axvline(
                                        x=i,
                                        color="r",
                                        linestyle="--",
                                        label="Peak_mine" if i == wave_peaks[0] else "",
                                    )

                                axs[
                                    (PLOT_INDEX - 1) // 3, (PLOT_INDEX - 1) % 3
                                ].set_xlabel("Index")
                                axs[
                                    (PLOT_INDEX - 1) // 3, (PLOT_INDEX - 1) % 3
                                ].set_ylabel(
                                    f"Correlation with Coefficient {index + 1} from FT"
                                )
                                PLOT_INDEX += 1

                                if index % 6 == 0:
                                    plt.suptitle(
                                        f"Correlation between coefficient 1 and {index-4}-{index+1}"
                                    )
                                    file_path = os.path.join(
                                        self.saving_path,
                                        f"Peak_extraction_{index+1}_{self.row_no[index_pixel]}{self.col_no[index_pixel]}_{_type}_comparsion.pdf",
                                    )
                                    if os.path.exists(file_path):
                                        os.remove(file_path)

                                    plt.legend(
                                        bbox_to_anchor=(1.02, 1), loc="upper left"
                                    )
                                    plt.tight_layout()
                                    plt.savefig(file_path)
                                    PLOT_INDEX = 1
                                    exit()
                                    _, axs = plt.subplots(2, 3, figsize=(15, 10))

                            datap = []
                            if len(wave_peaks) > 0:
                                datap = self.extract_peak_and_neighbors(
                                    coefficient, wave_peaks, radius
                                )
                            datap = [item for sublist in datap for item in sublist]

                            final_data.append(np.mean(datap))

                    hdf5_file.create_dataset(
                        f"/mean/Pixel_{row_number}_{col_number}/{_type}",
                        data=final_data,
                    )

                col_number += 1


def slice_plotting_real(
    filepath, coefficient_indices, pixel_shape=(50, 50), num_columns=5, plotting=False
):
    ticks = (1, 10, 20, 30, 40, 50)
    types = ["Real", "Complex"]
    savingpath = r"C:\Users\User\Documents\2024\project\test_figures\Pixel_all"

    num_coefficients = len(coefficient_indices)
    num_rows = (num_coefficients + num_columns - 1) // num_columns

    if plotting:
        fig, axs = plt.subplots(
            num_rows,
            num_columns,
            figsize=(num_columns * 3, num_rows * 3),
            sharex=True,
            sharey=True,
        )
        axs = axs.flatten()

    with h5py.File(filepath, "r") as h5:
        for _type in types:
            for i, coeff_index in enumerate(coefficient_indices):
                mean_values = []

                for row in range(pixel_shape[0]):
                    row_means = []

                    for col in range(pixel_shape[1]):
                        dataset_name = f"/mean/Pixel_{row}_{col}/{_type}"
                        dataset = h5[dataset_name]

                        if coeff_index < dataset.shape[0]:
                            mean_value = dataset[coeff_index]

                        row_means.append(mean_value)

                    mean_values.append(row_means)

                mean_values = np.array(mean_values)

                if plotting:
                    min_parameter = np.mean(mean_values) - 3 * np.std(mean_values)
                    max_parameter = np.mean(mean_values) + 3 * np.std(mean_values)

                    mean_values = mean_values[
                        ::-1, :
                    ]  # inverted to be inline with original

                    ax = axs[i]
                    im = ax.imshow(
                        mean_values,
                        vmin=min_parameter,
                        vmax=max_parameter,
                        cmap="gray",
                        interpolation="nearest",
                        origin="lower",
                    )
                    ax.set_title(f"Coefficient {coeff_index}")
                    ax.set_xlabel("Pixels")
                    ax.set_ylabel("Pixels")
                    ax.set_xticks(ticks)
                    ax.set_xticklabels(ticks)
                    ax.set_xlim(1, pixel_shape[1])
                    ax.set_yticks(ticks)
                    ax.set_yticklabels(ticks)
                    ax.set_ylim(1, pixel_shape[0])
                    ax.grid(visible=False)

                    print(f"Plotted {_type} coefficient {coeff_index}")

            file_path = os.path.join(
                savingpath,
                f"Sliced_data_full_image_meanvalues_{_type}coefficients{min(coefficient_indices)}-{max(coefficient_indices)}.pdf",
            )
            if os.path.exists(file_path):
                os.remove(file_path)

            plt.tight_layout()
            plt.savefig(file_path)

        plt.clf()


if __name__ == "__main__":

    file = "./Data/Measurement_of_2021-06-18_1825.h5"
    path = r"C:\Users\User\Documents\2024\project\test_figures\PP_figures\2DFT"
    sampling_rate = 160
    PIXEL_DIM = 50
    NO_COEFF = 31
    slicefactor = 20

    PIXEL = [[2], [1]]
    PIXEL2 = [[3], [4]]
    PIXELR = [np.random.randint(2, 49, size=13).tolist() for _ in range(2)]
    idx = np.arange(PIXEL_DIM)
    ALLPIXEL = [idx.tolist(), idx.tolist()]

    folder_name = f"Pixel_{PIXEL[0][0]},{PIXEL[1][0]}"
    folder_name = "Pixel_all"
    directory_name = os.path.join(path, folder_name)
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    original = original_plotting(
        file_name=file,
        no_coeff=NO_COEFF,
        saving_path=path,
        pixel_dim=PIXEL_DIM,
        preprocessed_data="2FT_DICE",
    )
    # original.plotting_spectrums()

    # dataset = Dataset_analysis(
    #     file_name=file,
    #     pixels=PIXEL,
    #     no_coeff=NO_COEFF,
    #     pixel_dim=PIXEL_DIM,
    #     saving_path=directory_name,
    #     allpixel=True,
    #     plotting_type="peak_extraction",
    # )

    # slice_data = dataset.cross_correlation()
    # slice_data = dataset.time_series_plot()

    # filename = r"C:\Users\User\Documents\2024\project\test_figures\Pixel_all\Sliced_data_FTpeaks_sliceN1_meanvalues.hdf5"

    # coefficient_indices = list(
    #     range(2, NO_COEFF + 1)
    # )  # List the coefficients you want to plot
    # coefficient_indices = np.sort(coefficient_indices)
    # slice_plotting_real(filename, coefficient_indices, plotting=True)
