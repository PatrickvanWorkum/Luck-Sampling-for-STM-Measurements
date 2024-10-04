# Lucky Sampling in STM Measurement

This repository contains the code used in a project in collaboration with **Prof. Dr. Fabian Natterer**, focused on applying lucky sampling to STM (Scanning Tunneling Microscopy) measurement data.

### Major Takeaways

- **Using color mapping** based on the mean ± 3 standard deviations is the most effective approach.
- Adjusting the **median** and **standard deviation** has no noticeable impact on the results.
- **Root Mean Squared (RMS)** provides a different perspective compared to the mean or median.
- **Important information** is not encoded in periodic points but is spread over a given time frame. Further investigation is needed to confirm this.
- **Reducing the time spent** on each pixel by up to 50% might be possible depending on the variation in the time spread. Further analysis is required for longer and shorter time experiments.
- To **better compare results numerically**, it would be useful to look at matching **2D Fourier Transform (FT)** features across the spectrums. Removing background noise might assist in this analysis.

---

### Data Information

- **Data collected by**: Dr. Berk Zengin
- **Date of experiment**: 18/06/2021
- **Dataset**: Contains 24.8 million points or 155,000 after averaging.
- **Experiment details**:
  - 50x50 pixels with 31 real and complex coefficients
  - Each pixel has 160 data points captured over 0.1 seconds
- **Plots**: Represent the mean value of pixels for each coefficient, with color mapping bound by ±3 standard deviations.

![image](https://github.com/user-attachments/assets/fa9d6103-4870-4b91-aa3d-18399363868f)

*Coefficients plotted are from 1 to 12 real components.*

---

### Visuals

#### Sliced Spectrums
- Sliced spectrums, at intervals compared to the full spectrum:

![Sliced Spectrum](https://github.com/user-attachments/assets/e6dff35e-8ff5-4da9-a70b-ac1db251a12f)

#### 2D Fourier Transformed Spectrum
- 2D Fourier Transformed sliced spectrum, compared to the full 2D Fourier Transformed spectrum:

![2D FT Spectrum](https://github.com/user-attachments/assets/deb0320f-e8af-423e-acab-b2377394a968)

#### Dice Coefficients (Sliced Up to the Middle Point)
- Dice coefficients across all coefficients when data is sliced up to the middle (80 data points):

![Dice Coefficients (First Half)](https://github.com/user-attachments/assets/db9d1a06-d31a-4e2a-97aa-6f2f55442352)

#### Dice Coefficients (Sliced From the Middle Point)
- Dice coefficients across all coefficients when data is sliced from the middle point (80 data points):

![Dice Coefficients (Second Half)](https://github.com/user-attachments/assets/66970e58-5596-4064-901a-c62fc0ecbe8c)

---

### Future Work

- Conduct further experiments to analyze the variation in the time spread.
- Investigate matching **2D FT features** across spectrums.
- Explore ways to remove background noise to improve the clarity of results.

---

### Raw Fourier Signal Analysis

While **Fourier analysis on the raw spectrum** was initially explored, it turned out to be a dead end. However, I encourage the reader to take a look at the code for further insights.

![image](https://github.com/user-attachments/assets/fdc812f3-18aa-46c6-93be-74b83c5f9329)

---

### General Steps

The following are the general steps used for signal analysis:

1. **Correlate Coefficient 1** with other coefficients and subtract off Coefficient 1.
2. **Apply a Fourier Transform (FT)** on the data to extract the major frequencies.
3. **Create a new signal** in the frequency domain using these frequencies.
4. **Reverse** the newly created signal back into the time domain.
5. **Slice the original data** based on this signal and plot the result.

---

![image](https://github.com/user-attachments/assets/ae8f3252-cf50-4a84-99d9-f23209bac563)

*FT Coefficients plotted are from 1 to 12 real components.*

Feel free to explore the code and experiment with different approaches based on these steps!

