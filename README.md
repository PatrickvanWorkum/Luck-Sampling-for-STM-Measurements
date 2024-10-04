## Lucky Sampling in STM measurement

Contains the code used in a project working with Prof. Dr. Fabian Nattere, focused around applying lucky sampling to STM measurement data.

The major takeaways:

- Using color mapping based on the mean ± 3 standard deviations is the most effective approach.
- The median and adjustments to the standard deviation have no noticeable impact on the results. 
- The Root Mean Squared (RMS) provides a different perspective compared to the mean or median
- Important information is not encoded in periodic points but rather across a given time spread, however further investigation into this is needed.
- It may be possible to reduce the time spent on each pixel by up to 75%, depending on the variation in the time spread. One would want to conmplete analysis on another propers running for longer and shorter times.
- Work to better numerically compare the results; one, would want to look at matching 2D FT features across the spectrums to see. The removal of background noise may help with this

Data take by Dr Berk Zengin.

Sliced spectrums, at intervals compared to the full spectrum:
![image](https://github.com/user-attachments/assets/e6dff35e-8ff5-4da9-a70b-ac1db251a12f)

2 dimensional Fourier Transformed sliced spectrum, compared to the 2 dimensional Fourier Transformed full spectrum
![image](https://github.com/user-attachments/assets/deb0320f-e8af-423e-acab-b2377394a968)


![image](https://github.com/user-attachments/assets/d186dee2-d16b-4a06-88ee-82024602642d)

![image](https://github.com/user-attachments/assets/55e8ed54-e88c-4e44-850d-accb8811a91a)


