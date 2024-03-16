import numpy as np

# data = [
#     76,
#     34,
#     12,
#     89,
#     45,
#     67,
#     23,
#     91,
#     18,
#     56,
#     77,
#     99,
#     42,
#     65,
#     10,
#     83,
#     29,
#     54,
#     37,
#     81,
#     14,
#     70,
#     95,
#     48,
#     62,
#     20,
#     73,
#     39,
#     85,
#     50,
#     16,
#     78,
#     32,
#     60,
#     97,
#     26,
#     69,
#     52,
#     30,
#     88,
#     7,
#     44,
#     68,
#     25,
#     98,
#     41,
#     15,
#     87,
#     64,
#     21,
# ]
# data = [
#     [76, 34, 12, 89, 45, 67, 23, 91, 18, 56],
#     [77, 99, 42, 65, 10, 83, 29, 54, 37, 81],
#     [14, 70, 95, 48, 62, 20, 73, 39, 85, 50],
#     [16, 78, 32, 60, 97, 26, 69, 52, 30, 88],
#     [7, 44, 68, 25, 98, 41, 15, 87, 64, 21],
# ]
# grouped_values = [data[i : i + 10] for i in range(0, len(data), 10)]

# # print(np.array(grouped_values).shape)
# # print(grouped_values)
# binned_data = []
# for coeff_data in data:

#     # add in check for multi pixels
#     bin_edges = np.linspace(min(coeff_data), max(coeff_data), 1000)
#     bins = np.digitize(coeff_data, bin_edges)
#     binned_data.append(bins)

# pixels_co = [[1, 6], [2]]
# row_no, col_no = pixels_co
# print(row_no, col_no)

# print(len(pixels_co[0]))


# labels = [0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0]

# label_counts = {}
# for label in labels:
#     label_counts[label] = label_counts.get(label, 0) + 1

# counts = [label_counts.get(label, 0) for label in sorted(label_counts)]

# print(counts)
# print(label_counts)

