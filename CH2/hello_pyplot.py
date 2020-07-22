import matplotlib.pyplot as plt
import numpy

import CH2.kNN as kNN

dating_data_mat, dating_labels = kNN.file2matrix('../source-by-author/Ch02/datingTestSet2.txt')
norm_mat, ranges, min_vals = kNN.auto_norm(dating_data_mat)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Percentage of time spent playing video games')
ax.set_ylabel('Liters of ice cream consumed weekly')
ax.scatter(norm_mat[:, 0], norm_mat[:, 1],
           15.0 * numpy.array(dating_labels), 15.0 * numpy.array(dating_labels))
plt.show()
