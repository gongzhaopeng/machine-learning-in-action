import matplotlib.pyplot as plt
import numpy

import CH2.kNN as kNN

dating_data_mat, dating_labels = kNN.file2matrix('source-by-author/Ch02/datingTestSet2.txt')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Percentage of time spent playing video games')
ax.set_ylabel('Liters of ice cream consumed weekly')
ax.scatter(dating_data_mat[:, 0], dating_data_mat[:, 1],
           15.0 * numpy.array(dating_labels), 15.0 * numpy.array(dating_labels))
plt.show()
