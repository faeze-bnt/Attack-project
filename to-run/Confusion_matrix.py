
from sklearn.metrics import confusion_matrix
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # For pretty confusion matrix


def one_hot_to_binary(one_hot_array):
  binary_data = []
  for i in range(one_hot_array.shape[0]):
    for j in range(0, 10):
        if (j == np.argmax(one_hot_array[i])):
            binary_data.append(j)
  binary_array = np.array(binary_data)
  return binary_array

class Conf_matrix:
  def __init__(self, trues, predictions, plt_file):
    true_label = one_hot_to_binary(trues)
    pred_label = one_hot_to_binary(predictions)

    c_matrix = confusion_matrix(true_label, pred_label)
    # print(c_matrix)

    # drae a plot to visually see the results.
    fig, ax = plt.subplots()
    sns.heatmap(c_matrix, annot=False, ax=ax, cmap='Blues')  # Annotate the cells with the numeric values


    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    plt.savefig(plt_file)



