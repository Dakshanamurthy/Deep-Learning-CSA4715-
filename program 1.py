import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
actual_labels = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1])
predicted_labels = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 1])
conf_matrix = confusion_matrix(actual_labels, predicted_labels)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Dog', 'Dog'], yticklabels=['Not Dog', 'Dog'])
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.title('Confusion Matrix - Dog or Not Dog')
plt.show()
