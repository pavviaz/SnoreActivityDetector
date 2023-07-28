from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np


class ConfusionMatrix:
    def __init__(self, num_classes, **kwargs):
        self.cm = np.zeros(shape=(num_classes, num_classes))
    
    def update(self, y_true, y_pred):
        self.cm += confusion_matrix(y_true, y_pred)
        return self.cm
    
    def plot(self):
        return ConfusionMatrixDisplay(self.cm)