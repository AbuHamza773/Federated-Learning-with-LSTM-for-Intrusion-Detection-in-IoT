
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on the test set."""
    y_pred = model.predict(X_test)
    y_pred_classes = (y_pred > 0.5).astype(int) if y_test.ndim == 1 else y_pred.argmax(axis=1)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred_classes),
        'precision': precision_score(y_test, y_pred_classes, average='weighted'),
        'recall': recall_score(y_test, y_pred_classes, average='weighted'),
        'f1_score': f1_score(y_test, y_pred_classes, average='weighted'),
        'confusion_matrix': confusion_matrix(y_test, y_pred_classes)
    }
    return metrics
