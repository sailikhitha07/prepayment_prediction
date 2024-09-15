#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class CustomPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, scaler, clf, reg):
        self.scaler = scaler
        self.clf = clf
        self.reg = reg

    def fit(self, X, y_class, y_reg):
        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y_class)
        X_filtered = X_scaled[y_class == 1]
        y_reg_filtered = y_reg[y_class == 1]
        self.reg.fit(X_filtered, y_reg_filtered)
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        y_class_pred = self.clf.predict(X_scaled)
        y_reg_pred = np.full_like(y_class_pred, np.nan, dtype=float)
        if any(y_class_pred == 0):
            X_filtered = X_scaled[y_class_pred == 0]
            y_reg_pred[y_class_pred == 0] = self.reg.predict(X_filtered)
        return y_class_pred, y_reg_pred


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'djangoMLdeployment.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
