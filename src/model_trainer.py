import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                             f1_score, roc_auc_score, roc_curve, mean_squared_error, r2_score,
                             precision_score, recall_score)
from xgboost import XGBClassifier, XGBRegressor

class ModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state

    def split_data(self, X, y, test_size=0.2):
        """Splits data into train and test sets."""
        return train_test_split(X, y, test_size=test_size, random_state=self.random_state)

    def train_rf_classifier(self, X_train, y_train, **kwargs):
        """Trains a Random Forest Classifier."""
        model = RandomForestClassifier(random_state=self.random_state, **kwargs)
        model.fit(X_train, y_train)
        return model

    def train_xgb_classifier(self, X_train, y_train, **kwargs):
        """Trains an XGBoost Classifier."""
        model = XGBClassifier(random_state=self.random_state, **kwargs)
        model.fit(X_train, y_train)
        return model

    def evaluate_classifier(self, model, X_test, y_test, threshold=0.5):
        """Evaluates a classification model."""
        y_pred_default = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Use optimal threshold if requested (via y_pred update)
        y_pred = (y_proba > threshold).astype(int)
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred, average='weighted'),
            "auc": roc_auc_score(y_test, y_proba),
            "report": classification_report(y_test, y_pred)
        }
        
        return metrics, y_proba

    def plot_roc_curve(self, y_test, y_proba):
        """Plots the ROC Curve."""
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="red", label=f"ROC (AUC = {roc_auc_score(y_test, y_proba):.2f})")
        plt.plot([0, 1], [0, 1], color="green", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic Curve")
        plt.legend()
        plt.grid(True)
        return plt

    def plot_confusion_matrix(self, y_test, y_proba, threshold=0.5):
        """Plots the Confusion Matrix."""
        y_pred = (y_proba > threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
                    xticklabels=['Positive (1)', 'Negative (0)'],
                    yticklabels=['Actual (1)', 'Actual (0)'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix (Threshold={threshold})')
        return plt

    def train_rf_regressor(self, X_train, y_train, **kwargs):
        """Trains a Random Forest Regressor."""
        model = RandomForestRegressor(random_state=self.random_state, **kwargs)
        model.fit(X_train, y_train)
        return model

    def train_xgb_regressor(self, X_train, y_train, **kwargs):
        """Trains an XGBoost Regressor."""
        model = XGBRegressor(random_state=self.random_state, **kwargs)
        model.fit(X_train, y_train)
        return model

    def evaluate_regressor(self, model, X_test, y_test):
        """Evaluates a regression model."""
        y_pred = model.predict(X_test)
        
        metrics = {
            "r2_score": r2_score(y_test, y_pred),
            "mse": mean_squared_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred))
        }
        
        return metrics, y_pred

    def plot_feature_importance(self, model, feature_names, top_n=10):
        """Plots top N feature importances."""
        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_n:]
        
        plt.figure(figsize=(10, 6))
        plt.title(f"Top {top_n} Feature Importances")
        plt.barh(range(len(indices)), importances[indices], align="center")
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel("Relative Importance")
        return plt

    def iterative_evaluation(self, X, y, model_type='xgb', n_iterations=10, test_size=0.2):
        """Performs iterative training and evaluation to get distribution of metrics."""
        results = []
        for i in range(n_iterations):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=self.random_state + i
            )
            
            if model_type == 'xgb':
                model = self.train_xgb_classifier(X_train, y_train)
            else:
                model = self.train_rf_classifier(X_train, y_train)
                
            metrics, _ = self.evaluate_classifier(model, X_test, y_test)
            results.append(metrics)
            
        return pd.DataFrame(results)
