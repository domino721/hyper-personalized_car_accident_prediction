import os
import pandas as pd
import matplotlib.pyplot as plt
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer

def main():
    """
    Main entry point for the car accident prediction pipeline.
    This script demonstrates data loading, cleaning, preprocessing, 
    and model training using modular components.
    """
    # 1. Initialization
    processor = DataProcessor()
    trainer = ModelTrainer(random_state=42)
    
    # Define dataset paths
    dataset_dir = "Dataset"
    file_names = [
        "(자동차보험) 고객별 사고 발생률 예측 모델링_1.csv",
        "(자동차보험) 고객별 사고 발생률 예측 모델링_2.csv",
        "(자동차보험) 고객별 사고 발생률 예측 모델링_3.csv"
    ]
    file_paths = [os.path.join(dataset_dir, name) for name in file_names]
    
    print("Step 1: Loading and merging data...")
    try:
        raw_df = processor.load_and_merge_data(file_paths)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Data Cleaning and Feature Engineering
    print("Step 2: Cleaning and engineering features...")
    df_cleaned = processor.clean_data(raw_df)
    df_engineered = processor.feature_engineering(df_cleaned)
    
    # 3. Preprocessing (Manual Label Encoding)
    print("Step 3: Performing label encoding...")
    # Drop columns not needed for modeling (as seen in notebook)
    df_for_model = df_engineered.drop(columns=['사고율', '사고건수', '유효대수'])
    # In the notebook, they also drop '운전자한정특별약관' before some models or handle it separately.
    # We'll drop it for this demonstration if it's not encoded yet.
    df_for_model = df_for_model.drop(columns=['운전자한정특별약관'])
    
    encoded_df = processor.manual_label_encode(df_for_model)
    
    # Target variable: Accident Status (사고유무)
    X = encoded_df.drop(columns=['사고유무'])
    y = encoded_df['사고유무']
    
    # 4. Model Training and Evaluation
    print("Step 4: Training Random Forest Classifier...")
    X_train, X_test, y_train, y_test = trainer.split_data(X, y)
    
    rf_model = trainer.train_rf_classifier(X_train, y_train)
    metrics, y_proba = trainer.evaluate_classifier(rf_model, X_test, y_test)
    
    print("\nModel Performance Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"AUC:      {metrics['auc']:.4f}")
    print("\nClassification Report:\n", metrics['report'])
    
    # 5. Visualization (Optional - saves to Results folder)
    if not os.path.exists("Results"):
        os.makedirs("Results")
        
    print("Step 5: Generating plots...")
    roc_plot = trainer.plot_roc_curve(y_test, y_proba)
    roc_plot.savefig("Results/rf_roc_curve.png")
    
    cm_plot = trainer.plot_confusion_matrix(y_test, y_proba)
    cm_plot.savefig("Results/rf_confusion_matrix.png")
    
    print("\nPipeline completed successfully! Results saved in 'Results' directory.")

if __name__ == "__main__":
    main()
