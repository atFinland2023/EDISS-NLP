import os
import multiprocessing
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

def train_evaluate_model(model_name, model, X_train, y_train, X_test, y_test, result_path, shared_dict):
    logger = shared_dict['logger']
    parameters = shared_dict['parameters']

    logger.info(f"Training and evaluating {model_name}...")
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    logger.info(f'Accuracy for {model_name}: {accuracy}')

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.savefig(os.path.join(parameters['result_path'], f'confusion_matrix_{model_name}.png'))

    # Classification report
    logger.info(f'Classification Report for {model_name}:\n{classification_report(y_test, predictions)}')
    logger.info('-' * 50)

def test_and_evaluate_models(df, logger, parameters):
    models = {
        'Logistic Regression': LogisticRegression(C=1, solver="liblinear", max_iter=1000),
        'Multinomial Naive Bayes': MultinomialNB(),
        # 'Random Forest': RandomForestClassifier(),
        # 'XGBoost': XGBClassifier()
    }
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(df['tweet_text_processed'], df['sentiment_label'], test_size=parameters['test_size'], random_state=parameters['seed'])

    # Encode the labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Use CountVectorizer to convert text to numerical features
    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Shared dictionary for passing logger and parameters to parallel processes
    manager = multiprocessing.Manager()
    shared_dict = manager.dict({'logger': logger, 'parameters': parameters})

    with multiprocessing.Pool() as pool:
        # Use starmap to pass multiple arguments to the function
        pool.starmap(train_evaluate_model, [(model_name, model, X_train_vectorized, y_train_encoded, X_test_vectorized, y_test_encoded, parameters['result_path'], shared_dict) for model_name, model in models.items()])