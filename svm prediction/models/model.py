import pandas as pd
import pickle
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from utils.preprocessing import preprocess_data

def train_model(X_train, y_train):
    if X_train.empty or y_train.empty:
        raise ValueError("X_train and y_train cannot be empty")
    X_train = preprocess_data(X_train)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    svc = svm.SVC(kernel='rbf', C=2)
    svc.fit(X_train, y_train)
    pickle.dump(svc, open('model.pkl', 'wb'))
    pickle.dump(scaler, open('scaler.pkl', 'wb'))
    return svc, scaler

def evaluate_model(X_test, y_test):
    if X_test.empty or y_test.empty:
        raise ValueError("X_test and y_test cannot be empty")
    X_test = preprocess_data(X_test)
    model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    X_test = scaler.transform(X_test)
    svc_predicted = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, svc_predicted)
    acc = accuracy_score(y_test, svc_predicted)
    class_report = classification_report(y_test, svc_predicted)
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Accuracy: {acc}")
    print(f"Classification Report:\n{class_report}")
    return conf_matrix, acc, class_report

def train_and_evaluate(data_path='data/heart.csv'):
    data = pd.read_csv(data_path)
    y = data["target"]
    X = data.drop('target', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    model, scaler = train_model(X_train, y_train)
    evaluate_model(X_test, y_test)
    return model, scaler

if __name__ == '__main__':
    train_and_evaluate()
