"""
Machine Learning Models

A collection of machine learning models implemented in Python.

This project includes a variety of models, including regression, classification, and clustering algorithms.
Each model is implemented using a combination of scikit-learn, TensorFlow, and PyTorch libraries.
"""

# dependencies
try:
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
    from sklearn.metrics import classification_report, confusion_matrix
except ImportError as e:
    print(f"Error: {e}")

# main module
def main():
    """
    Run the machine learning models.
    """
    # load data
    from sklearn.datasets import load_diabetes
    from sklearn.datasets import load_iris

    dataset = load_diabetes()
    X = dataset.data
    y = dataset.target

    dataset = load_iris()
    X_iris = dataset.data
    y_iris = dataset.target

    # train models
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model1 = LinearRegression()
    model1.fit(X_train, y_train)

    model2 = DecisionTreeRegressor()
    model2.fit(X_train, y_train)

    model3 = RandomForestRegressor()
    model3.fit(X_train, y_train)

    # evaluate models
    y_pred1 = model1.predict(X_test)
    y_pred2 = model2.predict(X_test)
    y_pred3 = model3.predict(X_test)

    print("Linear Regression:")
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred1))
    print("R-Squared Score:", r2_score(y_test, y_pred1))

    print("Decision Tree Regressor:")
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred2))
    print("R-Squared Score:", r2_score(y_test, y_pred2))

    print("Random Forest Regressor:")
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred3))
    print("R-Squared Score:", r2_score(y_test, y_pred3))

    # train and evaluate models on iris dataset
    X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)

    model1_iris = LinearRegression()
    model1_iris.fit(X_train_iris, y_train_iris)

    model2_iris = DecisionTreeRegressor()
    model2_iris.fit(X_train_iris, y_train_iris)

    model3_iris = RandomForestRegressor()
    model3_iris.fit(X_train_iris, y_train_iris)

    y_pred1_iris = model1_iris.predict(X_test_iris)
    y_pred2_iris = model2_iris.predict(X_test_iris)
    y_pred3_iris = model3_iris.predict(X_test_iris)

    print("\nLinear Regression on Iris Dataset:")
    print("Accuracy:", accuracy_score(y_test_iris, y_pred1_iris))
    print("Classification Report:\n", classification_report(y_test_iris, y_pred1_iris))
    print("Confusion Matrix:\n", confusion_matrix(y_test_iris, y_pred1_iris))

    print("Decision Tree Regressor on Iris Dataset:")
    print("Accuracy:", accuracy_score(y_test_iris, y_pred2_iris))
    print("Classification Report:\n", classification_report(y_test_iris, y_pred2_iris))
    print("Confusion Matrix:\n", confusion_matrix(y_test_iris, y_pred2_iris))

    print("Random Forest Regressor on Iris Dataset:")
    print("Accuracy:", accuracy_score(y_test_iris, y_pred3_iris))
    print("Classification Report:\n", classification_report(y_test_iris, y_pred3_iris))
    print("Confusion Matrix:\n", confusion_matrix(y_test_iris, y_pred3_iris))

if __name__ == "__main__":
    main()