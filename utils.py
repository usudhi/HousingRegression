import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib
matplotlib.use('Agg')  # Prevent Qt errors



def load_data():
    """Load Boston Housing dataset manually."""
    url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(url, sep=r"\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    feature_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]

    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target
    return df


def explore_data(df):
    """Display dataset info and generate correlation heatmap."""
    print("Dataset Shape:", df.shape)
    print("\n Dataset Info:")
    print(df.info())
    print("\n Dataset Description:")
    print(df.describe())
    print("\n Missing Values:\n", df.isnull().sum())

    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    return correlation_matrix


def preprocess_data(df, test_size=0.2, random_state=42):
    """Split data into train/test and apply feature scaling."""
    X = df.drop("MEDV", axis=1)
    y = df["MEDV"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def get_baseline_models():
    """Return dictionary of baseline models."""
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42)
    }


def get_hyperparameter_grids():
    """Return hyperparameter search grids."""
    return {
        "Decision Tree": {
            "max_depth": [3, 5, 10],
            "min_samples_split": [2, 5]
        },
        "Random Forest": {
            "n_estimators": [50, 100],
            "max_depth": [5, 10],
        }
    }


def perform_hyperparameter_tuning(models, param_grids, X_train, y_train):
    """Tune models using GridSearchCV."""
    tuned_models = {}

    for name, model in models.items():
        if name in param_grids:
            print(f" Tuning {name}...")
            grid = GridSearchCV(model, param_grids[name], cv=3, scoring="neg_mean_squared_error", n_jobs=-1)
            grid.fit(X_train, y_train)
            print(f" Best Params for {name}: {grid.best_params_}")
            tuned_models[name] = grid.best_estimator_
        else:
            print(f"  No tuning for {name}. Using default.")
            model.fit(X_train, y_train)
            tuned_models[name] = model

    return tuned_models


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a model and return its performance."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f" {model_name} -> MSE: {mse:.2f}, R²: {r2:.2f}")
    return {
        "model_name": model_name,
        "mse": mse,
        "r2": r2,
        "predictions": y_pred
    }


def plot_predictions(y_true, y_pred, title):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(6, 5))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{title} - Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}_prediction_plot.png")
    plt.close()


def compare_models(results):
    """Print model comparison."""
    df = pd.DataFrame(results)
    print("\n Model Comparison:\n", df[["model_name", "mse", "r2"]])
    return df


def save_results_to_file(results, filename):
    """Save evaluation results to a text file."""
    with open(filename, "w") as f:
        f.write("Model Performance Results\n")
        f.write("=" * 40 + "\n")
        for r in results:
            f.write(f"{r['model_name']}:\n")
            f.write(f"  MSE: {r['mse']:.2f}\n")
            f.write(f"  R²: {r['r2']:.2f}\n")
            f.write("-" * 30 + "\n")
