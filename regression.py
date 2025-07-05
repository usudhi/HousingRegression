import pandas as pd
from utils import (
    load_data, explore_data, preprocess_data,
    get_baseline_models, get_hyperparameter_grids,
    perform_hyperparameter_tuning, evaluate_model,
    plot_predictions, compare_models, save_results_to_file
)


def run_pipeline():
    print("📦 Loading dataset...")
    df = load_data()

    print("\n🔍 Exploring dataset...")
    explore_data(df)

    print("\n Preprocessing...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    print("\n Training baseline models...")
    models = get_baseline_models()
    baseline_results = []

    for name, model in models.items():
        print(f"\n  Training: {name}")
        model.fit(X_train, y_train)
        result = evaluate_model(model, X_test, y_test, f"{name} (Baseline)")
        baseline_results.append(result)
    print("\n Tuning hyperparameters...")
    param_grids = get_hyperparameter_grids()
    tuned_models = perform_hyperparameter_tuning(models, param_grids, X_train, y_train)

    print("\n Evaluating tuned models...")
    tuned_results = []

    for name, model in tuned_models.items():
        result = evaluate_model(model, X_test, y_test, f"{name} (Tuned)")
        tuned_results.append(result)
        plot_predictions(y_test, result['predictions'], f"{name} (Tuned)")

    all_results = baseline_results + tuned_results

    print("\n Comparing all models...")
    compare_models(all_results)

    print("\n Saving results to file...")
    save_results_to_file(all_results, "hypertuned_results.txt")

    print("\n Pipeline completed! Check plots and results in output folder.")




if __name__ == "__main__":
    run_pipeline()
