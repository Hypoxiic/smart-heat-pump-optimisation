import joblib
import matplotlib.pyplot as plt
import xgboost as xgb # Import xgboost to ensure its classes are recognized by joblib
import os # Added import os

def plot_loss_curve(model_path, output_path="loss_curve.png"):
    """
    Loads a trained XGBoost model and attempts to plot its loss curve.

    Args:
        model_path (str): Path to the saved XGBoost model (.joblib).
        output_path (str): Path to save the generated plot.
    """
    try:
        loaded_object = joblib.load(model_path)
        print(f"Object loaded successfully from {model_path}")

        model = None
        results = None # This will store the actual evaluation results data

        if isinstance(loaded_object, dict):
            print("Loaded object is a dictionary. Searching for model and/or evaluation results...")
            
            # Try to find the model object within the dictionary
            model_keys = ['model', 'estimator', 'bst', 'xgb_model', 'regressor', 'classifier']
            for key in model_keys:
                if key in loaded_object:
                    potential_model = loaded_object[key]
                    if hasattr(potential_model, 'evals_result') or isinstance(potential_model, xgb.Booster):
                        model = potential_model
                        print(f"Found potential model object under key: '{key}' of type {type(model)}")
                        break
            
            # Try to find evaluation results directly within the dictionary
            # This is useful if results were saved separately, or if the model is native and doesn't store them internally
            eval_keys = ['eval_results', 'history', 'evals_result', 'eval_metric', 'evals']
            for key in eval_keys:
                if key in loaded_object and isinstance(loaded_object[key], dict):
                    potential_results_dict = loaded_object[key]
                    # Basic check for the structure of eval results: dict of dicts of lists
                    if (potential_results_dict and 
                        all(isinstance(eval_set_metrics, dict) for eval_set_metrics in potential_results_dict.values()) and
                        all(all(isinstance(metric_values, list) for metric_values in eval_set_metrics.values()) 
                            for eval_set_metrics in potential_results_dict.values() if eval_set_metrics)):
                        results = potential_results_dict
                        print(f"Found potential evaluation results directly under key: '{key}'")
                        break
            
            if model is None and results is None:
                print("Could not find a recognizable XGBoost model or evaluation results within the dictionary.")
                print(f"Dictionary keys found: {list(loaded_object.keys())}")
                return
        else:
            # Assume the loaded object is the model itself
            if hasattr(loaded_object, 'evals_result') or isinstance(loaded_object, xgb.Booster):
                model = loaded_object
                print(f"Loaded object is assumed to be the model itself, type: {type(model)}")
            else:
                print(f"Loaded object is of type {type(loaded_object)}, which is not a recognized model type or a dictionary containing one.")
                return

        # Now, try to get results from the model if not already found in the dict
        if results is None and model is not None and hasattr(model, 'evals_result') and callable(model.evals_result):
            print("Attempting to get evaluation results from model.evals_result()")
            results = model.evals_result()

        # At this point, 'results' should be populated if data is available,
        # or 'model' might be a native Booster (and 'results' could be from the dict or still None).

        if not results: # This covers results being None or empty
            print("No evaluation results data found or it's empty.")
            if model is not None:
                if isinstance(model, xgb.Booster):
                    print("Model is a native XGBoost Booster. Evaluation results need to be saved and loaded separately if not found in the dictionary.")
                elif hasattr(model, 'evals_result'):
                    print("Model has 'evals_result' attribute, but no data was returned. Ensure it was trained with an evaluation set.")
                else:
                    print("The model object does not seem to have evaluation capabilities in the expected format.")
            else:
                print("No model object was successfully identified.")
            return

        if not isinstance(results, dict) or not all(isinstance(m, dict) for m in results.values()):
            print("Evaluation results are not in the expected format: {'eval_set_name': {'metric_name': [values]}}")
            print(f"Actual results structure: {type(results)}, Content (partial): {str(results)[:200]}")
            return
            
        # Determine the number of epochs.
        try:
            first_eval_set_name = next(iter(results))
            if not results[first_eval_set_name]: # Check if the first eval set has any metrics
                print(f"The first evaluation set '{first_eval_set_name}' has no metrics recorded.")
                return
            first_metric_name = next(iter(results[first_eval_set_name]))
            epochs = len(results[first_eval_set_name][first_metric_name])
        except (StopIteration, TypeError, KeyError) as e:
            print(f"Could not determine the number of epochs from the evaluation results: {e}")
            print("Results structure might be unexpected or empty:", results)
            return

        if epochs == 0:
            print("Evaluation results are present but empty (0 epochs recorded). Cannot plot.")
            return
            
        x_axis = range(0, epochs)
        fig, ax = plt.subplots(figsize=(12, 7))
        plot_successful = False

        for eval_set_name, metrics_dict in results.items():
            if not isinstance(metrics_dict, dict):
                print(f"Warning: Expected a dictionary of metrics for '{eval_set_name}', got {type(metrics_dict)}. Skipping.")
                continue
            for metric_name, values in metrics_dict.items():
                if isinstance(values, list) and len(values) == epochs:
                    ax.plot(x_axis, values, label=f'{eval_set_name} - {metric_name}')
                    plot_successful = True
                else:
                    print(f"Warning: Mismatch in length or type for {eval_set_name} - {metric_name}. Expected list of {epochs} values, got {type(values)} with length {len(values) if isinstance(values,list) else 'N/A'}. Skipping this metric.")
        
        if not plot_successful:
            print("No valid data series were found to plot after checking metric lengths and types.")
            return

        ax.legend()
        plt.ylabel('Loss / Metric Value')
        plt.xlabel('Epochs / Boosting Rounds')
        plt.title('XGBoost Model Training Curve')
        plt.grid(True)
        plt.tight_layout()

        # Ensure output directory exists before saving
        output_dir = os.path.dirname(output_path)
        if output_dir: # Make sure there's a directory part to create
            os.makedirs(output_dir, exist_ok=True)

        plt.savefig(output_path)
        print(f"Loss curve saved to {output_path}")
        # plt.show() # Uncomment if you want to display the plot interactively

    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
    except AttributeError as e:
        print(f"An AttributeError occurred: {e}. This might indicate an issue with model structure or missing methods.")
        print("This means the training history (loss curve data) is not available in the model object in the expected way, or the model object itself is not what was expected.")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    model_file_path = "models/price_predictor_xgb_tuned.joblib"
    print(f"Attempting to generate loss curve for model: {model_file_path}")
    plot_loss_curve(model_file_path, output_path="loss_curve.png") 