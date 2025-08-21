from flask import Flask, request, render_template
import joblib
import numpy as np
import os

# --- Initialize the Flask App ---
# The 'template_folder' argument tells Flask to look for HTML files in the 'templates' directory.
app = Flask(__name__, template_folder='templates')

# --- Load the Saved Model Files ---
# This is done once when the app starts to avoid reloading on each request.
knn_model = None
scaler = None
label_encoder = None
feature_names = None
model_error = None  # Variable to store any error during model loading

try:
    # Define paths to model files within the 'model_files' directory
    model_dir = 'model_files'
    model_path = os.path.join(model_dir, 'knn_model.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
    features_path = os.path.join(model_dir, 'feature_names.pkl')

    # Check if all necessary model files exist before attempting to load them
    if not all(os.path.exists(p) for p in [model_path, scaler_path, encoder_path, features_path]):
        model_error = "One or more model files are missing. Please run the train_model.py script first to generate them."
        print(f"Error: {model_error}")
    else:
        # Load the files using joblib
        knn_model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        label_encoder = joblib.load(encoder_path)
        feature_names = joblib.load(features_path)
        print("Model and supporting files loaded successfully.")

except Exception as e:
    model_error = f"An unexpected error occurred during model loading: {e}"
    print(f"Error: {model_error}")


# --- Define App Routes ---

@app.route('/')
def home():
    """
    Renders the main page (index.html) with the input form.
    It passes the list of feature names and any model loading errors to the template.
    """
    return render_template('index.html', features=feature_names, error=model_error)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the form submission from the user. It processes the input,
    makes a prediction using the loaded model, and displays the result on a new page.
    """
    # If the model failed to load, prevent prediction and show the error.
    if model_error:
        return render_template('index.html', features=feature_names, error=model_error)

    try:
        # Extract features from the form. The order is determined by the 'feature_names' list
        # to ensure consistency with the model's training data.
        input_features = [float(request.form[feature]) for feature in feature_names]
        
        # Convert the list of features to a 2D numpy array for the scaler and model.
        final_features = np.array(input_features).reshape(1, -1)
        
        # Scale the features using the same scaler that was fit on the training data.
        scaled_features = scaler.transform(final_features)
        
        # Make a prediction using the trained KNN model.
        prediction_encoded = knn_model.predict(scaled_features)
        
        # Decode the numerical prediction back to its original string label (e.g., 'Tier-1').
        prediction_label = label_encoder.inverse_transform(prediction_encoded)
        
        # Render the result page (result.html) and pass the prediction to it.
        return render_template('result.html', prediction=prediction_label[0])

    except Exception as e:
        # Handle potential errors during prediction (e.g., user entering text instead of numbers).
        error_message = f"An error occurred during prediction: {e}"
        print(error_message)
        # Re-render the main page and display the error message.
        return render_template('index.html', features=feature_names, error=error_message)


# --- Run the App ---
if __name__ == '__main__':
    # The debug=True flag automatically reloads the server when you make changes to the code.
    # In a production environment, this should be set to False.
    app.run(debug=True)
