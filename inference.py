import os
import json
import joblib
import numpy as np

def model_fn(model_dir):
    """Load the model from the model_dir."""
    model_path = os.path.join(model_dir, "model.joblib")
    model = joblib.load(model_path)
    return model

def input_fn(request_body, content_type='text/csv'):
    """Parse input data from a CSV format."""
    if content_type == 'text/csv':
        # Convert CSV input to numpy array.
        return np.array([float(x) for x in request_body.split(',')]).reshape(1, -1)
    raise ValueError("Unsupported content type: " + content_type)

def predict_fn(input_data, model):
    """Apply model to the incoming request."""
    predictions = model.predict_proba(input_data)
    return predictions

def output_fn(prediction, accept='application/json'):
    """Format prediction output."""
    result = prediction.tolist()
    return json.dumps(result), accept
