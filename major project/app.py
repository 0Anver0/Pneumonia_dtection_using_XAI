from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import tensorflow as tf
import cv2
import traceback
import logging
from PIL import Image
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environment

# Import functions from the combined explainer
from lime_explainer import explain_image, get_image_base64

# Set up logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the model
model_path = "./PneumoniaLENET_LIME.h5"
model = None

def load_model():
    """Load the TensorFlow model"""
    global model
    try:
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            logger.info("Model loaded successfully")
        else:
            logger.error(f"Model file {model_path} not found. Please train the model first.")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(traceback.format_exc())

def preprocess_image(image_data):
    """
    Preprocess image data for the model.
    
    Parameters:
    - image_data: Image data as bytes or file path
    
    Returns:
    - Preprocessed image ready for prediction
    """
    try:
        if isinstance(image_data, str):
            # If image_data is a file path
            image = cv2.imread(image_data)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # If image_data is bytes
            try:
                # Try using PIL first (better for different image formats)
                image = Image.open(io.BytesIO(image_data))
                image = np.array(image)
                if len(image.shape) == 2:  # Convert grayscale to RGB if needed
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.shape[2] == 4:  # Convert RGBA to RGB if needed
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            except Exception as e:
                logger.warning(f"PIL processing failed, trying OpenCV: {str(e)}")
                # Fallback to OpenCV
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize the image to the expected input size
        image = cv2.resize(image, (128, 128))
        
        return image
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request and generate LIME explanation."""
    try:
        if request.method == 'POST':
            # Check if model is loaded
            if model is None:
                load_model()
                if model is None:
                    return jsonify({
                        'error': 'Model not loaded. Please train the model first.'
                    }), 500
            
            # Get the image from the request
            file = request.files.get('file')
            if not file:
                return jsonify({'error': 'No file provided'}), 400
                
            img_bytes = file.read()
            logger.info(f"Received image with size: {len(img_bytes)} bytes")
            
            # Preprocess the image
            try:
                image = preprocess_image(img_bytes)
                logger.info(f"Image preprocessed with shape: {image.shape}")
            except Exception as e:
                logger.error(f"Error in preprocessing image: {str(e)}")
                return jsonify({'error': 'Failed to process image'}), 500
            
            # Make prediction
            try:
                # Create a normalized copy for the model input
                image_normalized = image / 255.0
                img_array = np.expand_dims(image_normalized, axis=0)
                prediction = model.predict(img_array)
                predicted_class = np.argmax(prediction[0])
                class_names = ["Normal", "Pneumonia"]
                prediction_probability = float(prediction[0][predicted_class])
                logger.info(f"Prediction: {class_names[predicted_class]} with probability {prediction_probability}")
            except Exception as e:
                logger.error(f"Error in making prediction: {str(e)}")
                return jsonify({'error': 'Failed to make prediction'}), 500
            
            # Generate explanation
            try:
                # Define a predict wrapper function that works with LIME
                def model_predict(images):
                    # LIME requires batched predictions
                    batch_size = len(images)
                    # Reshape images if needed
                    images_reshaped = np.array(images)
                    predictions = model.predict(images_reshaped)
                    return predictions
                
                # Generate LIME explanation
                explanation, explanation_data = explain_image(
                    image=image_normalized,  # Use normalized image for explanation
                    model=model,
                    index=0,  # Single image index
                    true_label=None,  # No ground truth in prediction mode
                    class_names=class_names,
                    use_perturbed=False,  # Use original image (not perturbed)
                    n_segments=100,  # Number of segments for SLIC
                    compactness=10  # Compactness parameter for SLIC
                )
                
                # Convert matplotlib figure to base64 for web display
                import matplotlib.pyplot as plt
                explanation_figure = plt.gcf()  # Get current figure
                explanation_image_str = get_image_base64(explanation_figure)
                plt.close()  # Close the figure to free memory
                
                # Add the base64 image to explanation data
                explanation_data['explanation_image'] = explanation_image_str
                
                logger.info("LIME explanation generated successfully")
            except Exception as e:
                logger.error(f"Error generating LIME explanation: {str(e)}")
                logger.error(traceback.format_exc())
                return jsonify({
                    'prediction': class_names[predicted_class],
                    'probability': prediction_probability,
                    'error': 'Failed to generate explanation'
                }), 200  # Return at least the prediction result
            
            # Return prediction results and explanation data
            return jsonify({
                'prediction': class_names[predicted_class],
                'probability': prediction_probability,
                'explanation': explanation_data
            })
    except Exception as e:
        logger.error(f"Unhandled exception in /predict route: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify the service is running."""
    status = {'status': 'ok'}
    if model is None:
        status['model'] = 'not loaded'
    else:
        status['model'] = 'loaded'
    return jsonify(status)

# Optional route for advanced analysis with specific parameters
@app.route('/analyze', methods=['POST'])
def analyze():
    """Advanced analysis endpoint with customizable parameters."""
    try:
        if request.method == 'POST':
            # Check if model is loaded
            if model is None:
                load_model()
                if model is None:
                    return jsonify({
                        'error': 'Model not loaded. Please train the model first.'
                    }), 500
            
            # Get the image file
            file = request.files.get('file')
            if not file:
                return jsonify({'error': 'No file provided'}), 400
            
            # Get analysis parameters
            params = request.form.to_dict()
            use_perturbed = params.get('use_perturbed', 'false').lower() == 'true'
            n_segments = int(params.get('n_segments', 100))
            compactness = float(params.get('compactness', 10))
            
            # Process the image
            img_bytes = file.read()
            image = preprocess_image(img_bytes)
            image_normalized = image / 255.0
            
            # Make prediction
            img_array = np.expand_dims(image_normalized, axis=0)
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction[0])
            class_names = ["Normal", "Pneumonia"]
            
            # Generate explanation with custom parameters
            explanation, explanation_data = explain_image(
                image=image_normalized,
                model=model,
                index=0,
                true_label=None,
                class_names=class_names,
                use_perturbed=use_perturbed,
                n_segments=n_segments,
                compactness=compactness
            )
            
            # Convert matplotlib figure to base64
            import matplotlib.pyplot as plt
            explanation_figure = plt.gcf()
            explanation_image_str = get_image_base64(explanation_figure)
            plt.close()
            
            # Add the base64 image to explanation data
            explanation_data['explanation_image'] = explanation_image_str
            
            return jsonify({
                'prediction': class_names[predicted_class],
                'probability': float(prediction[0][predicted_class]),
                'explanation': explanation_data,
                'parameters': {
                    'use_perturbed': use_perturbed,
                    'n_segments': n_segments,
                    'compactness': compactness
                }
            })
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load the model when the app starts
    load_model()
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)