import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
from lime import lime_image
from matplotlib.lines import Line2D
import cv2
import io
import base64

def calculate_jaccard_coefficient(true_segments, lime_segments):
    """Calculate the Jaccard coefficient between two segment arrays."""
    intersection = np.logical_and(true_segments, lime_segments)
    union = np.logical_or(true_segments, lime_segments)
    jaccard_coefficient = np.sum(intersection) / np.sum(union)
    return jaccard_coefficient

def slic_segmentation(image, n_segments=100, compactness=10):
    """
    Perform SLIC segmentation on the input image.
    
    Parameters:
    - image: Input image (grayscale or RGB).
    - n_segments: Number of desired segments (superpixels).
    - compactness: Compactness parameter for SLIC.
    
    Returns:
    - segments: Labeled segments.
    - segmented_image: Original image with segment boundaries.
    """
    segments = slic(image, n_segments=n_segments, compactness=compactness)
    segmented_image = mark_boundaries(image, segments)
    return segments, segmented_image

def perturb_image(image):
    """
    Add random noise to the image.
    
    Parameters:
    - image: Input image to perturb.
    
    Returns:
    - perturbed_image: Image with added noise.
    """
    perturbed_image = image + np.random.normal(0, 0.1, image.shape)
    return perturbed_image

def get_image_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return img_str

def explain_image(image, model, index=0, true_label=None, class_names=["Normal", "Pneumonia"], use_perturbed=False, n_segments=100, compactness=10):
    """
    Explain the model's prediction for an image using LIME.
    
    Parameters:
    - image: Input image to explain.
    - model: Trained model for prediction.
    - index: Index of the image (for reporting purposes).
    - true_label: Ground truth label of the image.
    - class_names: Names of the classes.
    - use_perturbed: Whether to use perturbed image for explanation.
    - n_segments: Number of segments for SLIC.
    - compactness: Compactness parameter for SLIC.
    
    Returns:
    - explanation: LIME explanation object.
    - explanation_data: Dictionary with explanation data and images.
    """
    # Preprocess the image if needed
    if image.max() > 1:
        processed_image = image / 255.0
    else:
        processed_image = image.copy()
    
    # Create explainer
    explainer = lime_image.LimeImageExplainer()
    
    # Get segmentation
    segments, segmented_image = slic_segmentation(processed_image, n_segments=n_segments, compactness=compactness)
    
    # Use perturbed image if requested
    if use_perturbed:
        perturbed_image = perturb_image(processed_image)
        explain_image = perturbed_image.astype('double')
    else:
        explain_image = processed_image.astype('double')
    
    # Get explanation
    explanation = explainer.explain_instance(
        explain_image,
        model.predict,
        top_labels=len(class_names),
        hide_color=0,
        num_samples=1000,
        num_features=10
    )
    
    # Calculate Jaccard coefficient
    jaccard_coefficient = calculate_jaccard_coefficient(segments, explanation.segments)
    print(f"Jaccard Coefficient for image {index}: {jaccard_coefficient}")
    
    # Calculate statistics for each segment
    mean_values = []
    variances = []
    std_devs = []
    
    for segment_num in np.unique(segments):
        segment_pixels = processed_image[segments == segment_num]
        mean_values.append(np.mean(segment_pixels))
        variances.append(np.var(segment_pixels))
        std_devs.append(np.std(segment_pixels))
    
    print(f"Mean Value: {np.mean(mean_values)}")
    print(f"Variance: {np.mean(variances)}")
    print(f"Standard Deviation: {np.mean(std_devs)}")
    
    # Create visual explanation
    fig, axs = plt.subplots(2, 3, figsize=(20, 8))
    
    # Get predicted class
    predicted_class = explanation.top_labels[0]
    predicted_label = class_names[predicted_class]
    
    # 1. Original image
    axs[0, 0].imshow(processed_image)
    if true_label:
        axs[0, 0].set_title(f'Original Image (True Label: {true_label})')
    else:
        axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')
    
    # 2. LIME mask
    mask_image = axs[0, 1].imshow(explanation.segments, cmap='viridis', alpha=0.5)
    axs[0, 1].set_title(f"LIME Mask - {predicted_label}")
    colorbar = plt.colorbar(mask_image, ax=axs[0, 1])
    colorbar.set_label('Importance')
    
    # Add legend to the mask image
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Supports Prediction', markerfacecolor='yellow', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Against Prediction', markerfacecolor='violet', markersize=10)
    ]
    axs[0, 1].legend(handles=legend_elements, loc='lower left')
    
    # 3. Combined image with boundaries
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=False,
        num_features=10,
        hide_rest=False
    )
    axs[0, 2].imshow(mark_boundaries(temp / 2 + 0.5, mask))
    
    title_text = f'Predicted: {predicted_label}'
    if true_label:
        title_text += f'\nGround Truth: {true_label}'
    title_text += '\nGreen Regions -> Supporting the predicted label\nRed Regions -> Against the predicted label'
    axs[0, 2].set_title(title_text)
    
    # 4. Superpixel regions with numbers
    superpixel_regions = explanation.segments
    axs[1, 0].imshow(superpixel_regions, cmap='viridis')
    axs[1, 0].set_title('Superpixel Regions with Numbers')
    
    for segment_num in np.unique(superpixel_regions):
        y, x = np.where(superpixel_regions == segment_num)
        if len(y) > 0 and len(x) > 0:  # Only add text if there are pixels in this segment
            axs[1, 0].text(np.mean(x), np.mean(y), str(segment_num), color='black', ha='center', va='center', fontsize=8)
    
    # 5. Feature importance bar plot
    features_list = []
    weights_list = []
    
    for i, class_name in enumerate(class_names):
        if i in explanation.local_exp:
            features = [f[0] for f in explanation.local_exp[i] if f[1] > 0]  # Only positive contributions
            weights = [f[1] for f in explanation.local_exp[i] if f[1] > 0]
            features_list.append(features)
            weights_list.append(weights)
    
    if len(features_list) >= 2:
        axs[1, 1].barh(features_list[0], weights_list[0], color='skyblue', label=class_names[0])
        axs[1, 1].barh(features_list[1], weights_list[1], color='orange', label=class_names[1])
        axs[1, 1].set_xlabel('Weight')
        axs[1, 1].set_title('Feature Importance')
        axs[1, 1].legend()
    
    # 6. Additional space for custom visualization or text
    axs[1, 2].axis('off')
    stats_text = f"Statistical Analysis:\n"
    stats_text += f"Mean Value: {np.mean(mean_values):.4f}\n"
    stats_text += f"Variance: {np.mean(variances):.4f}\n"
    stats_text += f"Standard Deviation: {np.mean(std_devs):.4f}\n"
    stats_text += f"Jaccard Coefficient: {jaccard_coefficient:.4f}"
    axs[1, 2].text(0.1, 0.5, stats_text, fontsize=12, va='center')
    
    plt.tight_layout()
    
    # Create explanation data dictionary
    explanation_data = {
        'predicted_class': predicted_label,
        'prediction_probability': float(explanation.predict_proba[predicted_class]),
        'mean_value': float(np.mean(mean_values)),
        'variance': float(np.mean(variances)),
        'std_dev': float(np.mean(std_devs)),
        'jaccard_coefficient': float(jaccard_coefficient),
        'segments': explanation.segments.tolist()
    }
    
    return explanation, explanation_data

# Example usage
def batch_explain_images(images, model, y_true=None, y_pred=None, class_names=["Normal", "Pneumonia"], n_images=10):
    """
    Explain a batch of images using LIME.
    
    Parameters:
    - images: List or array of images to explain.
    - model: Trained model for prediction.
    - y_true: Ground truth labels (optional).
    - y_pred: Predicted labels (optional).
    - class_names: Names of the classes.
    - n_images: Number of images to explain.
    
    Returns:
    - explanations: List of LIME explanation objects.
    - explanation_data: List of dictionaries with explanation data.
    """
    explanations = []
    explanation_data_list = []
    
    for index, img in enumerate(images[:n_images]):
        if y_true is not None:
            true_label = class_names[y_true[index]]
        else:
            true_label = None
        
        explanation, explanation_data = explain_image(
            img, 
            model, 
            index=index, 
            true_label=true_label,
            class_names=class_names
        )
        
        explanations.append(explanation)
        explanation_data_list.append(explanation_data)
        
        plt.show()
    
    return explanations, explanation_data_list