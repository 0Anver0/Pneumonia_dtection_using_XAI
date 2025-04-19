# #!/usr/bin/env python
# # coding: utf-8

# # In[25]:


# import pandas as pd
# import numpy as np
# import pathlib
# import cv2
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.utils import to_categorical
# from keras.callbacks import ReduceLROnPlateau, EarlyStopping
# import seaborn as sns
# import matplotlib.pyplot as plt
# from lime import lime_image
# from skimage.segmentation import mark_boundaries
# from matplotlib.lines import Line2D
# from skimage import segmentation, filters, color
# from skimage.util import img_as_ubyte
# import numpy as np
# import matplotlib.pyplot as plt
# from skimage import segmentation, filters, color
# from skimage.util import img_as_ubyte
# from skimage.segmentation import mark_boundaries
# from lime import lime_image
# from matplotlib.lines import Line2D



# # In[26]:


# path_dict = {
#     'train_data_dir_normal': r"C:\Users\beher\OneDrive\Desktop\seminar\major project\major project\chest_xray\train\NORMAL",
#     'train_data_dir_pneumonia': r"C:\Users\beher\OneDrive\Desktop\seminar\major project\major project\chest_xray\train\PNEUMONIA",
#     'test_data_dir_normal': r"C:\Users\beher\OneDrive\Desktop\seminar\major project\major project\chest_xray\test\NORMAL",
#     'test_data_dir_pneumonia': r"C:\Users\beher\OneDrive\Desktop\seminar\major project\major project\chest_xray\test\PNEUMONIA",
#     'data_val_dir_normal': r"C:\Users\beher\OneDrive\Desktop\seminar\major project\major project\chest_xray\val\NORMAL",
#     'data_val_dir_pneumonia': r"C:\Users\beher\OneDrive\Desktop\seminar\major project\major project\chest_xray\val\PNEUMONIA",
# }


# # In[27]:


# for key, value in path_dict.items():
#     path_dict[key] = pathlib.Path(value)


# # In[28]:


# X = []
# y = []

# # Define the new image size (e.g., 128x128 pixels)
# new_image_size = (128, 128)

# for key, value in path_dict.items():
#     images = list(path_dict[key].glob('*.jpeg'))
#     for img in images:
#         image = cv2.imread(str(img))
#         resized_img = cv2.resize(image, new_image_size)  # Resize to the new size
#         X.append(resized_img)
#         if 'normal' in key:
#             y.append(0)  # 0 - normal
#         else:
#             y.append(1)  # 1 - pneumonia

# X = np.array(X)
# y = np.array(y)


# # In[29]:


# print("X shape:", X.shape)
# print("y shape:", y.shape)


# # In[30]:


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# # In[31]:


# X_train.shape, X_test.shape, y_train.shape, y_test.shape


# # In[32]:


# X_train_scaled = X_train / 255
# X_test_scaled = X_test / 255


# # In[33]:


# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)


# # In[34]:


# # Create a LeNet-inspired model
# model = keras.Sequential([
#     keras.layers.Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=new_image_size + (3,)),
#     keras.layers.MaxPooling2D(pool_size=(2, 2)),
#     keras.layers.Conv2D(16, kernel_size=(5, 5), activation='relu'),
#     keras.layers.MaxPooling2D(pool_size=(2, 2)),
#     keras.layers.Flatten(),
#     keras.layers.Dense(120, activation='relu'),
#     keras.layers.Dense(84, activation='relu'),
#     keras.layers.Dense(2, activation='softmax')
# ])
# model.summary()


# # In[35]:


# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# # In[36]:


# # Added EarlyStopping callback to stop training if validation loss doesn't improve for 10 epochs
# early_stopping = EarlyStopping(monitor='val_loss', patience=25, verbose=1)


# # In[37]:


# model.fit(X_train_scaled, y_train, epochs=30, validation_data=(X_test_scaled, y_test), callbacks=[early_stopping])


# # In[38]:


# model.evaluate(X_test_scaled, y_test)


# # In[39]:


# y_pred = model.predict(X_test)
# y_pred[:5]


# # In[40]:


# y_pred_scaled = np.argmax(y_pred, axis=1)
# y_test_scaled = np.argmax(y_test, axis=1)


# # In[41]:


# confusion_matrix = confusion_matrix(y_test_scaled, y_pred_scaled)
# sns.heatmap(confusion_matrix, annot=True, fmt='d')


# # In[42]:


# def calculate_jaccard_coefficient(true_segments, lime_segments):
#     intersection = np.logical_and(true_segments, lime_segments)
#     union = np.logical_or(true_segments, lime_segments)
#     jaccard_coefficient = np.sum(intersection) / np.sum(union)
#     return jaccard_coefficient


# # In[43]:


# # Define the LimeImageExplainer
# explainer = lime_image.LimeImageExplainer()


# # In[44]:


# # Import required libraries
# import numpy as np
# import matplotlib.pyplot as plt
# from lime import lime_image
# from skimage.segmentation import slic, mark_boundaries
# from matplotlib.lines import Line2D

# # SLIC segmentation function
# def slic_segmentation(image, n_segments=100):
#     """
#     Perform SLIC segmentation on the input image.

#     Parameters:
#     - image: Input image (grayscale or RGB).
#     - n_segments: Number of desired segments (superpixels).

#     Returns:
#     - segments: Labeled segments.
#     - image: Original image, returned for compatibility.
#     """
#     segments = slic(image, n_segments=n_segments, compactness=10)
#     return segments, image  # Return the original image for further processing

# # Function to explain the image using LIME
# def explain_image(image, index, true_label, segments, image_segments):
#     explainer = lime_image.LimeImageExplainer()
#     explanation = explainer.explain_instance(
#         image.astype('double'),
#         model.predict,  # Replace `model.predict` with your prediction function
#         top_labels=5,
#         hide_color=0,
#         num_samples=1000
#     )
#     temp, mask = explanation.get_image_and_mask(
#         explanation.top_labels[0],
#         positive_only=False,
#         num_features=10,
#         hide_rest=False
#     )

#     # Statistics for superpixels
#     mean_values = []
#     variances = []
#     std_devs = []

#     for segment_num in np.unique(segments):
#         segment_pixels = image[segments == segment_num]  # Use the image directly
#         mean_values.append(np.mean(segment_pixels))
#         variances.append(np.var(segment_pixels))
#         std_devs.append(np.std(segment_pixels))

#     print(f"Mean Value: {np.mean(mean_values)}")
#     print(f"Variance: {np.mean(variances)}")
#     print(f"Standard Deviation: {np.mean(std_devs)}")
    
#     # Visualization
#     fig, axs = plt.subplots(2, 3, figsize=(20, 8))
    
#     # Original image
#     axs[0, 0].imshow(image)
#     axs[0, 0].set_title(f'Original Image (True Label: {true_label})')

#     # LIME mask
#     cmap = 'viridis'
#     mask_image = axs[0, 1].imshow(mask, cmap=cmap, alpha=0.5)
#     axs[0, 1].set_title(f"LIME Mask - {'Normal' if y_pred_scaled[index] == 0 else 'Pneumonia'}")
#     colorbar = plt.colorbar(mask_image, ax=axs[0, 1])
#     colorbar.set_label('Importance')
    
#     # Legend
#     legend_elements = [
#         Line2D([0], [0], marker='o', color='w', label='Supports Prediction', markerfacecolor='yellow', markersize=10),
#         Line2D([0], [0], marker='o', color='w', label='Against Prediction', markerfacecolor='violet', markersize=10)
#     ]
#     axs[0, 1].legend(handles=legend_elements, loc='lower left')
    
#     # Combined image with boundaries
#     axs[0, 2].imshow(mark_boundaries(temp / 2 + 0.5, mask))
#     axs[0, 2].set_title(
#         f'Predicted - {"Normal" if y_pred_scaled[index] == 0 else "Pneumonia"}\n'
#         f'Ground Truth - {"Normal" if y_test_scaled[index] == 0 else "Pneumonia"}\n'
#         'Green Regions -> Supporting the predicted label\n'
#         'Red Regions -> Against the predicted label'
#     )
    
#     # Superpixel regions
#     superpixel_regions = explanation.segments
#     axs[1, 0].imshow(superpixel_regions, cmap='viridis')
#     axs[1, 0].set_title('Superpixel Regions with Numbers')
    
#     for segment_num in np.unique(superpixel_regions):
#         y, x = np.where(superpixel_regions == segment_num)
#         axs[1, 0].text(np.mean(x), np.mean(y), str(segment_num), color='black', ha='center', va='center', fontsize=8)

#     # Feature importance bar plot
#     features_normal = [f[0] for f in explanation.local_exp[0] if f[1] > 0]
#     weights_normal = [f[1] for f in explanation.local_exp[0] if f[1] > 0]

#     features_pneumonia = [f[0] for f in explanation.local_exp[1] if f[1] > 0]
#     weights_pneumonia = [f[1] for f in explanation.local_exp[1] if f[1] > 0]

#     axs[1, 1].barh(features_normal, weights_normal, color='skyblue', label='Normal')
#     axs[1, 1].barh(features_pneumonia, weights_pneumonia, color='orange', label='Pneumonia')
#     axs[1, 1].set_xlabel('Weight')
#     axs[1, 1].set_title('Feature Importance')
#     axs[1, 1].legend()

#     plt.tight_layout()
#     plt.show()
#     return explanation

# # Loop through test images
# for index, img in enumerate(X_test_scaled[:10]):
#     true_label = 'Normal' if y_test_scaled[index] == 0 else 'Pneumonia'
    
#     # Obtain segments and image_segments
#     segments, image_segments = slic_segmentation(img)
    
#     # Explain the image
#     explain_image(img, index, true_label, segments, image_segments)


# # In[45]:


# #Quick-shift
# def explain_image(image, index, true_label, segments, image_segments):
#     explainer = lime_image.LimeImageExplainer()
#     explanation = explainer.explain_instance(image.astype('double'), model.predict, top_labels=5, hide_color=0, num_samples=1000)
#     temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
    
#     # Calculate Jaccard coefficient
#     true_segments, _ = slic_segmentation(image)
#     jaccard_coefficient = calculate_jaccard_coefficient(true_segments, explanation.segments)
#     print(f"Jaccard Coefficient for image {index + 1}: {jaccard_coefficient}")

#     mean_values = []
#     variances = []
#     std_devs = []

#     for segment_num in np.unique(segments):
#         segment_pixels = image_segments[segments == segment_num]
#         mean_values.append(np.mean(segment_pixels))
#         variances.append(np.var(segment_pixels))
#         std_devs.append(np.std(segment_pixels))

#     print(f"Mean Value: {np.mean(mean_values)}")
#     print(f"Variance: {np.mean(variances)}")
#     print(f"Standard Deviation: {np.mean(std_devs)}")
    
#     # Create a figure with 2 rows and 3 columns
#     fig, axs = plt.subplots(2, 3, figsize=(20, 8))
    
#     # Display the original image
#     axs[0, 0].imshow(image)
#     axs[0, 0].set_title('Original Image (True Label: {})'.format(true_label))

#     # Display the LIME mask with color representation and labels
#     cmap = 'viridis'
#     mask_image = axs[0, 1].imshow(mask, cmap=cmap, alpha=0.5)
#     axs[0, 1].set_title("LIME Mask - {}".format('Normal' if y_pred_scaled[index] == 0 else 'Pneumonia'))
#     colorbar = plt.colorbar(mask_image, ax=axs[0, 1])
#     colorbar.set_label('Importance')
    
#     # Add legend to the mask image
#     legend_elements = [Line2D([0], [0], marker='o', color='w', label='Supports Prediction', markerfacecolor='yellow', markersize=10),
#                        Line2D([0], [0], marker='o', color='w', label='Against Prediction', markerfacecolor='violet', markersize=10)]
#     axs[0, 1].legend(handles=legend_elements, loc='lower left')
    
#     # Display the combined image with boundaries
#     axs[0, 2].imshow(mark_boundaries(temp / 2 + 0.5, mask))
#     axs[0, 2].set_title('Predicted - ' + str('Normal' if y_pred_scaled[index] == 0 else 'Pneumonia') + '\n Ground Truth - ' + str('Normal' if y_test_scaled[index] == 0 else 'Pneumonia')
#              + ' \n Green Regions -> Supporting the predicted label \n Red Regions -> Against the predicted label')
    
#     # Display the superpixel regions with numbers
#     superpixel_regions = explanation.segments
#     axs[1, 0].imshow(superpixel_regions, cmap='viridis')
#     axs[1, 0].set_title('Superpixel Regions with Numbers')
    
#     for segment_num in np.unique(superpixel_regions):
#         y, x = np.where(superpixel_regions == segment_num)
#         axs[1, 0].text(np.mean(x), np.mean(y), str(segment_num), color='black', ha='center', va='center', fontsize=8)

#     # Generate a bar plot for features and their weights
#     features_normal = [f[0] for f in explanation.local_exp[0] if f[1] > 0]  # Only positive contributions
#     weights_normal = [f[1] for f in explanation.local_exp[0] if f[1] > 0]

#     features_pneumonia = [f[0] for f in explanation.local_exp[1] if f[1] > 0]  # Only positive contributions
#     weights_pneumonia = [f[1] for f in explanation.local_exp[1] if f[1] > 0]

#     axs[1, 1].barh(features_normal, weights_normal, color='skyblue', label='Normal')
#     axs[1, 1].barh(features_pneumonia, weights_pneumonia, color='orange', label='Pneumonia')
#     axs[1, 1].set_xlabel('Weight')
#     axs[1, 1].set_title('Feature Importance')
#     axs[1, 1].legend()

#     plt.tight_layout()
#     plt.show()
#     return explanation

# # Assuming you have model, X_test_scaled, y_pred_scaled, y_test_scaled, segments, and image_segments defined
# # Assuming you have model, X_test_scaled, y_pred_scaled, and y_test_scaled defined
# # Use the existing slic_segmentation function for segmentation
# for index, img in enumerate(X_test_scaled[:10]):
#     true_label = 'Normal' if y_test_scaled[index] == 0 else 'Pneumonia'
    
#     # Call slic_segmentation to obtain segments and image_segments
#     segments, image_segments = slic_segmentation(img)
    
#     # Call explain_image with the image, index, true_label, and explainer
#     explain_image(img, index, true_label, segments, image_segments)


# # In[46]:


# #SLIC
# import numpy as np
# import matplotlib.pyplot as plt
# from skimage.color import rgb2gray
# from skimage.segmentation import slic, mark_boundaries
# from lime import lime_image
# from matplotlib.lines import Line2D

# def slic_segmentation(image, n_segments=100, compactness=10):
#     """
#     Perform SLIC segmentation on the input image.
    
#     Parameters:
#     - image: Input image (grayscale or RGB).
#     - n_segments: Number of desired segments (superpixels).
    
#     Returns:
#     - segments: Labeled segments.
#     """
#     # Perform SLIC segmentation (converts to grayscale internally if needed)
#     segments = slic(image, n_segments=n_segments, compactness=compactness)
#     return segments, mark_boundaries(image, segments)

# def explain_image_slic(image, index, true_label, explainer):
#     # Ensure the image has only three channels (RGB)
#     if image.shape[-1] == 4:
#         image = image[..., :-1]  # Exclude the last channel (if alpha channel exists)

#     # Convert RGB to grayscale if needed
#     if image.ndim == 3 and image.shape[-1] == 3:
#         gray_img = rgb2gray(image)
#     else:
#         gray_img = image  # Already grayscale

#     # Call slic_segmentation to obtain segments
#     segments, image_segments = slic_segmentation(image)

#     # Create a wrapper function for model prediction
#     def wrapper_predict(images):
#         images = np.array([img[..., np.newaxis] if img.ndim == 2 else img for img in images])
#         return model.predict(images)

#     # Use LIME to explain the image
#     explanation = explainer.explain_instance(
#         gray_img.astype('double'),
#         wrapper_predict,
#         top_labels=5,
#         hide_color=0,
#         num_samples=1000
#     )

#     # Calculate Jaccard coefficient
#     true_segments, _ = slic_segmentation(image)
#     jaccard_coefficient = calculate_jaccard_coefficient(true_segments, explanation.segments)
#     print(f"Jaccard Coefficient for image {index + 1}: {jaccard_coefficient}")

#     # Calculate statistical metrics
#     mean_values, variances, std_devs = [], [], []
#     for segment_num in np.unique(segments):
#         segment_pixels = gray_img[segments == segment_num]
#         mean_values.append(np.mean(segment_pixels))
#         variances.append(np.var(segment_pixels))
#         std_devs.append(np.std(segment_pixels))

#     print(f"Mean Value: {np.mean(mean_values)}")
#     print(f"Variance: {np.mean(variances)}")
#     print(f"Standard Deviation: {np.mean(std_devs)}")

#     # Create a figure with 2 rows and 3 columns
#     fig, axs = plt.subplots(2, 3, figsize=(20, 8))

#     # Original Image
#     axs[0, 0].imshow(image, cmap='gray')
#     axs[0, 0].set_title(f'Original Image (True Label: {true_label})')

#     # LIME Mask Visualization
#     temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
#     axs[0, 1].imshow(mark_boundaries(temp / 2 + 0.5, mask))
#     axs[0, 1].set_title(f'LIME Explanation')

#     # Superpixel Regions with Numbers
#     superpixel_regions = explanation.segments
#     axs[1, 0].imshow(superpixel_regions, cmap='viridis')
#     axs[1, 0].set_title('Superpixel Regions with Numbers')
#     for segment_num in np.unique(superpixel_regions):
#         y, x = np.where(superpixel_regions == segment_num)
#         axs[1, 0].text(np.mean(x), np.mean(y), str(segment_num), color='black', ha='center', va='center', fontsize=8)

#     # Feature Importance Plot (Optional)
#     axs[1, 1].barh(range(len(mean_values)), mean_values, color='skyblue')
#     axs[1, 1].set_title('Feature Importance')

#     plt.tight_layout()
#     plt.show()
#     return explanation

# # Run the explanation for each image
# explainer = lime_image.LimeImageExplainer()
# for index, img in enumerate(X_test_scaled[:10]):
#     true_label = 'Normal' if y_test_scaled[index] == 0 else 'Pneumonia'
#     explain_image_slic(img, index, true_label, explainer)


# # In[47]:


# #noice
# def perturb_image(image):
#     # Add random noise to the image
#     perturbed_image = image + np.random.normal(0, 0.1, image.shape)
#     return perturbed_image

# def explain_image(image, index, true_label):
#     explainer = lime_image.LimeImageExplainer()
    
#     # Create a single perturbed image using perturb_image function
#     perturbed_image = perturb_image(image)
    
#     # Use explain_instance with the perturbed image
#     explanation = explainer.explain_instance(
#         perturbed_image.astype('double'), 
#         model.predict, 
#         top_labels=5, 
#         hide_color=0, 
#         num_samples=1000,
#         num_features=10  # Adjust the number of features as needed
#     )
    
#     # Calculate Jaccard coefficient
#     true_segments, _ = slic_segmentation(image)
#     jaccard_coefficient = calculate_jaccard_coefficient(true_segments, explanation.segments)
#     print(f"Jaccard Coefficient for image {index + 1}: {jaccard_coefficient}")

#     mean_values = []
#     variances = []
#     std_devs = []

#     for segment_num in np.unique(segments):
#         segment_pixels = image_segments[segments == segment_num]
#         mean_values.append(np.mean(segment_pixels))
#         variances.append(np.var(segment_pixels))
#         std_devs.append(np.std(segment_pixels))

#     print(f"Mean Value: {np.mean(mean_values)}")
#     print(f"Variance: {np.mean(variances)}")
#     print(f"Standard Deviation: {np.mean(std_devs)}")
    
#     # Rest of the code remains unchanged...


    
#     # Create a figure with 2 rows and 5 columns
#     fig, axs = plt.subplots(2, 3, figsize=(20, 8))

#     # Display the original image
#     axs[0, 0].imshow(image)
#     axs[0, 0].set_title('Original Image (True Label: {})'.format(true_label))

#     # Display the LIME mask with color representation and labels
#     cmap = 'viridis'
#     mask_image = axs[0, 1].imshow(explanation.segments, cmap=cmap, alpha=0.5)
#     axs[0, 1].set_title("LIME Mask - {}".format('Normal' if y_pred_scaled[index] == 0 else 'Pneumonia'))
#     colorbar = plt.colorbar(mask_image, ax=axs[0, 1])
#     colorbar.set_label('Importance')
    
#     # Add legend to the mask image
#     legend_elements = [Line2D([0], [0], marker='o', color='w', label='Supports Prediction', markerfacecolor='yellow', markersize=10),
#                        Line2D([0], [0], marker='o', color='w', label='Against Prediction', markerfacecolor='violet', markersize=10)]
#     axs[0, 1].legend(handles=legend_elements, loc='lower left')
    
#     # Display the combined image with boundaries
#     temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
#     axs[0, 2].imshow(mark_boundaries(temp / 2 + 0.5, mask))
#     axs[0, 2].set_title('Predicted - ' + str('Normal' if y_pred_scaled[index] == 0 else 'Pneumonia') + '\n Ground Truth - ' + str('Normal' if y_test_scaled[index] == 0 else 'Pneumonia')
#              + ' \n Green Regions -> Supporting the predicted label \n Red Regions -> Against the predicted label')
    
#     # Display the superpixel regions with numbers
#     superpixel_regions = explanation.segments
#     axs[1, 0].imshow(superpixel_regions, cmap='viridis')
#     axs[1, 0].set_title('Superpixel Regions with Numbers')
    
#     for segment_num in np.unique(superpixel_regions):
#         y, x = np.where(superpixel_regions == segment_num)
#         axs[1, 0].text(np.mean(x), np.mean(y), str(segment_num), color='black', ha='center', va='center', fontsize=8)

#     # Generate a bar plot for features and their weights
#     features_normal = [f[0] for f in explanation.local_exp[0] if f[1] > 0]  # Only positive contributions
#     weights_normal = [f[1] for f in explanation.local_exp[0] if f[1] > 0]

#     features_pneumonia = [f[0] for f in explanation.local_exp[1] if f[1] > 0]  # Only positive contributions
#     weights_pneumonia = [f[1] for f in explanation.local_exp[1] if f[1] > 0]

#     axs[1, 1].barh(features_normal, weights_normal, color='skyblue', label='Normal')
#     axs[1, 1].barh(features_pneumonia, weights_pneumonia, color='orange', label='Pneumonia')
#     axs[1, 1].set_xlabel('Weight')
#     axs[1, 1].set_title('Feature Importance')
#     axs[1, 1].legend()

#     plt.tight_layout()
#     plt.show()
#     return explanation
    
#     # Assuming you have model, X_test_scaled, y_pred_scaled, and y_test_scaled defined
# for index, img in enumerate(X_test_scaled[:10]):
#     true_label = 'Normal' if y_test_scaled[index] == 0 else 'Pneumonia'
#     explain_image(img, index, true_label)


# # In[48]:


# import gradio as gr
# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import Model

# # Load your saved model (ensure the path is correct)
# # Load your saved model (ensure the path is correct)
# model_path = "PneumoniaLENET_LIME.h5"  # Update to match your saved model file
# PneumoniaLENET_LIME = tf.keras.models.load_model(model_path)

# # Function to generate Grad-CAM heatmap
# def generate_gradcam(model, img, last_conv_layer_name="conv2d"):
#     img_array = tf.image.resize(img, (299, 299))
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)

#     grad_model = Model(
#         inputs=model.input,
#         outputs=[model.get_layer(last_conv_layer_name).output, model.output]
#     )

#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(img_array)
#         top_class = tf.argmax(predictions[0])
#         loss = predictions[:, top_class]

#     grads = tape.gradient(loss, conv_outputs)[0]
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#     conv_outputs = conv_outputs[0]
#     heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)
#     heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
#     return heatmap

# # Function to overlay heatmap on the input image
# def overlay_heatmap(img, heatmap, alpha=0.4, colormap="viridis"):
#     heatmap_resized = tf.image.resize(heatmap, (img.shape[0], img.shape[1])).numpy()
#     heatmap_resized = np.uint8(255 * heatmap_resized)

#     colormap = plt.get_cmap(colormap)
#     heatmap_colored = colormap(heatmap_resized)
#     overlay = heatmap_colored[:, :, :3] * alpha + img / 255.0
#     return np.uint8(255 * overlay)

# # Gradio interface function
# def classify_and_visualize_image(inp):
#     heatmap = generate_gradcam(PneumoniaLENET_LIME, inp, last_conv_layer_name="conv2d")
#     overlayed_img = overlay_heatmap(inp, heatmap)
#     return overlayed_img

# # Define Gradio components
# image_input = gr.Image(type="numpy", label="Upload X-ray Image")
# image_output = gr.Image(type="numpy", label="Heatmap Visualization")

# gr.Interface(
#     fn=classify_and_visualize_image,
#     inputs=image_input,
#     outputs=image_output,
#     title="Pneumonia Detection Heatmap",
#     description="Upload a chest X-ray image, and see the heatmap highlighting regions of interest for the model."
# ).launch()


# # In[ ]:





# # In[ ]:





# # In[ ]:





# # In[ ]:




import pandas as pd
import numpy as np
import pathlib
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
import os

def load_and_preprocess_data(path_dict, new_image_size=(128, 128)):
    """Load and preprocess the data from the given paths."""
    X = []
    y = []
    
    for key, value in path_dict.items():
        images = list(pathlib.Path(value).glob('*.jpeg'))
        for img in images:
            image = cv2.imread(str(img))
            resized_img = cv2.resize(image, new_image_size)
            X.append(resized_img)
            if 'normal' in key.lower():
                y.append(0)  # 0 - normal
            else:
                y.append(1)  # 1 - pneumonia
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y

def create_model(input_shape):
    """Create a LeNet-inspired model."""
    model = keras.Sequential([
        keras.layers.Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(16, kernel_size=(5, 5), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(120, activation='relu'),
        keras.layers.Dense(84, activation='relu'),
        keras.layers.Dense(2, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(X_train, y_train, X_test, y_test, model, epochs=30):
    """Train the model with the given data."""
    # Convert labels to categorical
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)
    
    # Scale the data
    X_train_scaled = X_train / 255.0
    X_test_scaled = X_test / 255.0
    
    # Set up callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=25, verbose=1)
    
    # Train the model
    history = model.fit(
        X_train_scaled, y_train_cat, 
        epochs=epochs, 
        validation_data=(X_test_scaled, y_test_cat),
        callbacks=[early_stopping]
    )
    
    return model, history, X_train_scaled, X_test_scaled, y_train_cat, y_test_cat

def evaluate_model(model, X_test_scaled, y_test_cat):
    """Evaluate the model and return metrics."""
    # Get predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test_cat, axis=1)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    
    # Calculate accuracy, precision, recall, and F1-score
    report = classification_report(y_test_classes, y_pred_classes, output_dict=True)
    
    return cm, report, y_pred_classes, y_test_classes

def save_model(model, path="PneumoniaLENET_LIME.h5"):
    """Save the trained model."""
    model.save(path)
    print(f"Model saved to {path}")

def main():
    # Define paths
    path_dict = {
        'train_data_dir_normal': "chest_xray/train/NORMAL",
        'train_data_dir_pneumonia': "chest_xray/train/PNEUMONIA",
        'test_data_dir_normal': "chest_xray/test/NORMAL",
        'test_data_dir_pneumonia': "chest_xray/test/PNEUMONIA",
        'data_val_dir_normal': "chest_xray/val/NORMAL",
        'data_val_dir_pneumonia': "chest_xray/val/PNEUMONIA",
    }
    
    # Load and preprocess data
    X, y = load_and_preprocess_data(path_dict)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Create and train the model
    model = create_model(input_shape=(128, 128, 3))
    model, history, X_train_scaled, X_test_scaled, y_train_cat, y_test_cat = train_model(
        X_train, y_train, X_test, y_test, model
    )
    
    # Evaluate the model
    cm, report, y_pred_classes, y_test_classes = evaluate_model(model, X_test_scaled, y_test_cat)
    
    # Print evaluation metrics
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test_classes, y_pred_classes))
    
    # Save the model
    save_model(model)
    
    return model, X_test_scaled, y_test_classes, y_pred_classes

if __name__ == "__main__":
    main()