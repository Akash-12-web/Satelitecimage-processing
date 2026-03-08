#### app.py
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

model = ResNet50(weights="imagenet")

def classify_land(img):

    img_resized = cv2.resize(img,(224,224))

    img_array = np.expand_dims(img_resized, axis=0)

    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)

    decoded = decode_predictions(preds, top=5)[0]

    labels = []
    probs = []

    for (_, label, prob) in decoded:
        labels.append(label)
        probs.append(prob)

    return labels, probs


st.title("Satellite Image Processing App")

# Dataset path
# Changed to a directory path where image files are expected to be.
# If your images are in a different directory, update this path accordingly.
dataset_path = "images"

# Get all images
# Check if the dataset_path is a directory
if not os.path.isdir(dataset_path):
    st.error(f"Error: The path '{dataset_path}' is not a directory or does not exist.")
    image_files = [] # No files to process
else:
    image_files = os.listdir(dataset_path)

# Filter for common image file extensions
image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')
image_files = [f for f in image_files if f.lower().endswith(image_extensions)]

if not image_files:
    st.error(f"No image files found in '{dataset_path}'. Please ensure the directory contains images or update the dataset_path.")
else:
    # Select image
    selected_image = st.selectbox("Select Image", image_files)

    img_path = os.path.join(dataset_path, selected_image)

    img = cv2.imread(img_path)
    if img is None:
        st.error(f"Could not load image {img_path}. Please check the file path and integrity.")
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        st.subheader("Original Image")
        st.image(gray, clamp=True)

        # Processing options
        options = st.multiselect(
            "Select Image Processing Operations",
            [
                "Sampling",
                "Quantization",
                "Padding",
                "Zoom",
                "Shrink",
                "Histogram",
                "Negative Transformation",
                "Spatial Filtering",
                "Masking",
                "Contrast Stretching",
                "Gray Level Slicing",
                "Bit Plane Slicing",
                "Histogram Equalization",
                "Gaussian Filter",
                "Edge Detection",
                "Land Use Classification"
            ]
        )

        # Apply selected operations

        for option in options:

            st.subheader(option)

            if option == "Sampling":
                sampled = cv2.resize(gray,(256,256))
                st.image(sampled, clamp=True)

            elif option == "Quantization":
                levels = 16
                quantized = np.floor(gray.astype(float)/(256.0/levels))*(256.0/levels)
                st.image(quantized.astype(np.uint8), clamp=True)

            elif option == "Padding":
                padded = cv2.copyMakeBorder(gray,50,50,50,50,cv2.BORDER_CONSTANT)
                st.image(padded, clamp=True)

            elif option == "Zoom":
                zoom = cv2.resize(gray,None,fx=2,fy=2)
                st.image(zoom, clamp=True)

            elif option == "Shrink":
                shrink = cv2.resize(gray,None,fx=0.5,fy=0.5)
                st.image(shrink, clamp=True)

            elif option == "Histogram":
                fig, ax = plt.subplots()
                ax.hist(gray.ravel(),256,[0,256])
                st.pyplot(fig)

            elif option == "Negative Transformation":
                negative = 255 - gray
                st.image(negative, clamp=True)

            elif option == "Spatial Filtering":
                kernel = np.ones((5,5))/25
                filtered = cv2.filter2D(gray,-1,kernel)
                st.image(filtered, clamp=True)

            elif option == "Masking":
                mask = np.zeros_like(gray)
                h, w = gray.shape
                mask_start_x, mask_end_x = max(0, w//4), min(w, w*3//4)
                mask_start_y, mask_end_y = max(0, h//4), min(h, h*3//4)
                mask[mask_start_y:mask_end_y, mask_start_x:mask_end_x] = 255
                masked = cv2.bitwise_and(gray,mask)
                st.image(masked, clamp=True)

            elif option == "Contrast Stretching":
                min_val = np.min(gray)
                max_val = np.max(gray)
                if (max_val - min_val) == 0:
                    contrast = np.zeros_like(gray)
                else:
                    contrast = ((gray.astype(float)-min_val)/(max_val-min_val)*255).astype(np.uint8)
                st.image(contrast, clamp=True)

            elif option == "Gray Level Slicing":
                slice_img = np.zeros_like(gray)
                slice_img[(gray>100)&(gray<150)] = 255
                st.image(slice_img, clamp=True)

            elif option == "Bit Plane Slicing":
                bit_plane = (gray >> 7) & 1
                st.image(bit_plane*255, clamp=True)

            elif option == "Histogram Equalization":
                equalized = cv2.equalizeHist(gray)
                st.image(equalized, clamp=True)

            elif option == "Gaussian Filter":
                gaussian = cv2.GaussianBlur(gray,(5,5),0)
                st.image(gaussian, clamp=True)

            elif option == "Edge Detection":
                edges = cv2.Canny(gray,100,200)
                st.image(edges, clamp=True)
            elif option == "Land Use Classification":

                    st.subheader("Land Use Classification")

                    labels, probs = classify_land(img)

                    # Convert to percentage
                    probs = [p*100 for p in probs]

                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots()

                    ax.bar(labels, probs)

                    ax.set_ylabel("Probability (%)")

                    ax.set_title("Land Use Prediction")

                    plt.xticks(rotation=45)

                    st.pyplot(fig)


print("Streamlit app saved to app.py")