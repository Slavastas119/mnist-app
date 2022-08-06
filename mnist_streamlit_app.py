import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import xgboost as xgb
from skimage import color
from skimage.transform import resize



# ML

def load_models(sklearn_file='sklearn_models.joblib', 
                xgboost_file='mnist_xgboost.json', 
                xgboost_pca_file='mnist_xgboost_pca.json', 
                cnn_file='mnist_cnn.h5', 
                dnn_file='mnist_dnn.h5'):
    # SKLearn models
    models = joblib.load(sklearn_file)
    # XGBoost models
    clf = xgb.XGBClassifier()
    models['xgboost'] = xgb.XGBClassifier()
    models['xgboost'].load_model(xgboost_file)
    models['xgboost_pca'] = xgb.XGBClassifier()
    models['xgboost_pca'].load_model(xgboost_pca_file)
    # Keras models
    models['cnn'] = load_model(cnn_file)
    models['dnn'] = load_model(dnn_file)
    
    return models

def load_df_accuracy(file='test_accuracy.joblib'):
    df_accuracy = pd.DataFrame.from_dict(joblib.load(file), orient='index') \
                                .rename(columns={0: 'Accuracy (test)'}) \
                                .mul(100).round(2)
    df_accuracy.index.name = 'Model'
    
    return df_accuracy

def make_df_predictions(models=load_models(), img=None, img_file='temp.jpg'):
    img = plt.imread(img_file) if img is None else img
    if img.shape[2] == 4:
        img = color.rgba2rgb(img)
    img_gray = color.rgb2gray(img)
    image_resized = resize(img_gray, (28, 28), anti_aliasing=True)
    
    df_proba = pd.DataFrame.from_dict({'cnn': models['cnn'].predict(image_resized.reshape(1, *image_resized.shape, 1), verbose=0)[0],
                                       'logistic_regression': models['logistic_regression'].predict_proba(image_resized.reshape(1, image_resized.size))[0],
                                       'logistic_regression_pca': models['logistic_regression_pca'].predict_proba(models['pca'].transform(image_resized.reshape(1, image_resized.size)))[0],
                                       'xgboost': models['xgboost'].predict_proba(image_resized.reshape(1, image_resized.size))[0],
                                       'xgboost_pca': models['xgboost_pca'].predict_proba(models['pca'].transform(image_resized.reshape(1, image_resized.size)))[0]},
                                      orient='index') \
                                .mul(100).round(2)
    df_pred = pd.DataFrame(df_proba.apply(np.argmax, axis=1)).rename(columns={0: 'Predicted number'})
    
    return df_pred, df_proba
    

def enlarge_image(image_resized, factor=10):
    img = []
    for row in image_resized:
        new_row = np.repeat(row, factor).tolist()
        for _ in range(factor):
            img.append(new_row)
    return np.array(img)


models = load_models()
df_accuracy = load_df_accuracy()






# Streamlit

st.title("Recognize handwritten digits")
st.markdown("""
### Based on [the MNIST dataset](http://yann.lecun.com/exdb/mnist/).
""")
st.sidebar.header("Configuration")

# Specify canvas parameters in application
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:",
    ("freedraw", "line", "rect", "circle", "transform", "polygon", "point"),
)
stroke_width = st.sidebar.slider("Stroke width: ", 10, 30, 20)
if drawing_mode == 'point':
    point_display_radius = st.sidebar.slider("Point display radius: ", 10, 30, 10)
stroke_color = st.sidebar.color_picker("Stroke color hex: ", "#FFFFFF")
bg_color = st.sidebar.color_picker("Background color hex: ")
#bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
realtime_update = st.sidebar.checkbox("Update in realtime", False)

col1, col2 = st.columns(2)
with col1:
    st.markdown('##### Drawing window (280x280)')
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        #background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=realtime_update,
        height=280,
        width=280,
        drawing_mode=drawing_mode,
        point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
        display_toolbar=st.sidebar.checkbox("Display toolbar", True),
        key="full_app",
    )

# Do something interesting with the image data and paths
if canvas_result.image_data is not None:
    #st.image(canvas_result.image_data)
    img = color.rgba2rgb(canvas_result.image_data)
    img_gray = color.rgb2gray(img)
    image_resized = resize(img_gray, (28, 28), anti_aliasing=True)
    
    df_pred, df_proba = make_df_predictions(models=models, img=canvas_result.image_data)
    df = pd.concat([df_accuracy, df_pred, df_proba], axis=1)
    
    with col2:
        st.markdown('##### Reshaped image (28x28)')
        st.image(enlarge_image(image_resized))
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('##### PCA result (5x5)')
        st.image(enlarge_image(models['pca'].transform(image_resized.reshape(1, image_resized.size)).reshape(5, 5), factor=280//5), clamp=True)
    with col2:
        st.markdown('##### Deconvolutional image (28x28)')
        st.image(enlarge_image(models['dnn'].predict(df_pred.loc['cnn',].values.reshape(-1, 1), verbose=0)[0]))
    
    st.write('-'*100)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('##### Models performance on the test set.')
        st.dataframe(df_accuracy.style.background_gradient(axis=0, vmin=80, vmax=100).format(precision=2))
    with col2:
        st.markdown('##### Prediction results.')
        st.dataframe(df_pred.style.format("{:20d}"))
    
    st.markdown('##### Predicted probabilities for each digit.')
    st.dataframe(df_proba.style.background_gradient(axis=1, vmin=0, vmax=100).format("{:7.2f}"))
