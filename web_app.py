import streamlit as st
from predict import make_prediction
from PIL import Image

st.title("Flower spicie Classification")
st.header("Flower spicie Classification")
st.caption("This image classifier recognizes different species of flowers. It tells you the most likely spicie your flower belongs to and the corresponding probability.")
st.text("Upload a flower image whose spicie you wish to predict")


uploaded_file = st.file_uploader("Choose a flower image ...", type="jpg")
top_k = st.number_input('Insert a number for the top most likely class(es)',min_value=1, max_value=5,value=1, step=1)
st.write('You want the top', top_k, 'most likely class(es)')
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Flower', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    predictions = make_prediction(uploaded_file, 'saved_model.h5', top_k, 'label_map.json')
    st.write(predictions)
    for label, prob in predictions.items():
        st.write('The flower spicie is',label,'with a probability of', prob[0])
