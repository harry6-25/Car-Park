import pandas as pd
import numpy as np
import streamlit as st
df = pd.read_csv("Lilith.csv")
import joblib
import datetime
import math
from Pillow import Image
filename = 'polyreg_model.sav'

loaded_model = joblib.load(filename)

image1 = Image.open('HKSTP.png')
image2 = Image.open('preface.png')
st.title("Car Park Vacancies Prediction")
st.image([image1,image2],width=345)
district = st.selectbox(
   'Please select district for prediction',
   ('Eastern','Kowloon City','Sai Kung','Kwai Tsing','Yau Tsim Mong','Tuen Mun','Wong Tai Sin','Southern','Islands','Yuen Long','Wan Chai',
'Sham Shui Po','Central & Western','Sha Tin','Tai Po','Kwun Tong','Tsuen Wan','North'))

user_input_time = st.text_input("Please input time you want to predict.")
try:
    converted_time =  datetime.datetime.strptime(user_input_time, '%Y-%m-%d %H:%M:%S')
    day = converted_time.weekday()
    hour = converted_time.hour
    minute =converted_time.minute
    st.write("Predicted Availibility: ",math.floor(loaded_model.predict(np.array([[day, hour, minute]]))[0]))
except:
    st.write("Please enter a valid date in yyyy-mm-dd HH:MM:SS")
