import pandas as pd
import numpy as np
import streamlit as st
import joblib
import datetime
import math
from PIL import Image
import traceback
##################################################################################################################################################

park_list = ['tdcp10', 'tdcp11', 'tdcp12', 'tdcp2', 'tdcp3', 'tdcp5', 'tdcp6',
       'tdcp7', 'tdcp8', 'tdcp9', 'tdstt2p2', 'tdc17p1', 'tdstt30', 'tdc1p5',
       'tdc1p4', 'tdc1p3', 'tdc1p2', 'tdc1p1', 'tdc17p3', 'tdstt3p1',
       'tdc17p2', 'tdstt5p1', 'tdstt38', 'tdstt37', 'tdc25p42', 'tdc25p41',
       'tdc25p40', 'tdc25p38', 'tdc25p37', 'tdc44p1', 'tdc43p2', 'tdc43p1',
       'tdstt31', 'tdc44p2', 'tdstt7p1', 'tdc2p1', 'tdstt4p1', 'tdstt10',
       'tdstt11', 'tdstt15', 'tdstt12', 'tdstt17', 'tdc27p1', 'tdc6p6',
       'tdc48p2', 'tdstt19', 'tdc48p1', 'tdc6p15', 'tdc6p17', 'tdc9p3',
       'tdc29p1', 'tdstt21', 'tdstt23', 'tdstt24', 'tdstt22', 'tdstt25',
       'tdstt26', 'tdstt27', 'tdstt28', 'tdc6p19', 'tdc6p20', 'tdc6p21',
       'tdc32p1', 'tdc32p2', 'tdc32p3', 'tdc32p4', 'tdc32p5', 'tdc32p6',
       'tdc33p1', 'tdc33p2', 'tdc36p1', 'tdc36p2', 'tdc39p1', 'tdc38p1',
       'tdc38p2', 'tdc42p1', 'tdc41p1', 'tdc41p2', 'tdstt29', 'tdstt39',
       'tdstt40', 'tdstt41', 'tdstt42', 'tdstt43', 'tdc44p3', 'tdc47p1',
       'tdstt35', 'tdstt36', 'tdstt34', 'tdstt32', 'tdstt33', 'tdstt46',
       'tdstt44', 'tdc25p43']
Central_Western = ['tdcp2', 'tdcp5', 'tdcp7', 'tdcp8']

Eastern = ['tdcp9', 'tdc25p42', 'tdc25p41', 'tdc25p40', 'tdc6p21', 'tdc39p1', 'tdc42p1', 'tdc41p1', 'tdstt33']

Islands = ['tdc2p1', 'tdc47p1', 'tdstt34']

Kowloon_City = ['tdc25p37', 'tdc27p1']

Kwai_Tsing = ['tdcp6']

Kwun_Tong = ['tdc43p1', 'tdstt31', 'tdc6p15', 'tdc33p1', 'tdc33p2']

North = ['tdstt10', 'tdstt24', 'tdstt39'] 

Sai_Kung = ['tdstt37', 'tdstt25', 'tdstt29', 'tdstt40', 'tdstt41', 'tdstt42', 'tdstt44'] 

Sha_Tin = ['tdstt2p2', 'tdstt38', 'tdstt4p1', 'tdstt11', 'tdc29p1', 'tdc32p4', 'tdc32p6', 'tdstt32'] 

Sham_Shui_Po = ['tdc48p2', 'tdc48p1', 'tdc6p19', 'tdc32p3'] 

Southern = ['tdcp10', 'tdc25p38', 'tdc6p20', 'tdc32p1', 'tdc32p2', 'tdc36p1', 'tdc36p2', 'tdc41p2', 'tdc44p2', 'tdc44p3', 'tdstt35'] 

Tai_Po = ['tdstt5p1', 'tdstt7p1', 'tdc6p6', 'tdc9p3', 'tdstt23'] 

Tsuen_Wan = ['tdcp3', 'tdstt46'] 

Tuen_Mun = ['tdstt3p1', 'tdc43p2', 'tdstt15', 'tdstt17', 'tdstt21', 'tdstt26', 'tdstt27', 'tdstt28', 'tdc38p2', 'tdstt36', 'tdc25p43'] 

Wan_Chai = ['tdcp11', 'tdc1p5', 'tdc1p4', 'tdc1p3', 'tdc1p2', 'tdc1p1', 'tdc38p1'] 

Wong_Tai_Sin = ['tdcp12', 'tdstt22'] 

Yau_Tsim_Mong = ['tdc17p1', 'tdstt30', 'tdc17p3', 'tdc17p2', 'tdstt19', 'tdc44p1'] 

Yuen_Long = ['tdstt12', 'tdc6p17', 'tdc32p5', 'tdstt43']
###########################################################################################################################################

def prediction(parklist,day,hour,minute):
    result = []
    for park in parklist:
        result.append(eval((f"{park}model")).predict(np.array([[day,hour,minute]]))[0])
    result_df = pd.DataFrame({"park_id":eval(district),"Vacancies":result})
    result_df['Vacancies'] = result_df['Vacancies'].apply(math.floor)
    result_df = pd.merge(result_df,info[['park_id','name_en','displayAddress_en']], how='left', on='park_id')
    result_df.set_index("park_id",inplace=True)
    result_df.sort_values(by=['Vacancies'],inplace=True,ascending=False)
    return result_df

for park in park_list:
        filename = park+"_dt.sav"
        globals()[f"{park}model"] = joblib.load("models/"+filename)

image = Image.open('logos.png')

st.title("Secure Car Park Vacancies Prediction")
st.image(image)
district = st.selectbox(
   'Please select district for prediction',
   ('Eastern','Kowloon City','Sai Kung','Kwai Tsing','Yau Tsim Mong','Tuen Mun','Wong Tai Sin','Southern','Islands','Yuen Long','Wan Chai',
'Sham Shui Po','Central Western','Sha Tin','Tai Po','Kwun Tong','Tsuen Wan','North'))

user_input_time = st.text_input("Please input time you want to predict.","2021-03-10 00:00")

try:
    info = pd.read_csv("car_park_district.csv")
    district = district.replace(" ", "_")
    converted_time =  datetime.datetime.strptime(user_input_time, '%Y-%m-%d %H:%M')
    day = converted_time.weekday()
    hour = converted_time.hour
    minute =converted_time.minute
    st.dataframe(prediction(eval(district),day,hour,minute))
    # st.write("Predicted Availibility: ",math.floor(loaded_model.predict(np.array([[day, hour, minute]]))[0]))
except Exception:
    #  st.write("Please enter a valid date in yyyy-mm-dd HH:MM")
    traceback.print_exc()
