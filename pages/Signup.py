import streamlit as st
import streamlit_extras.switch_page_button as ste
import time
import pandas as pd
import numpy as np
st.set_page_config(layout="wide")
df = pd.read_csv("pages/Database.csv")
col1,col2 = st.columns([1,1])
with col1:
	st.image("Simple-OCR.gif")
	#st.image("Register logo.jpg")
with col2:
	col3,col4 = st.columns([1,1])
	with col4:
		st.header("Sign Up")
	uname = st.text_input("UserName: ")
	pwd = st.text_input("Password: ",type="password")
	col3,col4 = st.columns([1,1])
	with col4:
		button = st.button("Submit")
	if button:
		ubool = uname in list(df['Name'])
		if(ubool):
			st.warning('Username already in use', icon="❌")
		else:
			new_usr = pd.DataFrame({'Name':[uname],'pwd':[pwd]})
			df = pd.concat([df,new_usr],axis=0)
			df.to_csv('pages/Database.csv',index=False)
			st.warning('Account created successfully', icon="✅")
			time.sleep(1)
			st.warning('Redirecting to Login page', icon="⚙️")
			time.sleep(1)
			ste.switch_page("Login")