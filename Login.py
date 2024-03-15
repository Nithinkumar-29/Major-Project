import streamlit as st
import time
import streamlit_extras.switch_page_button as ste
import pandas as pd
import numpy as np
st.set_page_config(layout="wide")
if 'button1' not in st.session_state:
    st.session_state["button1"] = False
if 'button2' not in st.session_state:
    st.session_state.button2 = False
if 'button3' not in st.session_state:
    st.session_state.button3 = False
if 'button4' not in st.session_state:
    st.session_state.button4 = False
if 'login' not in st.session_state:
    st.session_state.login = False
df = pd.read_csv("pages/Database.csv")
col1,col2 = st.columns([1,1])
with col1:
	#video_file = open('video.mp4', 'rb')
	#video_bytes = video_file.read()
	#st.video(video_bytes,start_time=1)
	st.image("ocr.gif")
with col2:
	col3,col4 = st.columns([1,1])
	with col4:
		st.header("Login")
	uname = st.text_input("UserName: ")
	pwd = st.text_input("Password: ",type="password")
	col3,col4 = st.columns([1,1])
	with col4:
		button = st.button("Submit")
	if button:
		ubool = uname in list(df['Name'])
		if(ubool):
			ind = list(df['Name']).index(uname)
			if(pwd==df['pwd'][ind]):
				st.session_state.login=True
				st.warning('Login Successful', icon="✅")
				time.sleep(1)
				st.warning('Redirecting to Home page', icon="⚙️")
				time.sleep(1)
				ste.switch_page("Home")
			else:
				st.warning('Wrong password', icon="❌")
		else:
			st.warning('Not a registered user', icon="⚠️")
			time.sleep(1)
	col3,col4,col5 = st.columns([8,8,15])
	with col5:
		st.markdown("[Sign Up](Signup)")