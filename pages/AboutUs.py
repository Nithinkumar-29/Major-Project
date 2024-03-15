import streamlit as st
st.set_page_config(layout="wide")
col1,col2,col3 = st.columns([3,1,3])
with col2:
	st.header("About Us")
st.divider()
col1,col2,col3 = st.columns([1,1,1])
with col2:
	st.image("clg-logo.jpg")
col1,col2,col3 = st.columns([1,3,1])
with col2:
	st.markdown("##### This application is developed as a Mini project by the students of CVR College of Engineering.")
st.divider()
st.subheader("Goal:")
col1,col2 = st.columns([4,12])
with col1:
	st.image("ocr.png")
with col2:
	st.write("The primary motive for developing our project lies in the intersection of language recognition, OCR (Optical Character Recognition), translation, and text-to-speech technologies. With an increasing interconnectedness in our globalized world, the ability to seamlessly understand and communicate across languages becomes essential. Our project aims to address this need by leveraging image recognition to identify the language within an image, then employing OCR to extract textual content. Subsequently, the extracted text undergoes translation into a preferred language, enabling users to comprehend the content regardless of its original language. Finally, utilizing text-to-speech capabilities, the translated text is converted into spoken words, enhancing accessibility and facilitating comprehension for users with varying linguistic abilities. By integrating these functionalities, our project endeavors to bridge linguistic barriers and facilitate smoother communication and understanding across diverse cultural and linguistic landscapes.")
st.divider()
st.subheader("Developers:")
st.write("We are a team driven by the problems that surround us and strive to provide a solution accessible to all. Our team's motivation is to develop a system that recognizes languages in images, performs OCR, translation, and text-to-speech to facilitate cross-linguistic communication.")
col1,col2,col3 = st.columns([3,3,3])
with col1:
	st.markdown("##### Abhinaya Rapolu")
with col2:
	st.markdown("##### Jagan Mohan Kandukuri")
with col3:
	st.markdown("##### Nithin Kumar Reddy Vangala")
col1,col2,col3 = st.columns([3,3,3])
with col1:
	st.write("Roll No.: 20B81A6603")
with col2:
	st.write("Roll No.: 20B81A6622")
with col3:
	st.write("Roll No.: 20B81A6629")
col1,col2,col3 = st.columns([3,3,3])
with col1:
	st.write("Phone No.: 9100504749")
with col2:
	st.write("Phone No.: 9390822554")
with col3:
	st.write("Phone No.: 9502072420")
col1,col2,col3 = st.columns([3,3,3])
with col1:
	st.write("20B81A6603@cvr.ac.in")
with col2:
	st.write("20B81A6622@cvr.ac.in")
with col3:
	st.write("20B81A6629@cvr.ac.in")
st.divider()
st.subheader("Thank you for visiting our website")
st.write("For any queries, Please email us through any of the addresses mentioned above")
