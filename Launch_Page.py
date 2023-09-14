import streamlit as st
import base64

# Set page layout
st.set_page_config(
    page_title="A recommender system for Tiki.vn",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Set background image
def background_image(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

image_file = 'background.png'
background_image(image_file)

st.markdown("# A recommender system for Tiki.vn 🛒")

