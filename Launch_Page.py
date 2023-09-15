import streamlit as st
import base64

# Set page layout
st.set_page_config(
    page_title="A recommender system for Tiki.vn",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

###### GUI
page_bg_img = '''
<style>

[data-testid="stSidebar"] {
background: linear-gradient(-225deg, #5271C4 0%, #B19FFF 48%, #ECA1FE 100%);
}

</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

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

st.markdown('<h1 style="color:#CCFF00;">A recommender system for Tiki.vn 🛒</h1>', unsafe_allow_html=True)


# Hiển thị danh sách tên thành viên
st.markdown('<div style="color: #CCFF00; font-size: 28px;">Thành viên nhóm:</div>', unsafe_allow_html=True)
st.markdown('<div style="color: #CCFF00; font-size: 23px;">1. Nguyễn Văn Quyết</div>', unsafe_allow_html=True)
st.markdown('<div style="color: #CCFF00; font-size: 23px;">2. Tô Thị Lành</div>', unsafe_allow_html=True)
