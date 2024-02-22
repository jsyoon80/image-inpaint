"""Application to demo inpainting using streamlit.

Run using: streamlit run 10_03_image_inpaint_streamlit.py
"""

import streamlit as st
import pathlib
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import io
import base64
from PIL import Image

# 로고 준비해서 넣기
col1, col2, col3 = st.columns(3)
with col1:
    st.image('AiProVision_Disit_Logo.png')  # 'image_path'를 실제 이미지 경로로 바꿔주세요.
with col2:
    st.write('')
with col3:
    st.write('')

# Set title.
st.title('이미지 인페인트 프로그램')

# Set title.
st.sidebar.title('Image Inpaint')


STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / 'static'

# We create a downloads directory within the streamlit static asset directory
# and we write output files to it.
DOWNLOADS_PATH = (STREAMLIT_STATIC_PATH / "downloads")
if not DOWNLOADS_PATH.is_dir():
    DOWNLOADS_PATH.mkdir()


def get_image_download_link(img, filename, text):
    """Generates a link to download a particular image file."""
    buffered = io.BytesIO()
    img.save(buffered, format='JPEG')
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href



# Specify canvas parameters in application
uploaded_file = st.sidebar.file_uploader('이미지 파일을 선택하세요:', type=['jpg', 'jpeg', 'png'])
image = None
res = None

if uploaded_file is not None:

    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    stroke_width = st.sidebar.slider("스토로크 두께: ", 1, 25, 5)
    h, w = image.shape[:2]
    if w > 800:
        h_, w_ = int(h * 800 / w), 800
    else:
        h_, w_ = h, w

    # Create a canvas component.
    canvas_result = st_canvas(
        fill_color='white',
        stroke_width=stroke_width,
        stroke_color='black',
        background_image=Image.open(uploaded_file).resize((h_, w_)),
        update_streamlit=True,
        height=h_,
        width=w_,
        drawing_mode='freedraw',
        key="canvas",
    )
    stroke = canvas_result.image_data

    if stroke is not None:

        if st.sidebar.checkbox('페인팅 마스크 보기'):
            st.image(stroke)

        mask = cv2.split(stroke)[3]
        mask = np.uint8(mask)
        mask = cv2.resize(mask, (w, h))

    st.sidebar.caption('아래의 모드를 선택하여 주시기 바랍니다.')
    option = st.sidebar.selectbox('모드', ['None', 'Telea', 'NS', 'Telea와 NS 동시 비교'])

    if option == 'Telea':
        st.subheader('Telea 결과 이미지')
        res = cv2.inpaint(src=image, inpaintMask=mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)[:,:,::-1]
        st.image(res)
    elif option == 'Telea와 NS 동시 비교':
        col1, col2 = st.columns(2)
        res1 = cv2.inpaint(src=image, inpaintMask=mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)[:,:,::-1]
        res2 = cv2.inpaint(src=image, inpaintMask=mask, inpaintRadius=3, flags=cv2.INPAINT_NS)[:,:,::-1]
        with col1:
            st.subheader('Telea 결과 이미지')
            st.image(res1)
        with col2:
            st.subheader('NS 결과 이미지')
            st.image(res2)
        if res1 is not None:
            # Display link.
            result1 = Image.fromarray(res1)
            st.sidebar.markdown(
                get_image_download_link(result1, 'telea.png', 'Download Output of Telea'),
                unsafe_allow_html=True)
        if res2 is not None:
            # Display link.
            result2 = Image.fromarray(res2)
            st.sidebar.markdown(
                get_image_download_link(result2, 'ns.png', 'Download Output of NS'),
                unsafe_allow_html=True)

    elif option == 'NS':
        st.subheader('NS 결과 이미지')
        res = cv2.inpaint(src=image, inpaintMask=mask, inpaintRadius=3, flags=cv2.INPAINT_NS)[:,:,::-1]
        st.image(res)
    else:
        pass

    if res is not None:
        # Display link.
        result = Image.fromarray(res)
        st.sidebar.markdown(
            get_image_download_link(result, 'output.png', 'Download Output'),
            unsafe_allow_html=True)
