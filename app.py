from io import StringIO
import streamlit as st
import pickle
from PIL import Image
import warnings
import time
import cv2 as cv
from image_processing import process_image

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="PCB Defects Detection",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://github.com/tomisile/PCB-defects-detection/issues",
        'About': "# Detects up to 6 types of defects that occurs during Printed Circuit Board fabrication."
    }
)

# load trained model
#pickle_in = open("vqc_model.pkl", 'rb')
#mlpc = pickle.load(pickle_in)

def main():
    """ this function defines the webpage """

    # side-bar
    st.sidebar.markdown(' # How it works \n **PCB fabrication process** often suffers from defects, '
                        'such as _missing holes, spur, spurious coopper, short, open circuit and mouse bites_ '
                        '\nThese defects, if not detected before electronic components are fabricated onto the board, '
                        'poses serious dangers to the components and its use')
    
    # Main page
    st.title('PCB Defects Detection')
    st.subheader('Detects up to 6 types of defects that occurs during Printed Circuit Board fabrication')
    
    # view sample checkbox
    with st.expander('View sample', expanded=False):
        image1 = Image.open("sample_image.jpg")
        st.image(image1, caption='Sample: defective PCB with open circuit')

    template_options = ["templates/01.jpg", "templates/bw1.jpg", "templates/04.jpg", "templates/05.jpg",
                        "templates/06.jpg", "templates/07.jpg", "templates/08.jpg",
                        "templates/09.jpg", "templates/10.jpg", "templates/11.jpg", "templates/12.jpg"]
    
    tab1, tab2, tab3 = st.tabs(["Select Template", "Upload Query", "Detect Defect"])
    
    with tab1:
        # select template
        option = st.selectbox('Choose from 10 different templates',
            options=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
        )
        selection = template_options[option-1]
        disp_selection = Image.open(selection)
        st.image(disp_selection, caption="Selected Template", width=500)        

    with tab2:
        uploaded_file = st.file_uploader("Send us a PCB image that matches the template you selected",
                                        type=['png', 'jpg'], accept_multiple_files=False)
        if uploaded_file is not None:
            st.image(uploaded_file, caption='test image', width=500)

            # on_click of predict button
            if st.button('View difference'):
                with st.spinner('Wait for a few seconds...'):
                    time.sleep(2)
                # run image processing pipeline
                st.image(process_image(test_img=uploaded_file, template_img=selection),
                        caption=None, width=500
                )
                st.success('Finished morphological transformations')
    
    with tab3:
        st.write("Draws bounding box around defects")

        with open("sample_image.jpg", "rb") as file:
            btn = st.download_button(
                label="Download",
                data=file,
                file_name="sample_image.jpg",
                mime="image/jpg"
            )


# run webpage
if __name__ == '__main__':
    main()