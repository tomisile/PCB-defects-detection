#from io import StringIO
import streamlit as st
#import pickle
from PIL import Image
import warnings
import time
#import cv2 as cv
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
    with st.sidebar:
        st.markdown('# How it works \n **PCB fabrication process** often suffers from defects, '
                        'such as \n - missing holes ')
        st.image('missing_hole.jpg', caption=None, width=150)
        st.markdown('\n - spur')
        st.image('spur.jpg', caption=None, width=150)
        st.markdown('\n - spurious copper')
        st.image('spurious_copper.jpg', caption=None, width=150)
        st.markdown('\n - short')
        st.image('short.jpg', caption=None, width=150)
        st.markdown('\n - open circuit')
        st.image('open_circuit.jpg', caption=None, width=150)
        st.markdown('\n - mouse bites')
        st.image('mouse_bite.jpg', caption=None, width=150)
        st.markdown('\n These defects, if not detected before electronic components are fabricated onto the board, '
                    'poses serious dangers to the components and its use')
    
    # Main page
    st.title('PCB Defects Detection')
    st.subheader('Detects up to 6 types of defects that occurs during Printed Circuit Board fabrication')
    
    # view sample checkbox
    with st.expander('View sample', expanded=False):
        image1 = Image.open("sample_image.jpg")
        st.image(image1, caption='Sample: defective PCB with open circuit')

    template_options = ["01.JPG", "04.JPG", "05.JPG",
                        "06.JPG", "07.JPG", "08.JPG",
                        "09.JPG", "10.JPG", "11.JPG", "12.JPG"]
    
    test_options = ["01_missing_hole.jpg", "01_mouse_bite.jpg", "04_spurious_copper.jpg",
                    "05_spur.jpg", "08_short.jpg", "09_mouse_bite.jpg",
                    "10_mouse_bite.jpg", "11_short.jpg", "12_short.jpg"]
    
    tab1, tab2, tab3 = st.tabs(["Select Template", "Upload Query", "Detect Defect"])
    
    with tab1:
        # select template
        template_source = st.radio("Choose template source",
                        ('Use a pre-loaded template', 'Upload a custom template'))

        if template_source == 'Use a pre-loaded template':
            option = st.selectbox('Choose from 10 different templates',
                options=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
            )
            selection = template_options[option-1]
            disp_selection = Image.open(selection)
            st.image(disp_selection, caption="Selected Template", width=500)
        
        else:
            selection = st.file_uploader("Upload your custom template image",
                                        type=['png', 'jpg'], accept_multiple_files=False)
            if selection is not None:
                st.image(selection, caption='template image', width=500)        

    with tab2:
        # select test
        test_source = st.radio("Choose test source",
                            ('Use a pre-loaded test image', 'Upload a custom test image'))

        if test_source == 'Use a pre-loaded test image':
            test_option = st.selectbox('Choose from 9 different test samples',
                options=(1, 2, 3, 4, 5, 6, 7, 8, 9)
            )
            test_selection = test_options[test_option-1]
            disp_test_selection = Image.open(test_selection)
            st.image(disp_test_selection, caption="Selected Test Image", width=500)
        
        else:
            test_selection = st.file_uploader("Send us a PCB image that matches the template you selected",
                                            type=['png', 'jpg'], accept_multiple_files=False)
            
            if test_selection is not None:
                st.image(test_selection, caption='test image', width=500)

        # on_click of predict button
        if st.button('View difference'):
            with st.spinner('Wait for a few seconds...'):
                time.sleep(2)
            # run image processing pipeline
            st.image(process_image(test_img=test_selection, template_img=selection),
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