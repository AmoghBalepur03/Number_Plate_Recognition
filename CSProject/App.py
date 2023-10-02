from re import L
from matplotlib import image
from save_in_csv import predict_image
from tensorflow.python.keras.models import load_model
import numpy as np
from tkinter import *
from nembir_pilate_localijation import main as number_plate_localizer
from save_in_csv import *
from PIL import Image
from imaje_resije import conTO28x28, image_resize_sklearn


#********************************************************************************************************************************************************************************

def stream_lit_app():
#STREAMLIT_APP
    try:
        import cv2
        from nembir_pilate_localijation import main as number_plate_localizer
        import streamlit as st  #Web App
        from PIL import Image #Image Processing
        import numpy as np #Image Processing 
        import os
        
        
#title
        st.set_page_config(page_title="Autoatic NumberPLate Recogniton" )
        st.title("NUMBER PLATE RECOGNITION")
#subtitle


#image uploader
        image = st.sidebar.file_uploader(label = "Upload the image of the car here",type=['png','jpg','jpeg'])
        if image is not None:
            #CHECKING IF IMAGE EXISTS OR NOT 

            input_image = Image.open(image) #read image
            file_details = image.name,image.type
            with open(os.path.join("tempdir" ,"temp.png"),"wb") as f:
                f.write(image.getbuffer())
            #saving numberplate in a directory called tempdir

        
            st.sidebar.image(input_image) 
        #display image
            st.sidebar.write(file_details[0])
            
        #displaying name of file user uploaded
        
            st.sidebar.success("Image successfully uploaded!")
            st.balloons()
        
            if st.sidebar.button("Click here to read the numer plate!"):
                number_plate_localizer("tempdir/temp.png")
                format='.png'
                myDir = "plates"
                def createFileList(myDir, format='.png'):
                    fileList = []
                    for root, dirs, files in os.walk(myDir, topdown=False):
                        for name in files:
                            # print(name)
                            if name.endswith(format):
                                fullName = os.path.join(root, name)
                                fileList.append(fullName)
                    return fileList
                plates = createFileList(myDir)
                if plates == []:
                    st.write("NumberPlate Not Found")
                else:
                    try:
                        for image in createFileList(myDir):
                            pred = predict_image(image)
                            if len(pred) > 4:
                                img=Image.open(image)
                                st.write("")
                                st.write("")
                                st.write("")
                                st.markdown('Localized NumberPlate')
                                st.image(img , caption="Localized Numberplate")
                        
                                st.write("\t\t\t\tPrediction Is " ,"\n\t\t\t\t" , pred)
                                save_in_csv()
                        
                
                    except:
                        pass                
                    #st.write("Can't Read The Numberplate")

        else:
         st.sidebar.write("Upload an Image")

       
    except:
        #st.write("Can't Read the Numberplate")
        pass
       
                     

        st.caption('''BY Sathish 
                  and Amogh''')     

#*********************************************************************************************************************************************************************************

stream_lit_app()
            
          





