import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import joblib
from sklearn.preprocessing import StandardScaler
import mysql.connector
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Set initial session states at the beginning of the script
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'home'


def custom_notification(message, message_type="info"):
    """
    Display a custom notification styled like a pop-up.

    Args:
    - message (str): The message to display.
    - message_type (str): The type of message ('info', 'success', 'warning', 'error').
    """
    if message_type == "info":
        color = "#29B6F6"  # Light blue
    elif message_type == "success":
        color = "#66BB6A"  # Green
    elif message_type == "warning":
        color = "#FFA726"  # Orange
    elif message_type == "error":
        color = "#EF5350"  # Red
    else:
        color = "#29B6F6"  # Default to info if an unrecognized type is provided

    st.markdown(f"""
    <div style="padding: 10px; border-radius: 10px; color: white; background-color: {color}; margin: 10px 0;">
        {message}
    </div>
    """, unsafe_allow_html=True)


# Database Connection
def create_db_connection():
    try:
        return mysql.connector.connect(
            host='localhost', user='root', password='', database='movie_recommender'
        )
    except Exception as e:
        st.error(f"The error '{e}' occurred")
        return None

# User Verification
def verify_user(username, password):
    connection = create_db_connection()
    if connection is not None:
        with connection.cursor(buffered=True) as cursor:
            query = "SELECT * FROM user WHERE namaUser = %s AND password = %s"
            cursor.execute(query, (username, password))
            result = cursor.fetchone()
        connection.close()
        return result
    return None

# Function to Build and Load the CNN Model
def build_and_load_model(model_weights_path):
    model = Sequential([
        InputLayer(shape=(25, 25, 3)),  # Updated this line
        Conv2D(32, kernel_size=(4, 4), activation='relu'),
        Conv2D(32, kernel_size=(4, 4), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model.load_weights(model_weights_path)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# CNN Prediction Function
def predict_image_cnn(image_array, model):
    if not isinstance(image_array, np.ndarray) or len(image_array.shape) != 3 or image_array.shape[2] != 3:
        raise ValueError("The image must be a numpy array with 3 channels (RGB).")
    image_array = tf.convert_to_tensor(image_array, dtype=tf.float32)
    image_array = tf.image.resize(image_array, (25, 25))
    image_array = image_array / 255.0
    image_array = tf.expand_dims(image_array, axis=0)
    prediction = model.predict(image_array)
    labels = ['Benign', 'Malignant']  # Define your labels based on the model's training
    predicted_label = labels[np.argmax(prediction)]
    return predicted_label

# SVM Prediction Function
def predict_csv_svm(file, clf):
    # Assuming the file is a CSV, we will read it into a DataFrame
    input_df = pd.read_csv(file)
    # Standardize the features by removing the mean and scaling to unit variance
    scaler = StandardScaler().fit(input_df)
    input_data = scaler.transform(input_df)

    # Make predictions
    predictions = clf.predict(input_data)
    # Convert numerical predictions to labels
    labels = ['Benign', 'Malignant']
    predicted_labels = [labels[pred] for pred in predictions]
    return predicted_labels

# Function to Load the SVM Model
def load_svm_model():
    clf = joblib.load('svm_model.joblib')
    return clf

# Load models
model_weights_path = "model.h5"
cnn_model = build_and_load_model(model_weights_path)
svm_model = load_svm_model()

def navigate_to_use_ml():
    # This function will set a state variable to trigger navigation.
    st.session_state['navigation'] = "Use ML"

def home_page():
    st.title("Welcome to the Breast Cancer Prediction Application")
    
    st.markdown("""
    This application leverages advanced machine learning to help predict breast cancer malignancy from diagnostic images and data. 
    Our goal is to empower individuals with cutting-edge technology for early detection and informed decision-making.
    """)
    
    st.image("Breast_Cancer_Detection_Journey.jpg", use_column_width=True)
    
    st.markdown("## Understanding Breast Cancer")
    st.video("https://www.youtube.com/watch?v=mCmJQGpjGNA")

    st.markdown("""
    Breast cancer develops when breast cells begin to grow abnormally. These cells divide more rapidly than healthy cells and continue to accumulate, forming a lump or mass. 
    Cells may spread (metastasize) through your breast to your lymph nodes or to other parts of your body.
    
    Early detection significantly increases the chances for successful treatment. Screening tests can detect breast cancer early, often before symptoms arise.
    """)
    
    st.markdown("## Early Detection Strategies")
    st.video("https://www.youtube.com/watch?v=9ByBCWt7-JM")
    
    st.markdown("""
    ### Screening and Awareness
    - **Self-examination:** Familiarize yourself with your breasts through self-exams. Look for any changes and report them to your healthcare provider.
    - **Mammograms:** Schedule regular mammograms for early detection even before symptoms appear.
    - **Clinical exams:** Have a healthcare provider check your breasts periodically.
    
    Learn more about early detection from the [American Cancer Society](https://www.cancer.org/cancer/breast-cancer/screening-tests-and-early-detection.html).
    """)

    # Survivor stories
    st.markdown("## Survivor Stories")

    # Define CSS to ensure all images are the same size.
    st.markdown("""
        <style>
            .image-container {
                display: flex;
                justify-content: center;
                align-items: center;
            }
            .image-container img {
                width: 100%;  /* Set the width you want */
                height: 250px; /* Set the height you want */
                object-fit: cover; /* This will handle the aspect ratio */
            }
        </style>
    """, unsafe_allow_html=True)

    #  Survivor data with images and links
    survivor_data = {
        "Charlotte's Story": {
            "image": "https://www.parkwaycancercentre.com/images/default-source/about-cancer/i-am-a-cancer-survivor_mdmgoh.webp?Status=Master&sfvrsn=211525fe_2",
            "link": "https://www.parkwaycancercentre.com/my/news-events/news-articles/news-articles-details/breast-cancer-stories-of-hope---i-am-a-cancer-survivor"
        },
        "Roberta's Journey": {
            "image": "https://ysm-res.cloudinary.com/image/upload/ar_16:9,c_fill,dpr_2.0,f_auto,g_faces:auto,q_auto:eco,w_1500/v1/yale-medicine/live-prod/ym_new/Roberta%20_4_edited_400031_5_v1.jpg",
            "link": "https://www.yalemedicine.org/survivor-stories/roberta-breast-cancer-survivor"
        },
        "Others' Perspectives": {
            "image": "https://www.foxchase.org/sites/default/files/styles/patient_story_alt_teaser/public/2024-01/IMG_1924.jpeg?itok=lfmWQ457",
            "link": "https://www.foxchase.org/clinical-care/conditions/breast-cancer/breast-cancer-patient-stories"
        }
    }

    # Create columns for images and wrap each image with the custom CSS class
    cols = st.columns(3)
    for col, (title, details) in zip(cols, survivor_data.items()):
        with col:
            st.markdown(f'<div class="image-container"><img src="{details["image"]}" alt="{title}"></div>', unsafe_allow_html=True)
            st.markdown(f"[Read More]({details['link']})", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    #### About the Breast Cancer Prediction Application
    For inquiries or further information, please reach out to us at [aloysius.dealson@student.usm.my](mailto:aloysius.dealson@student.usm.my).
    """)



def dashboard_page():
    st.title("Breast Cancer Data Dashboard")
    
    # Placeholder text
    st.write("This dashboard provides a visual analysis of breast cancer data metrics.")

    # Simulating some data for plotting
    data = pd.DataFrame({
        'type': ['Benign', 'Malignant', 'In-situ', 'Invasive'],
        'count': [50, 20, 15, 30]
    })

    # Horizontal bar chart using matplotlib
    st.subheader("Breast Cancer Cases")
    fig, ax = plt.subplots()
    ax.barh(data['type'], data['count'], color=['skyblue', 'red', 'green', 'orange'])
    ax.set_xlabel('Count')
    st.pyplot(fig)

    # Line chart using Streamlit built-in function
    st.subheader("Trends over Time")
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['Benign', 'In-situ', 'Malignant']
    )
    st.line_chart(chart_data)

    # Correlation heatmap using seaborn
    st.subheader("Feature Correlation Matrix")
    corr_data = np.random.randn(10, 10)
    corr_df = pd.DataFrame(corr_data, columns=[f'Feature {i}' for i in range(1, 11)])
    fig, ax = plt.subplots(figsize=(10, 7))  # Create a figure and a set of subplots
    sns.heatmap(corr_df.corr(), annot=True, ax=ax)  # ax=ax assigns the seaborn heatmap to the created axes
    st.pyplot(fig)  # Display the figure

    # Display some summary statistics
    st.subheader("Summary Statistics")
    st.table(data.describe())


def about_page():
    st.title("About the Breast Cancer Prediction Application")

    # Banner image at the top of the page
    st.image("/Users/dustin/Downloads/CAT405 Code/Banner.png", use_column_width=True)

    # Use columns to balance text and imagery
    col1, col2 = st.columns(2)

    with col1:
        st.header("Empowering Through Technology")
        st.markdown("""
            Our application harnesses the power of machine learning to provide early breast cancer detection capabilities. Through advanced image processing and data analysis techniques, we aim to offer a resource for individuals to assess their health with the assistance of artificial intelligence.

            **Key Features:**

            - **Innovative Technology**: Using state-of-the-art machine learning models to analyze medical data.
            - **User-Friendly Interface**: Designed for ease of use without compromising on functionality.
            - **Education and Awareness**: Providing users with valuable insights into breast cancer.

            **Technology Stack:**

            - **Frontend**: Developed with Streamlit.
            - **Machine Learning**: TensorFlow, Keras, and scikit-learn.
            - **Data Management**: MySQL.
        """)

    with col2:
        # Add vertical space before the image
        st.write("              ")  # This adds a little space. You can add multiple lines for more space.
        st.markdown("\n\n\n\n\n\n\n\n")  # Adds two lines of space
        st.image("/Users/dustin/Downloads/CAT405 Code/BC.jpeg", caption="Breast Cancer Awareness", use_column_width=True)

    # Section on commitment and additional information
    st.subheader("Our Commitment")
    st.markdown("""
        We are committed to contributing to healthcare advancements through technology. By providing tools for early detection, we hope to make a difference in the lives of many by harnessing the capabilities of AI and machine learning.
    """)

    st.subheader("Learn More")
    if st.button('Click here for more information'):
        st.balloons()
        st.markdown("[Breast Cancer Research Foundation](https://www.bcrf.org/)")

    # Footer separator
    st.markdown("---")
    
    # Our Team section
    st.header("Our Team")
    col3, col4 = st.columns([1, 2])
    with col3:
        st.image("/Users/dustin/Downloads/CAT405 Code/Dustin.jpg", width=200)

    with col4:
        st.subheader("Aloysius Dustin Dealson")
        st.markdown("""
            **Lead Data Scientist**

            Aloysius brings his extensive expertise in data science and machine learning to spearhead our analytics initiatives. With his leadership, our team develops cutting-edge algorithms that are the core of our breast cancer prediction platform.
        """)

    # Add more team members in a similar way
    # ...

        



# def use_ml_page():
#     st.title("Use ML")

#     # Choose model
#     option = st.selectbox(
#         'Choose the ML Model for Prediction:',
#         ('CNN Image Prediction', 'SVM CSV Prediction')  # Ensure both options are here
#     )

#     if option == 'CNN Image Prediction':
#         st.subheader("Image Prediction")
#         uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
#         if uploaded_image is not None:
#             image = Image.open(uploaded_image)
#             image_array = np.array(image)
#             prediction = predict_image_cnn(image_array, cnn_model)  # This function returns "Malignant" or "Benign"

#             if prediction == "Malignant":
#                 custom_notification("Don't Panic. Please consult a doctor.", "warning")
#                 with st.expander("See advice and information"):
#                     st.markdown("For guidance and more information, you may visit: [What to do if diagnosed with breast cancer](https://www.cancer.org/cancer/breast-cancer.html)")
#             elif prediction == "Benign":
#                 custom_notification("The prediction indicates a Benign tumor. Continue monitoring and maintaining a healthy lifestyle.", "success")
#                 with st.expander("See advice on maintaining a healthy lifestyle"):
#                     st.markdown("Learn how to keep a healthy life: [Maintaining a Healthy Lifestyle](https://www.healthline.com/health/healthy-lifestyle)")

#             # # Use the custom_notification function based on the prediction
#             # if prediction == "Malignant":
#             #     custom_notification("Don't Panic. Please consult a doctor.", "warning")
#             #     st.markdown("For guidance and more information, you may visit: [What to do if diagnosed with breast cancer](https://www.cancer.org/cancer/breast-cancer.html)")
#             # elif prediction == "Benign":
#             #     custom_notification("The prediction indicates a Benign tumor. Continue monitoring and maintaining a healthy lifestyle.", "success")
#             #     st.markdown("Learn how to keep a healthy life: [Maintaining a Healthy Lifestyle](https://www.healthline.com/health/healthy-lifestyle)")

#     elif option == 'SVM CSV Prediction':
#         st.subheader("CSV Prediction")
#         uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
#         if uploaded_file is not None:
#             predictions = predict_csv_svm(uploaded_file, svm_model)  # Assume this function returns a list of predictions

#             if prediction == "Malignant":
#                 custom_notification("Don't Panic. Please consult a doctor.", "warning")
#                 with st.expander("See advice and information"):
#                     st.markdown("For guidance and more information, you may visit: [What to do if diagnosed with breast cancer](https://www.cancer.org/cancer/breast-cancer.html)")
#             elif prediction == "Benign":
#                 custom_notification("The prediction indicates a Benign tumor. Continue monitoring and maintaining a healthy lifestyle.", "success")
#                 with st.expander("See advice on maintaining a healthy lifestyle"):
#                     st.markdown("Learn how to keep a healthy life: [Maintaining a Healthy Lifestyle](https://www.healthline.com/health/healthy-lifestyle)")

def use_ml_page():
    st.title("🔍 Use ML for Prediction")

    # Introduction
    st.markdown("""
        Welcome to the ML Prediction page! Choose between CNN Image Prediction and SVM CSV Prediction to get started with your breast cancer prediction.
    """)

    # No need to create a list of columns when only one is used
    # Choose model
    option = st.selectbox(
        'Choose the ML Model for Prediction:',
        ('CNN Image Prediction', 'SVM CSV Prediction')  # Ensure both options are here
    )
    
    if option == 'CNN Image Prediction':
        st.subheader("🖼️ Image Prediction")
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="image_uploader")
        # if uploaded_image is not None:
        #     # Resize the image to a maximum width/height while keeping aspect ratio
        #     image = Image.open(uploaded_image)
        #     image_array = np.array(image)
        #     image.thumbnail((300, 300))  # Resize to 300x300 maximum while keeping aspect ratio
        #     st.image(image, caption="Uploaded Image")
        #     prediction = predict_image_cnn(image_array, cnn_model)

        if uploaded_image is not None:
            # Open the image and convert it to an RGB array for prediction
            image = Image.open(uploaded_image).convert('RGB')
            image_array = np.array(image)
            # Create a thumbnail version for display
            display_image = image.copy()
            display_image.thumbnail((300, 300))  # Resize to 300x300 maximum while keeping aspect ratio
            st.image(display_image, caption="Uploaded Image")
            # Pass the original image array to the prediction function
            prediction = predict_image_cnn(image_array, cnn_model)

            # Use the custom_notification function based on the prediction
            if prediction == "Malignant":
                custom_notification("Don't Panic. Please consult a doctor.", "warning")
                st.markdown("For guidance and more information, you may visit: [What to do if diagnosed with breast cancer](https://www.cancer.org/cancer/breast-cancer.html)")
            elif prediction == "Benign":
                custom_notification("The prediction indicates a Benign tumor. Continue monitoring and maintaining a healthy lifestyle.", "success")
                st.markdown("Learn how to keep a healthy life: [Maintaining a Healthy Lifestyle](https://www.healthline.com/health/healthy-lifestyle)")

    elif option == 'SVM CSV Prediction':
        st.subheader("📊 CSV Prediction")
        uploaded_file = st.file_uploader("Upload your CSV file", type="csv", key="csv_uploader")
        if uploaded_file is not None:
            predictions = predict_csv_svm(uploaded_file, svm_model)  # This will now call your prediction function
        # Now use the 'predictions' list in your notifications
            if "Malignant" in predictions:
                custom_notification("Don't Panic. Please consult a doctor.", "warning")
                st.markdown("For guidance and more information, you may visit: [What to do if diagnosed with breast cancer](https://www.cancer.org/cancer/breast-cancer.html)")
            if all(pred == "Benign" for pred in predictions):
                custom_notification("All predictions indicate Benign tumors. Continue monitoring and maintaining a healthy lifestyle.", "success")
                st.markdown("Learn how to keep a healthy life: [Maintaining a Healthy Lifestyle](https://www.healthline.com/health/healthy-lifestyle)")


# Define your main application page function
def main_app():
    if 'nav_to_ml' in st.session_state and st.session_state['nav_to_ml']:
        use_ml_page()  # Navigate to the Use ML page
        st.session_state['nav_to_ml'] = False  # Reset the flag to avoid automatic redirection after refresh
    else:
        st.sidebar.image("/Users/dustin/Downloads/CAT405 Code/BCL_logo.png", width=100)  

        # Sidebar navigation
        st.sidebar.title("Navigation")
        choice = st.sidebar.radio("Go to", ["Home Page", "Dashboard", "About", "Use ML"])

        # Conditional navigation
        if choice == "Home Page":
            home_page()
        elif choice == "Dashboard":  
            dashboard_page() 
        elif choice == "About":
            about_page()
        elif choice == "Use ML":
            use_ml_page()


def remove_shadow_script():
    return """
    <script>
    window.onload = function() {
        var element = document.querySelector('.element-class');
        if (element) {
            element.style.boxShadow = 'none';
        }
    }
    </script>
    """

def login_page():
   # Custom styles for the login page
    st.markdown("""
        <style>
            /* Remove shadows and styles from all images and their containers */
            img, .login-container, .login-logo, .login-logo img::before, .login-logo img::after {
                box-shadow: none !important;
                filter: none !important;
                content: none !important;
            }

            /* Style the login container */
            .login-container {
                padding: 2rem;
                margin: 0 auto;
                max-width: 330px;
                background-color: #fff;
                border-radius: 10px;
            }

            /* Style the login logo container */
            .login-logo {
                display: flex;
                justify-content: center;
                margin-bottom: 1rem;
            }

            /* Style the logo image */
            .login-logo img {
                max-width: 150px; /* Control the size of the logo */
                margin: 0 auto; /* Center logo horizontally */
            }

            /* Style the login button */
            .stButton > button {
                width: 100%;
                padding: 0.5rem;
                margin-top: 1rem;
                border-radius: 5px; /* Rounded edges on button */
                font-size: 1rem;
                font-weight: bold;
            }
        </style>
    """, unsafe_allow_html=True)

    # Ensuring the login page is displayed without the sidebar and other Streamlit UI elements
    hide_streamlit_ui_style = """
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .stButton > button { /* Style for the button */
                width: 100%;
                padding: 10px; /* Larger button with more padding */
                border-radius: 5px; /* Rounded edges on button */
                font-size: 1rem;
                font-weight: bold;
            }
        </style>
    """
    st.markdown(hide_streamlit_ui_style, unsafe_allow_html=True)

    # Display the logo
    st.image("/Users/dustin/Downloads/CAT405 Code/BCL_logo.png", width=100, use_column_width=True)

    # Login form
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    username = st.text_input('Username', placeholder="Enter your username")
    password = st.text_input('Password', type='password', placeholder="Enter your password")
    if st.button('Login'):
        if verify_user(username, password):
            st.session_state['logged_in'] = True
            st.rerun()  # Updated method to rerun the app
        else:
            st.error('Incorrect username or password')
    st.markdown('</div>', unsafe_allow_html=True)



# Show the appropriate page based on login status
if 'logged_in' in st.session_state and st.session_state['logged_in']:
    main_app()  # If logged in, show the main app
else:
    login_page()  # If not logged in, show the login page

# This code listens for a button click on the "Use ML" link and updates the state
if 'navigation' in st.session_state and st.session_state['navigation'] == 'Use ML':
    st.session_state['nav_to_ml'] = True
    st.experimental_rerun()
