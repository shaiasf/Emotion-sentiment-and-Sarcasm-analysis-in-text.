# Import necessary libraries
import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
import pandas as pd
import base64
from keras.preprocessing.sequence import pad_sequences


# Load tokenizer
with open('C:/Users/HP/Documents/S3/NLP/project/Sentiment/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load model
sentiment_model = load_model('C:/Users/HP/Documents/S3/NLP/project/Sentiment/sentiment_lstm_model.h5')
emotion_model = load_model('C:/Users/HP/Documents/S3/NLP/project/Emotion/BalanceNet_trained.h5')

# Define function to predict sentiment class
def predict_sentiment(text):
    sentiment_classes = ['Negative', 'Neutral', 'Positive']
    max_len = 50
    
    # Transforms text to a sequence of integers using a tokenizer object
    xt = tokenizer.texts_to_sequences(text)
    # Pad sequences to the same length
    xt = sequence.pad_sequences(xt, padding='post', maxlen=max_len)
    # Do the prediction using the loaded model
    yt = sentiment_model.predict(xt).argmax(axis=1)
    # Return the predicted sentiment
    return sentiment_classes[yt[0]]

def predict_emotion(text):
    emotion_classes = ["Neutral", "Happy", "Sad", "Love", "Anger"]
    max_len = 30
    
    # Transforms text to a sequence of integers using a tokenizer object
    xt = tokenizer.texts_to_sequences(text)
    # Pad sequences to the same length
    xt = sequence.pad_sequences(xt, padding='post', maxlen=max_len)

    data_int_t = pad_sequences(xt, padding='pre', maxlen=(max_len-5))
    data_test = pad_sequences(data_int_t, padding='post', maxlen=(max_len))
    # Do the prediction using the loaded model
    yt = emotion_model.predict(data_test).argmax(axis=1)
    # Return the predicted emotion
    return emotion_classes[yt[0]]

def predict_emotion_sequence(xt):
    emotion_classes = ["Neutral", "Happy", "Sad", "Love", "Anger"]
    max_len = 30

    xt = sequence.pad_sequences(xt, padding='post', maxlen=max_len)
    
    data_int_t = pad_sequences(xt, padding='pre', maxlen=(max_len-5))
    data_test = pad_sequences(data_int_t, padding='post', maxlen=(max_len))

    yt = emotion_model.predict(data_test).argmax(axis=-1)
    return emotion_classes[yt[0]]

# Streamlit App
def main():
    st.title("NLP Analysis App")
    
    # Sidebar navigation
    page = st.sidebar.selectbox("Select a page", ["Sentiment Analysis", "Emotion Detection"])
    
    if page == "Sentiment Analysis":
        st.header("Sentiment Analysis")
        st.markdown("Enter a sentence, and I'll predict its sentiment.")
        
        # User input
        user_input_sentiment = st.text_input("Enter text for sentiment analysis:")
        
        # Prediction for sentiment analysis
        if st.button("Predict Sentiment"):
            if user_input_sentiment:
                prediction_sentiment = predict_sentiment([user_input_sentiment])
                st.success(f"The predicted sentiment is: {prediction_sentiment}")
            else:
                st.warning("Please enter a sentence for sentiment analysis.")
    
    elif page == "Emotion Detection":
        st.header("Emotion Detection")
        st.markdown("Enter a sentence, and I'll predict its emotion.")
        
        # User input
        user_input_emotion = st.text_input("Enter text for emotion detection:")
        
        # Prediction for emotion detection
        if st.button("Predict Emotion"):
            if user_input_emotion:
                prediction_emotion = predict_emotion([user_input_emotion])
                st.success(f"The predicted emotion is: {prediction_emotion}")
            else:
                st.warning("Please enter a sentence for emotion detection.")
        
        # Option to upload a CSV file
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                # Read the CSV file into a DataFrame
                df = pd.read_csv(uploaded_file)


                # Tokenize the text in the DataFrame
                df['Tokenized Text'] = df.iloc[:, 0].apply(lambda text: tokenizer.texts_to_sequences([text]))

                # Predict emotion for each tokenized sequence in the DataFrame
                df['Predicted Emotion'] = df['Tokenized Text'].apply(predict_emotion_sequence)

             
                # Display the DataFrame with predicted emotions
                st.write("Predicted Emotions for Uploaded Texts:")
                st.write(df)
                
                # Allow the user to download the modified CSV file
                st.markdown(get_csv_download_link(df), unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error: {e}")



# Function to create a download link for a DataFrame as a CSV file
def get_csv_download_link(df):
    csv_file = df.to_csv(index=False)
    b64 = base64.b64encode(csv_file.encode()).decode()  # Encode as bytes and convert to string
    href = f'<a href="data:file/csv;base64,{b64}" download="predicted_emotions.csv">Download Predicted Emotions CSV</a>'
    return href

if __name__ == '__main__':
    main()
