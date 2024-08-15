# Import necessary libraries
import streamlit as st
import pickle
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
import re
from nltk.tokenize import word_tokenize
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string


nltk.download("punkt")
nltk.download("stopwords")



# Load tokenizer
with open('Sentiment/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Loading models
model = load_model('Sentiment/sentiment_lstm_model.h5')
emotion_model = load_model('Emotion/BalanceNet_trained.h5')
sarcasm_model = load_model('Sarcasm/sarcasm_detector.h5')


# Define function to predict sentiment class
def predict_sentiment(text):
    sentiment_classes = ['Negative', 'Neutral', 'Positive']
    max_len = 50
    
    # Transforms text to a sequence of integers using a tokenizer object
    xt = tokenizer.texts_to_sequences(text)
    # Pad sequences to the same length
    xt = sequence.pad_sequences(xt, padding='post', maxlen=max_len)
    # Do the prediction using the loaded model
    yt = model.predict(xt).argmax(axis=1)
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
    yt = emotion_model.predict(data_test).argmax(axis=-1)
    # Return the predicted emotion
    return emotion_classes[yt[0]]

# helper fucntions for sarcasm page

def clean_text(text):
    text = text.lower()
    
    pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    text = pattern.sub('', text)
    text = " ".join(filter(lambda x:x[0]!='@', text.split()))
    emoji = re.compile("["
                           u"\U0001F600-\U0001FFFF"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    
    text = emoji.sub(r'', text)
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)        
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text) 
    text = re.sub(r"\'ll", " will", text)  
    text = re.sub(r"\'ve", " have", text)  
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"did't", "did not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"couldn't", "could not", text)
    text = re.sub(r"have't", "have not", text)
    text = re.sub(r"[,.\"\'!@#$%^&*(){}?/;`~:<>+=-]", "", text)
    return text

def CleanTokenize(df):
    head_lines = list()
    lines = df["headline"].values.tolist()

    for line in lines:
        line = clean_text(line)
        # tokenize the text
        tokens = word_tokenize(line)
        # remove puntuations
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        # remove non alphabetic characters
        words = [word for word in stripped if word.isalpha()]
        stop_words = set(stopwords.words("english"))
        # remove stop words
        words = [w for w in words if not w in stop_words]
        head_lines.append(words)
    return head_lines

# Load the sarcasm tokenizer
with open('Sarcasm/tokenizer.pickle', 'rb') as handle:
    tokenizer_obj = pickle.load(handle)

def predict_sarcasm(text):
    # Create a DataFrame with the provided headline
    x_final = pd.DataFrame({"headline": [text]})
    
    # Clean and tokenize the headline
    test_lines = CleanTokenize(x_final)  # Assuming you have a function named CleanTokenize
    # Convert the text to sequences using the tokenizer
    test_sequences = tokenizer_obj.texts_to_sequences(test_lines)
    
    # Pad the sequences to the specified max length
    test_review_pad = pad_sequences(test_sequences, maxlen=25, padding='post')
    
    
    # Make predictions using the loaded model
    pred = sarcasm_model.predict(test_review_pad)
    # Multiply the prediction by 100 for better readability
    pred *= 100
    
    # Check if the sarcasm probability is greater than or equal to 50%
    if pred[0][0] >= 50:
        return "Sarcasm Detected!" 
    else:
        return "Not Sarcasm."



# Streamlit App
def main():
    st.title("NLP Analysis App")
    
    # Sidebar navigation
    page = st.sidebar.selectbox("Select a page", ["Sentiment Analysis", "Emotion Detection", "Sarcasm Detection"])
    
    if page == "Sentiment Analysis":
        # Navigation bar at the top
        nav_option = st.radio("Choose Analysis Type:", ["Single Sentence", "Batch Analysis"])

        if nav_option == "Single Sentence":
            st.header("Sentiment Analysis - Single Sentence")
            st.markdown("Enter a sentence, and I'll predict its sentiment.")
            
            # User input for single sentence analysis
            user_input_sentiment = st.text_input("Enter text for sentiment analysis:")
            
            # Prediction for sentiment analysis for a single sentence
            if st.button("Predict Sentiment"):
                if user_input_sentiment:
                    prediction_sentiment = predict_sentiment([user_input_sentiment])
                    st.success(f"The predicted sentiment is: {prediction_sentiment}")
                else:
                    st.warning("Please enter a sentence for sentiment analysis.")

        elif nav_option == "Batch Analysis":
            st.header("Sentiment Analysis - Batch Analysis")
            st.markdown("Upload a CSV file with a single column of sentences, and I'll predict the sentiment for each sentence.")

            # File upload for batch analysis
            uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

            if uploaded_file is not None:
                # Read CSV file
                sentences_df = pd.read_csv(uploaded_file)
                sentences_df=sentences_df.head(10)

                # Display uploaded data
                st.subheader("Uploaded Sentences:")
                st.write(sentences_df)
                column_ = sentences_df.columns[0]

                # Predict sentiment for each sentence
                # Predict sentiment for each sentence using a for loop

                for index, row in sentences_df.iterrows():
                    sentences_df.at[index, 'Predicted Sentiment'] = predict_sentiment([row[column_]])

                # Display predicted sentiment
                st.subheader("Predicted Sentiment:")
                st.write(sentences_df[[column_, 'Predicted Sentiment']])

                # Save results to a new CSV file
                save_button = st.button("Save Results to CSV")
                if save_button:
                    result_file_path = "SentimentResults.csv"
                    sentences_df.to_csv(result_file_path, index=False)
                    st.success(f"Results saved to {result_file_path}")


    elif page == "Emotion Detection":
        # Navigation bar at the top
        nav_option = st.radio("Choose Analysis Type:", ["Single Sentence", "Batch Analysis"])

        if nav_option == "Single Sentence":
            st.header("Emotion Analysis - Single Sentence")
            st.markdown("Enter a sentence, and I'll predict its emotion.")
            
            # User input for single sentence analysis
            user_input_emotion = st.text_input("Enter text for emotion analysis:")
            
            # Prediction for sentiment analysis for a single sentence
            if st.button("Predict emotion"):
                if user_input_emotion:
                    predicted_emotion = predict_emotion([user_input_emotion])
                    st.success(f"The predicted emotion is: {predicted_emotion}")
                else:
                    st.warning("Please enter a sentence for emotion analysis.")

        elif nav_option == "Batch Analysis":
            st.header("Emotion Analysis - Batch Analysis")
            st.markdown("Upload a CSV file with a single column of sentences, and I'll predict the emotion for each sentence.")

            # File upload for batch analysis
            uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

            if uploaded_file is not None:
                # Read CSV file
                sentences_df = pd.read_csv(uploaded_file)

                # Display uploaded data
                st.subheader("Uploaded Sentences:")
                st.write(sentences_df)

                # Predict sentiment for each sentence
                # Predict sentiment for each sentence using a for loop
                for index, row in sentences_df.iterrows():
                    sentences_df.at[index, 'Predicted Emotion'] = predict_emotion([row['Text']])

                # Display predicted sentiment
                st.subheader("Predicted Emotion:")
                st.write(sentences_df[['Text', 'Predicted Emotion']])

                # Save results to a new CSV file
                save_button = st.button("Save Results to CSV")
                if save_button:
                    result_file_path = "EmotionResults.csv"
                    sentences_df.to_csv(result_file_path, index=False)
                    st.success(f"Results saved to {result_file_path}")


    elif page == "Sarcasm Detection":

        # Navigation bar at the top
        nav_option = st.radio("Choose Analysis Type:", ["Single Sentence", "Batch Analysis"])

        if nav_option == "Single Sentence":
            st.header("Sarcasm Detection - Single Sentence")
            st.markdown("Enter a sentence, and I'll predict if it's sarcastic or not.")

            # User input for single sentence analysis
            user_input_sarcasm = st.text_input("Enter text for sarcasm detection:")

            # Prediction for sarcasm detection for a single sentence
            if st.button("Predict Sarcasm"):
                if user_input_sarcasm:
                    prediction_sarcasm = predict_sarcasm(user_input_sarcasm)
                    st.success(f"The prediction for sarcasm is: {prediction_sarcasm}")
                else:
                    st.warning("Please enter a sentence for sarcasm detection.")

        elif nav_option == "Batch Analysis":
            st.header("Sarcasm Analysis - Batch Analysis")
            st.markdown("Upload a CSV file with a single column of sentences, and I'll predict the sarcasm for each sentence.")

            # File upload for batch analysis
            uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

            if uploaded_file is not None:
                # Read CSV file
                sentences_df = pd.read_csv(uploaded_file)
                sentences_df=sentences_df.head(10)


                # Display uploaded data
                st.subheader("Uploaded Sentences:")
                st.write(sentences_df)

                # Predict sentiment for each sentence
                # Predict sentiment for each sentence using a for loop
                for index, row in sentences_df.iterrows():
                    sentences_df.at[index, 'sarcasm or not'] = predict_sarcasm(row['headline'])

                # Display predicted sentiment
                st.subheader("Predicted Sarcasm:")
                st.write(sentences_df[['headline', 'sarcasm or not']])

                # Save results to a new CSV file
                save_button = st.button("Save Results to CSV")
                if save_button:
                    result_file_path = "SarcasmResults.csv"
                    sentences_df.to_csv(result_file_path, index=False)
                    st.success(f"Results saved to {result_file_path}")



if __name__ == '__main__':
    main()