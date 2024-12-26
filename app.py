import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import joblib
import os

def main():
    st.title("Sentiment Analysis of Play Store Reviews for BCA Mobile App")

    # Create two tabs: "Input Teks" and "Upload File"
    tab1, tab2 = st.tabs(["Input Teks", "Upload File"])
    
    # Tab 1: Input Teks
    with tab1:
        st.subheader("Input Text for Sentiment Analysis")
        
        # Input text box
        text_input = st.text_area("Enter your review text:")
        
        # Button to trigger sentiment analysis
        if st.button("Analyze Sentiment"):
            if text_input.strip():  # Check if text input is not empty
                result = predict_sentiment(text_input)  # Predict sentiment using a function
                # Display sentiment result with styling
                if result == 'Positive':
                    st.success(f"Hasil Sentimen: **{result}**")
                else:
                    st.error(f"Hasil Sentimen: **{result}**")
            else:
                st.warning("Harap masukkan teks terlebih dahulu!")

    # Tab 2: Upload File
    with tab2:
        st.subheader("Upload CSV/XLSX File")
        uploaded_file = st.file_uploader("Choose a CSV/XLSX file", type=["csv", "xlsx"])
        
        if uploaded_file is not None:
            # Save the uploaded file to the 'dataset' folder
            dataset_folder = "dataset"
            if not os.path.exists(dataset_folder):
                os.makedirs(dataset_folder)
            
            file_path = os.path.join(dataset_folder, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Read the uploaded file based on its extension
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            
            # Display a success message and preview the data
            st.write("File successfully uploaded and saved. Here's a preview of the data:")
            st.dataframe(df, use_container_width=True)
            
            # Display all other sections once the file is uploaded
            show_sections(df)

def predict_sentiment(text_input):
    """Predict sentiment for the given text input."""
    # Load pre-trained model and vectorizer
    try:
        model = joblib.load('model/naive_bayes_model.pkl')
        vectorizer = joblib.load('model/tfidf_vectorizer.pkl')
    except FileNotFoundError:
        st.error("Pre-trained model or vectorizer not found. Please upload the necessary files.")
        return ""
    
    # Vectorize the input text using the same vectorizer as used during training
    X = vectorizer.transform([text_input])
    
    # Predict sentiment using the Random Forest model
    prediction = model.predict(X)
    return 'Positive' if prediction == 0 else 'Negative'

def show_sections(df):

    # Section: Sentiment Analysis
    with st.expander("Sentiment Analysis"):
        st.write("This section performs sentiment analysis using Random Forest with a 90:10 Train-Test Split.")
        
        try:
            model = joblib.load('model/naive_bayes_model.pkl')
            vectorizer = joblib.load('model/tfidf_vectorizer.pkl')
        except FileNotFoundError:
            st.error("Pre-trained model or vectorizer not found. Please upload the necessary files.")
            return
        
        # Preprocess the data
        if 'content' in df.columns:
            # Vectorize the content using the same vectorizer as used during training
            X = vectorizer.transform(df['content'])
            
            # Predict sentiment using the Random Forest model
            predictions = model.predict(X)
            
            # Add predictions to the dataframe
            df['predicted_label'] = predictions
            df['predicted_label'] = df['predicted_label'].map({0: 'Positive', 1: 'Negative'})
            
            st.write("Sentiment analysis completed. Here's the data with predicted sentiment:")
            
            # Function to apply color styling
            def color_sentiment(val):
                bg_color, text_color = style_sentiment(val)
                return f'background-color: {bg_color}; color: {text_color}; font-size: 14px;'

            # Apply styling to the 'sentiment' column
            styled_df = df[['content', 'predicted_label']].rename(columns={'content': 'text', 'predicted_label': 'sentiment'})
            styled_df = styled_df.style.applymap(color_sentiment, subset=['sentiment'])
            
            # Display the styled dataframe
            st.dataframe(styled_df)
        else:
            st.error("The dataset does not contain the required 'content' column for prediction.")

    # Section: Dropdown for Visualizations
    with st.expander("Visualizations"):
        st.write("This section contains visualizations of the label distribution and WordCloud for each sentiment.")
        
        # Label Distribution Bar Chart
        st.subheader("Distribusi Label pada Dataset")
        label_counts = df['predicted_label'].value_counts()

        # Determine colors for Positive (green) and Negative (red)
        colors = ['green' if label == 'Positive' else 'red' for label in label_counts.index]

        # Create bar chart
        plt.figure(figsize=(8, 4))
        ax = label_counts.plot(kind='bar', color=colors, edgecolor='black')

        # Add labels and title
        plt.title("Distribusi Label pada Dataset", fontsize=14)
        plt.xlabel("Label", fontsize=12)
        plt.ylabel("Jumlah", fontsize=12)

        # Add counts on top of each bar
        for i, v in enumerate(label_counts):
            ax.text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=12)

        # Show bar chart
        plt.xticks(rotation=0)
        st.pyplot(plt)

        # WordCloud for Positive Labels
        st.subheader("WordCloud for Positive Labels")
        positive_text = ' '.join(df[df['predicted_label'] == 'Positive']['content'])
        wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_positive, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

        # WordCloud for Negative Labels
        st.subheader("WordCloud for Negative Labels")
        negative_text = ' '.join(df[df['predicted_label'] == 'Negative']['content'])
        wordcloud_negative = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(negative_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_negative, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

def style_sentiment(sentiment):
    if sentiment == 'Positive':
        return '#d4edda', '#155724'
    else:
        return '#f8d7da', '#721c24'

# Run the app
if __name__ == "__main__":
    main()
