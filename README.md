# 🥣🔤 Needs More Words - Optimize Your Content

**Updated to final version 0.3**

Let's you know what topics and words you need to include in your content and suggests how long it should be, based on top 20 results from Google.

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
3. Run with Docker:

https://docs.streamlit.io/deploy/tutorials/docker


4. Download the GloVe embedding file and put it in the same folder as the main app:
   ```
      # Download the GloVe zip file
   curl -O http://nlp.stanford.edu/data/glove.6B.zip
   
   # Unzip the downloaded file
   unzip glove.6B.zip
   
   # Remove the zip file
   rm glove.6B.zip
   ```
