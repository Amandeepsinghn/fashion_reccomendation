import streamlit as st
import numpy as np
import PIL.Image
import requests
import google.generativeai as genai
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import chromadb
from dotenv import load_dotenv
import os
import warnings
from langchain_community.llms import OpenAI
import openai

# Load environment variables
load_dotenv()
api_key = os.getenv("OPEN_AI_KEY")
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module='streamlit')

# Initialize the necessary components
db_folder_path = "Vector_database"
chroma_client = chromadb.PersistentClient(path=db_folder_path)
clip = OpenCLIPEmbeddingFunction()

answer = []

# Define the prompt
Prompt = """
You are an expert fashion recommendation system. 
Your task is to classify the clothing item in the image and provide a set of 2-4 fashion products that would match well with the classified item. 
Please ensure that the recommendations are relevant and stylish. 
The clothing items in the image may include tops, pants, dresses, shoes, or accessories.

Return only the names of the recommended fashion products in a list format, with each item separated by a comma.
Example:
If the user uploads an image of a red top, you should only return something like this:
"blue jeans, white shoes, black jacket"

The recommendations should be in this format only:
- "red clutch"
- "wide-brim hat"
- "metallic heels"
- "white shoes"
"""

# Streamlit UI setup
st.title("Fashion Recommendation System")
st.markdown("Upload an image of a clothing item, and I'll classify it and suggest matching products!")

# Upload Image Section
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = PIL.Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Process the image and extract relevant information
    if st.button("Get Fashion Recommendations"):
        with st.spinner('Classifying the clothing item and finding matching products...'):
            # Initialize the OpenAI model using LangChain
            openai_model = OpenAI(
                openai_api_key=api_key,  # Your OpenAI API key
                temperature=0.7,  # Adjust creativity
                max_tokens=150  # Set max tokens for the response
            )

            # Call OpenAI to get the response based on the prompt
            response = openai_model(Prompt)
            
            # Check the response and extract product recommendations
            print("OpenAI Response:", response)  # Debugging the response
            recommendations = [item.strip().replace('"', '') for item in response.split('\n') if item.strip()]
            if not recommendations:
                st.error("No recommendations found.")
                st.stop()

            print("Recommendations:", recommendations)

            # Query the database for matching products
            for query in recommendations:
                query_embedding = clip([query])[0]  # Get the embedding for the recommendation

                image_vdb = chroma_client.get_collection(name="product_embedding")
                result = image_vdb.query(query_embeddings=[query_embedding], n_results=3)

                ids = result.get("ids", [])
                metadata = result.get("metadatas", [])
                if ids:
                    random_id = np.random.choice(ids[0]) 
                    index = ids[0].index(random_id) # This will be a list of lists
                      # This will be a list of lists
                    selected_metadata = metadata[0][index]

                    if isinstance(selected_metadata, dict):
                        print({selected_metadata.get('uri', 'URI not found')})
                        answer.append(selected_metadata.get('uri', 'URI not found'))
                    else:
                        print(f"ID: {random_id}, Metadata format unexpected.")

                else:
                    answer.append("No matching product found")

            # Display the classification and recommendations
            st.subheader("Recommended Products:")
            if answer:
                for image_url in answer:
                    st.image(image_url,caption="Reccommended Product Image",use_column_width=True)
                st.write(answer)
            else:
                st.write("No recommendations available.")
