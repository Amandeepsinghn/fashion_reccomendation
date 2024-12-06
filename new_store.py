import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import os

# Database path
db_folder_path = "Vector_database"

# Initialize the persistent chromadb client
chroma_client = chromadb.PersistentClient(path=db_folder_path)

# Initialize the OpenCLIP embedding function
clip = OpenCLIPEmbeddingFunction()

# Path to your image dataset folder
dataset_folder = "Data"  # Update to the actual folder where your images are stored

# Initialize the image loader
image_loader = ImageLoader()

# Prepare image IDs, URIs, and metadata
ids = []
uris = []
metadatas = []

for i, filename in enumerate(sorted(os.listdir(dataset_folder))):
    if filename.endswith(".png"):
        file_path = os.path.join(dataset_folder, filename)
        ids.append(str(i))
        uris.append(file_path)
        # Adding metadata as a dictionary (can include any additional information)
        metadatas.append({'uri': file_path})  # Store the file path as metadata

# Break into smaller batches (e.g., 166 items per batch)
batch_size = 166

# Create or get the collection once outside the loop to avoid re-initializing it in every batch
image_vdb = chroma_client.get_or_create_collection(
    name="product_embedding",
    embedding_function=clip,
    data_loader=image_loader
)

for i in range(0, len(ids), batch_size):
    batch_ids = ids[i:i + batch_size]
    batch_uris = uris[i:i + batch_size]
    batch_metadatas = metadatas[i:i + batch_size]  # Use corresponding metadata for the batch
    
    # Add data in smaller batches to avoid the batch size limit
    image_vdb.add(
        ids=batch_ids,
        uris=batch_uris,
        metadatas=batch_metadatas  # Pass metadata along with each image
    )

    print(f"Batch {i // batch_size + 1} added successfully.")

