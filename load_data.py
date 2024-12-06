from datasets import load_dataset 
import os 

dataset = load_dataset("detection-datasets/fashionpedia")

dataset_folder="dataset"
os.makedirs(dataset_folder,exist_ok=True)


def save_image(dataset,dataset_folder,num_image=1000):
    for i in range(num_image):
        image=dataset["train"][i]["image"]

        image.save(os.path.join(dataset_folder,f'image_{i+1}.png'))
    
save_image(dataset,dataset_folder,num_image=1000)

print(f"Saved the first 1000 images to {dataset_folder}")
