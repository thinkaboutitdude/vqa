# import shutil

# shutil.unpack_archive(filename='/mnt/c/Users/nevazhno/Downloads/Train.zip', extract_dir='.')

# import os
# import requests
# import torch
# import pickle
# import pyrallis
# import wandb
# import pandas as pd

# from transformers import BlipProcessor, BlipForQuestionAnswering
# from datasets import load_dataset
# from PIL import Image
# from torch.utils.data import DataLoader, Dataset
# from tqdm import tqdm
# from dataclasses import dataclass
# from set_seed import set_random_seed

# @dataclass
# class Config:
#     wandb_project: str = "vlip_vqa"
#     model: str = "Salesforce/blip-vqa-base"
#     processor: str = "Salesforce/blip-vqa-base"
#     device: str = "cuda" if torch.cuda.is_available() else "cpu"
#     train_seed: int = 1
#     lr: float = 4e-5
#     scheduler_gamma: int = 0.9
#     batch_size: int = 32
#     num_epochs: int = 100
#     patience: int = 10
#     seed: int = 1


# class VQADataset(Dataset):
#     def __init__(self, csv_dataset_path, images_path, processor):
#         self.csv_dataset_path = pd.read_csv(csv_dataset_path)
#         self.images_path = images_path
#         self.processor = processor

#     def __len__(self):
#         return len(self.csv_dataset_path)

#     def __getitem__(self, idx):
#         # get image + text
#         question = self.csv_dataset_path.loc[idx]["question"]
#         answer = self.csv_dataset_path.loc[idx]["answer"]
#         image_id_str = str(self.csv_dataset_path.loc[idx]["image_id"])
#         converted_image_id = ["0"] * 12
#         converted_image_id[-len(image_id_str):] = image_id_str
#         image_path = f"{self.images_path}/COCO_val2014_{''.join(converted_image_id)}.jpg"
#         image = Image.open(image_path).convert("RGB")
#         text = question
        
#         encoding = self.processor(image, text, padding="max_length", truncation=True, return_tensors="pt")
#         labels = self.processor.tokenizer.encode(
#             answer, max_length=16, pad_to_max_length=True, return_tensors='pt'
#         )
#         encoding["labels"] = labels
#         # remove batch dimension
#         for k,v in encoding.items():  encoding[k] = v.squeeze()
#         return encoding

# config = Config()    
# processor = BlipProcessor.from_pretrained(config.processor)

# data = VQADataset(csv_dataset_path='./Train/Train_Qs.csv', images_path='./Train/images', processor=processor)
# a = data[0]
# print(])
# a.pop('pixel_values')
# print(a)


import pandas as pd

data = pd.read_csv('./Train/Train_Qs.csv')
print(len(data))
print(data.loc[36136])
new_data = pd.read_csv('./Train/Train_Qs.csv')[data['answer'] != 'none'].dropna()
print(len(new_data))