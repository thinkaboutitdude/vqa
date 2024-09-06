import os
import requests
import torch
import pickle
import pyrallis
import wandb
import pandas as pd

from transformers import BlipProcessor, BlipForQuestionAnswering
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from dataclasses import dataclass
from set_seed import set_random_seed

@dataclass
class Config:
    wandb_project: str = "vlip_vqa"
    model: str = "Salesforce/blip-vqa-base"
    processor: str = "Salesforce/blip-vqa-base"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    train_seed: int = 1
    lr: float = 4e-5
    grad_clip: int = 1
    scheduler_gamma: int = 0.9
    batch_size: int = 32
    num_epochs: int = 100
    patience: int = 10
    seed: int = 1


class VQADataset(Dataset):
    def __init__(self, csv_dataset_path, images_path, processor, mode="train"):
        self.csv_dataset_path = pd.read_csv(csv_dataset_path)
        self.images_path = images_path
        self.processor = processor
        self.mode = mode

    def __len__(self):
        self.total_len = len(self.csv_dataset_path)
        self.fraction = 0.8
        return int(self.fraction * self.total_len) if self.mode == "train" else int((1 - self.fraction) * self.total_len)

    def __getitem__(self, idx):
        # get image + text
        idx = idx if self.mode == "train" else idx + int(self.total_len * self.fraction)
        question = self.csv_dataset_path.loc[idx]["question"]
        answer = self.csv_dataset_path.loc[idx]["answer"]
        image_id_str = str(self.csv_dataset_path.loc[idx]["image_id"])
        converted_image_id = ["0"] * 12
        converted_image_id[-len(image_id_str):] = image_id_str
        image_path = f"{self.images_path}/COCO_val2014_{converted_image_id}.jpg"
        image = Image.open(image_path).convert("RGB")
        text = question
        
        encoding = self.processor(image, text, padding="max_length", truncation=True, return_tensors="pt")
        labels = self.processor.tokenizer.encode(
            answer, max_length=16, pad_to_max_length=True, return_tensors="pt"
        )
        encoding["labels"] = labels
        # remove batch dimension
        for k,v in encoding.items():  encoding[k] = v.squeeze()
        return encoding

@pyrallis.wrap()
def train(config: Config):
    wandb.init(project=config.wandb_project)

    model = BlipForQuestionAnswering.from_pretrained(config.model)
    processor = BlipProcessor.from_pretrained(config.processor)

    device = config.device
    model.to(device)

    torch.cuda.empty_cache()
    set_random_seed(config.seed)

    # training_dataset = load_dataset("csv", data_files="Data/train.jsonl", split="train[:90%]")
    # valid_dataset = load_dataset("csv", data_files="Data/train.jsonl", split="train[90%:]")
    # print("Training sets: {} - Validating set: {}".format(len(training_dataset), len(valid_dataset)))

    train_dataset = VQADataset(csv_dataset_path="./Train/Train_Qs.csv", images_path="./Train/images", processor=processor)
    valid_dataset = VQADataset(csv_dataset_path="./Train/Train_Qs.csv", images_path="./Train/images", processor=processor)

    batch_size = config.batch_size
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.scheduler_gamma)

    num_epochs = config.num_epochs
    patience = config.patience
    min_eval_loss = float("inf")
    early_stopping_hook = 0
    accuracy = 0

    for epoch in range(num_epochs):
        train_loss = 0
        model.train()
        for idx, batch in zip(tqdm(range(len(train_dataloader)), desc="Training batch: ..."), train_dataloader):
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").to(device)
            attention_masked = batch.pop("attention_mask").to(device)
            labels = batch.pop("labels").to(device)
            
            with torch.amp.autocast(device_type=config.device, dtype=torch.float16):
                outputs = model(input_ids=input_ids,
                            pixel_values=pixel_values,
                            # attention_mask=attention_masked,
                            labels=labels)
                
            loss = outputs.loss
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
            optimizer.step()
        wandb.log({"train_loss": train_loss})
        
        model.eval()
        eval_loss = 0
        for idx, batch in zip(tqdm(range(len(valid_dataloader)), desc="Validating batch: ..."), valid_dataloader):
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").to(device)
            attention_masked = batch.pop("attention_mask").to(device)
            labels = batch.pop("labels").to(device)

            with torch.amp.autocast(device_type=config.device, dtype=torch.float16):
                outputs = model(input_ids=input_ids,
                            pixel_values=pixel_values,
                            attention_mask=attention_masked,
                            labels=labels)
            
            loss = outputs.loss
            eval_loss += loss.item()
        wandb.log({"eval_loss": eval_loss})

        scheduler.step()
        if eval_loss < min_eval_loss:
            model.save_pretrained("vqa-saved-model", from_pt=True) 
            print("Saved model checkpoint")
            min_eval_loss = eval_loss
            early_stopping_hook = 0
        else:
            early_stopping_hook += 1
            if early_stopping_hook > patience:
                break
    
# pickle.dump(tracking_information, open("tracking_information.pkl", "wb"))
if __name__ == "__main__":
    train()
    print("The finetuning process has done!")