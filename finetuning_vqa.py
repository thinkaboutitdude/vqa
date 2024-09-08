import os
import requests
import torch
import pickle
import pyrallis
import wandb
import pandas as pd

from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from dataclasses import dataclass
from set_seed import set_random_seed

@dataclass
class Config:
    wandb_project: str = "vlip_vqa_test"
    model: str = "Salesforce/blip-vqa-base"
    processor: str = "Salesforce/blip-vqa-base"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    train_seed: int = 1
    lr: float = 4e-5
    grad_clip: int = 1
    scheduler_gamma: int = 0.9
    batch_size: int = 4
    num_epochs: int = 10
    patience: int = 5
    seed: int = 1


class VQADataset(Dataset):
    def __init__(self, csv_dataset_path, images_path, processor, mode="train"):
        self.csv_dataset_path = pd.read_csv(csv_dataset_path)
        self.csv_dataset_path = self.csv_dataset_path[pd.notnull(self.csv_dataset_path['answer'])]
        self.images_path = images_path
        self.processor = processor
        self.mode = mode

    def __len__(self):
        self.total_len = len(self.csv_dataset_path)
        self.fraction = 0.6
        if self.mode == "train":
            return int(self.fraction * self.total_len)
        else:
            return int(((1 - self.fraction) / 2) * self.total_len)

    def __getitem__(self, idx):
        # get image + text
        if self.mode == "val":
            idx = idx + int(self.total_len * self.fraction)
        elif self.mode == "test":
            idx = idx + int(self.total_len * self.fraction + (1 - self.fraction) // 2)
        question = self.csv_dataset_path.iloc[idx]["question"]
        answer = self.csv_dataset_path.iloc[idx]["answer"]
        image_id_str = str(self.csv_dataset_path.iloc[idx]["image_id"])
        converted_image_id = ["0"] * 12
        converted_image_id[-len(image_id_str):] = image_id_str
        image_path = f"{self.images_path}/COCO_val2014_{''.join(converted_image_id)}.jpg"
        image = Image.open(image_path).convert("RGB")
        text = question
        
        encoding = self.processor(image, text, padding="max_length", truncation=True, return_tensors="pt")
        labels = self.processor.tokenizer.encode(
            answer, max_length=8, pad_to_max_length=True, return_tensors="pt"
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


    train_dataset = VQADataset(csv_dataset_path="./Train/Train_Qs.csv", images_path="./Train/images", processor=processor, mode="train")
    valid_dataset = VQADataset(csv_dataset_path="./Train/Train_Qs.csv", images_path="./Train/images", processor=processor, mode="val")

    batch_size = config.batch_size

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.scheduler_gamma)

    num_epochs = config.num_epochs
    patience = config.patience
    min_eval_loss = float("inf")
    early_stopping_hook = 0

    for epoch in tqdm(range(num_epochs)):
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
        with torch.no_grad():
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
            torch.save({"model_state_dict": model.state_dict(), 
                        "optimizer_state_dict": optimizer.state_dict(), 
                        "loss": eval_loss}, "model_vqa.pt")
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