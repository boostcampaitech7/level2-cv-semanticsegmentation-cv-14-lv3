# trainer.py

import os
import time
import datetime
import random
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
import wandb
from dataset import CLASSES
from functions import dice_coef

def convert_seconds_to_hms(seconds):
    """초를 시, 분, 초로 변환하는 함수"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

def train(model, data_loader, val_loader, criterion, optimizer, num_epochs, val_interval, save_dir):
    print('Start training..')
    
    best_dice = 0.
    total_start_time = time.time()  # 총 학습 시작 시간 추가
    best_model_path = None  # 최적 모델 파일 경로를 추적하기 위한 변수
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()  # 에포크 시작 시간
        
        model.train()
        
        with tqdm(total=len(data_loader), desc=f"Epoch [{epoch+1}/{num_epochs}]") as pbar:
            for step, (images, masks) in enumerate(data_loader):
                images, masks = images.cuda(), masks.cuda()
                model = model.cuda()
                
                outputs = model(images)['out']
                loss = criterion(outputs, masks)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                pbar.set_postfix(loss=round(loss.item(), 4))
                pbar.update(1)
        
        # 에포크 시간 계산
        epoch_time = time.time() - epoch_start_time
        remaining_epochs = num_epochs - (epoch + 1)
        estimated_remaining_time = epoch_time * remaining_epochs

        # 시, 분, 초로 변환
        epoch_str = convert_seconds_to_hms(epoch_time)
        remaining_str = convert_seconds_to_hms(estimated_remaining_time)

        print(
            f'Epoch {epoch+1} completed in {epoch_str}. '
            f'Estimated remaining time: {remaining_str}.'
        )
        
        # Wandb에 로그 기록
        wandb.log(
            {
            "train_loss": loss.item()
            }
            )
        
        if (epoch + 1) % val_interval == 0:
            dice = validation(epoch + 1, model, val_loader, criterion)
            
            wandb.log(
                {
                "val_dice": dice
                }
                )
            
            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in checkpoint")
                best_dice = dice

                # 기존 최적 모델 파일이 있다면 삭제
                if best_model_path and os.path.exists(best_model_path):
                    os.remove(best_model_path)
                
                best_model_path = os.path.join(save_dir, f"best_dice_{best_dice:.4f}.pt")
                save_model(model, best_model_path)
    
    total_time = time.time() - total_start_time
    total_str = convert_seconds_to_hms(total_time)
    print(f'Total training completed in {total_str}.')

    wandb.finish()

def validation(epoch, model, data_loader, criterion, thr=0.5):
    print(f'Start validation #{epoch:2d}')
    model.eval()

    dices = []
    with torch.no_grad():
        n_class = len(CLASSES)
        total_loss = 0
        cnt = 0

        for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images, masks = images.cuda(), masks.cuda()         
            model = model.cuda()
            
            outputs = model(images)['out']
            
            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)
            
            # gt와 prediction의 크기가 다른 경우 prediction을 gt에 맞춰 interpolation 합니다.
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
            
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu()
            masks = masks.detach().cpu()
            
            dice = dice_coef(outputs, masks)
            dices.append(dice)
                
    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    dice_str = [
        f"{c:<12}: {d.item():.4f}"
        for c, d in zip(CLASSES, dices_per_class)
    ]
    dice_str = "\n".join(dice_str)
    print(dice_str)
    
    avg_dice = torch.mean(dices_per_class).item()
    
    return avg_dice

def save_model(model, model_path):
    torch.save(model, model_path)

def set_seed(seed=21):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
