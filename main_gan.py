"""GAN-based SNR enhancement main script
    Details
    -------
    Name: main_gan.py
    Authors: [Your Name]
    Created: [Today's Date]

    Purpose
    --------
    This script trains a GAN model to enhance SNR from -10dB to 10dB.
"""
import sys
import torch
import os
import copy
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from datetime import datetime
from src.system_model import SystemModelParams
from src.data_handler import *
from src.models import GAN_Model
from src.training import TrainingParams, simulation_summary
from torch.utils.data import DataLoader, TensorDataset
# 初始化
warnings.simplefilter("ignore")
os.system("cls||clear")
plt.close("all")

if __name__ == "__main__":
    # 初始化路径
    external_data_path = Path.cwd() / "data"
    simulations_path = external_data_path / "simulations"
    saving_path = external_data_path / "weights"
    simulations_path.mkdir(parents=True, exist_ok=True)
    saving_path.mkdir(parents=True, exist_ok=True)
    
    # 时间戳
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    dt_string_for_save = now.strftime("%d_%m_%Y_%H_%M")
    
    # 命令设置
    commands = {
        "SAVE_TO_FILE": True,
        "CREATE_DATA": True,
        "TRAIN_MODEL": True,
        "SAVE_MODEL": True,
    }

    # 系统参数
    base_params = (  
        SystemModelParams()
        .set_parameter("N", 32)
        .set_parameter("M", 2)
        .set_parameter("T", 1000)
        .set_parameter("grid_size", 241)  # 添加网格点参数
        .set_parameter("signal_type", "NarrowBand")
        .set_parameter("signal_nature", "non-coherent")
        .set_parameter("eta", 0)
        .set_parameter("bias", 0)
        .set_parameter("sv_noise_var", 0)
        .set_parameter("gap", 10)
    )

    # 数据集参数
    samples_size = 100
    snr_values = [-10]  # 输入SNR
    target_snr = 10     # 目标SNR

    # 日志文件设置
    if commands["SAVE_TO_FILE"]:
        file_path = (
            simulations_path / "results" / "train_scores" / Path(
                f"gan_train_{dt_string_for_save}.txt")
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)
        sys.stdout = open(file_path, "w")

    # 打印训练信息
    print("---------- GAN Training ----------")
    print("date and time =", dt_string)
    print(f"Input SNR: {snr_values}, Target SNR: {target_snr}")

    # 创建数据集
    if commands["CREATE_DATA"]:
        print("Creating Data...")
        # 创建低SNR训练数据
        low_snr_data = []
        for snr in snr_values:
            params = copy.deepcopy(base_params).set_parameter("snr", snr)
            data, _, _ = create_dataset(
                system_model_params=params,
                samples_size=samples_size,
                model_type="GAN",
                phase="train"
            )
            # low_snr_data.extend(data.reshape(-1, 3, 32, 32))

        # 创建高SNR目标数据
        high_snr_params = copy.deepcopy(base_params).set_parameter("snr", target_snr)
        high_snr_data, _, _ = create_dataset(
            system_model_params=high_snr_params,
            samples_size=samples_size,
            model_type="GAN",
            phase="train"
        )
        # 准备数据加载器
        dataset = TensorDataset(torch.tensor(low_snr_data), torch.tensor(high_snr_data))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 训练设置
    if commands["TRAIN_MODEL"]:
        # 初始化模型
        model = GAN_Model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # 训练参数
        simulation_parameters = (
            TrainingParams()
            .set_batch_size(64)
            .set_epochs(100)
            .set_model(model=model)
            .set_optimizer(optimizer="Adam", learning_rate=0.0002)
            .set_criterion()
        )
        # 准备数据加载器
        dataset = TensorDataset(torch.tensor(low_snr_data), torch.tensor(high_snr_data))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


        # 打印训练摘要
        simulation_summary(
            system_model_params=base_params,
            model_type="GAN",
            parameters=simulation_parameters,
            phase="training",
        )

        # 训练循环 (原有训练代码保持不变)
        num_epochs = 100
        for epoch in range(num_epochs):
            for i, (low_snr, high_snr) in enumerate(dataloader):
                real_labels = torch.ones(low_snr.size(0), 1).to(device)
                fake_labels = torch.zeros(low_snr.size(0), 1).to(device)
                
                # 训练判别器
                d_optimizer.zero_grad()
                
                # 真实数据损失
                real_outputs = discriminator(high_snr.to(device).view(-1, 3, 32, 32))
                d_loss_real = criterion(real_outputs, real_labels)
                # 生成数据损失 
                fake_data = generator(low_snr.to(device))
                fake_outputs = discriminator(fake_data)
                d_loss_fake = criterion(fake_outputs, fake_labels)
                
                # 判别器总损失
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                d_optimizer.step()
                
                # 训练生成器
                g_optimizer.zero_grad()
                fake_outputs = discriminator(fake_data)
                g_loss = criterion(fake_outputs, real_labels)
                g_loss.backward()
                g_optimizer.step()
                
            print(f"Epoch [{epoch+1}/{num_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")


        # 保存模型
        if commands["SAVE_MODEL"]:
            torch.save(
                model.state_dict(),
                saving_path / "final_models" / f"gan_model_{dt_string_for_save}.pth"
            )

    print("Training completed.")