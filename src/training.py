"""
Subspace-Net

Details
----------
Name: training.py
Authors: D. H. Shmuel
Created: 01/10/21
Edited: 17/03/23

Purpose
----------
This code provides functions for training and simulating the Subspace-Net model.

Classes:
----------
- TrainingParams: A class that encapsulates the training parameters for the model.

Methods:
----------
- train: Function for training the model.
- train_model: Function for performing the training process.
- plot_learning_curve: Function for plotting the learning curve.
- simulation_summary: Function for printing a summary of the simulation parameters.

Attributes:
----------
None
"""

# Imports
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import time
import copy
from pathlib import Path
import torch.optim as optim
from datetime import datetime
from torch.autograd import Variable
from tqdm import tqdm
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
from src.data_handler import config
from src.model_transform import My_transform_Model
from src.utils import *
from src.criterions import *
from src.system_model import SystemModel, SystemModelParams
from src.models import SubspaceNet, DeepCNN, DeepAugmentedMUSIC, ModelGenerator,GAN_Model
from src.evaluation import evaluate_dnn_model, evaluate_transformer_model


class TrainingParams(object):
    """
    A class that encapsulates the training parameters for the model.

    Methods
    -------
    - __init__: Initializes the TrainingParams object.
    - set_batch_size: Sets the batch size for training.
    - set_epochs: Sets the number of epochs for training.
    - set_model: Sets the model for training.
    - load_model: Loads a pre-trained model.
    - set_optimizer: Sets the optimizer for training.
    - set_schedular: Sets the scheduler for learning rate decay.
    - set_criterion: Sets the loss criterion for training.
    - set_training_dataset: Sets the training dataset for training.

    Raises
    ------
    Exception: If the model type is not defined.
    Exception: If the optimizer type is not defined.
    """

    def __init__(self):
        """
        Initializes the TrainingParams object.
        """

    def set_batch_size(self, batch_size: int):
        """
        Sets the batch size for training.

        Args
        ----
        - batch_size (int): The batch size.

        Returns
        -------
        self
        """
        self.batch_size = batch_size
        return self

    def set_epochs(self, epochs: int):
        """
        Sets the number of epochs for training.

        Args
        ----
        - epochs (int): The number of epochs.

        Returns
        -------
        self
        """
        self.epochs = epochs
        return self

    # TODO: add option to get a Model instance also
    def set_model(
        self,
        system_model: SystemModel = None,
        tau: int = None,
        diff_method: str = "root_music",
        model_type: str = "SubspaceNet",
        model: ModelGenerator = None,
    ):
        """
        Sets the model for training.

        Args
        ----
        - system_model (SystemModel): The system model object.
        - tau (int, optional): The number of lags for auto-correlation (relevant only for SubspaceNet model).
        - diff_method (str): the differentiable subspace method used for training SubspaceNet model.

        Returns
        -------
        self

        Raises
        ------
        Exception: If the model type is not defined.
        """
        if model is None:
            self.model_type = model_type
            # Assign the desired model for training
            if self.model_type.startswith("DA-MUSIC"):
                model = DeepAugmentedMUSIC(
                    N=system_model.params.N,
                    T=system_model.params.T,
                    M=system_model.params.M,
                )
            elif self.model_type.startswith("DeepCNN"):
                model = DeepCNN(N=system_model.params.N, grid_size=361)
            elif self.model_type.startswith("GAN_Model"):
                model = GAN_Model(num_angles=system_model.params.grid_size)
            elif self.model_type.startswith("My_transform_Model"):
                model =My_transform_Model(num_classes=241)
            elif self.model_type.startswith("SubspaceNet"):
                if not isinstance(tau, int):
                    raise ValueError(
                        "TrainingParams.set_model: tau parameter must be provided for SubspaceNet model"
                    )
                self.tau = tau
                self.diff_method = diff_method
                model = SubspaceNet(
                    tau=tau, M=system_model.params.M, diff_method=diff_method
                )
            elif self.model_type.startswith("OffgridDOA"):
                self.model = OffgridDOA(num_classes=241, N=16)
            else:
                raise Exception(
                    f"TrainingParams.set_model: Model type {self.model_type} is not defined"
                )
        elif isinstance(model, ModelGenerator):
            self.model_type = model.model_type
            self.tau = model.tau
            self.diff_method = model.diff_method
            model = model.model
        else:
            raise Exception("TrainingParams.set_model: model is not supported")
        # assign model to device
        self.model = model.to(device)
        return self

    def load_model(self, loading_path: Path):
        """
        Loads a pre-trained model.

        Args
        ----
        - loading_path (Path): The path to the pre-trained model.

        Returns
        -------
        self
        """
        # Load model from given path
        state_dict = torch.load(loading_path, map_location=device)
        # Handle multi-GPU trained model being loaded into single-GPU environment
        if all(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        return self

    def set_optimizer(self, optimizer: str, learning_rate: float, weight_decay: float):
        """
        Sets the optimizer for training.

        Args
        ----
        - optimizer (str): The optimizer type.
        - learning_rate (float): The learning rate.
        - weight_decay (float): The weight decay value (L2 regularization).

        Returns
        -------
        self

        Raises
        ------
        Exception: If the optimizer type is not defined.
        """
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        # Assign optimizer for training
        if optimizer.startswith("Adam"):
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        elif optimizer.startswith("SGD"):
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        elif optimizer == "SGD Momentum":
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=learning_rate, momentum=0.9
            )
        else:
            raise Exception(
                f"TrainingParams.set_optimizer: Optimizer {optimizer} is not defined"
            )
        return self

    def set_schedular(self, step_size: float, gamma: float):
        """
        Sets the scheduler for learning rate decay.

        Args:
        ----------
        - step_size (float): Number of steps for learning rate decay iteration.
        - gamma (float): Learning rate decay value.

        Returns:
        ----------
        self
        """
        # Number of steps for learning rate decay iteration
        self.step_size = step_size
        # learning rate decay value
        self.gamma = gamma
        # Assign schedular for learning rate decay
        self.schedular = lr_scheduler.StepLR(
            self.optimizer, step_size=step_size, gamma=gamma
        )
        return self

    def set_criterion(self):
        """
        Sets the loss criterion for training.

        Returns
        -------
        self
        """
        # Define loss criterion
        if self.model_type.startswith(("My_transform_Model","DeepCNN","GAN_Model")):
            self.criterion = nn.BCELoss()
        # elif self.model_type.startswith("DeepCNN"):
        #     self.criterion = nn.BCELoss()
        else:
            self.criterion = RMSPELoss()
        return self

    def set_training_dataset(self, train_dataset: list):
        """
        Sets the training dataset for training.

        Args
        ----
        - train_dataset (list): The training dataset.

        Returns
        -------
        self
        """
        # Divide into training and validation datasets
        train_dataset, valid_dataset = train_test_split(
            train_dataset, test_size=0.05, shuffle=True
        )# 进行train / valid划分
        print("Training DataSet size", len(train_dataset))
        print("Validation DataSet size", len(valid_dataset))
        # Transform datasets into DataLoader objects
        self.train_dataset = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            drop_last=False
        )
        self.valid_dataset = torch.utils.data.DataLoader(
            valid_dataset, 
            batch_size=1,
            drop_last=False
        )
        return self


def train(
    system_model_params: SystemModelParams,
    training_parameters: TrainingParams,
    model_name: str,
    plot_curves: bool = True,
    saving_path: Path = None,
):
    """
    Wrapper function for training the model.

    Args:
    ----------
    - training_params (TrainingParams): An instance of TrainingParams containing the training parameters.
    - model_name (str): The name of the model.
    - plot_curves (bool): Flag to indicate whether to plot learning and validation loss curves. Defaults to True.
    - saving_path (Path): The directory to save the trained model.

    Returns:
    ----------
    model: The trained model.
    loss_train_list: List of training loss values.
    loss_valid_list: List of validation loss values.

    Raises:
    ----------
    Exception: If the model type is not defined.
    Exception: If the optimizer type is not defined.
    """
    # Set the seed for all available random operations
    set_unified_seed()
    # Current date and time
    print("\n----------------------\n")
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    dt_string_for_save = now.strftime("%d_%m_%Y_%H_%M")
    print("date and time =", dt_string)
    # Train the model
    model, loss_train_list, loss_valid_list = train_model(system_model_params,
        training_parameters, model_name=model_name, checkpoint_path=saving_path
    )
    # Save models best weights
    torch.save(model.state_dict(), saving_path / Path(dt_string_for_save))
    # Plot learning and validation loss curves
    if plot_curves:
        plot_learning_curve(
            list(range(training_parameters.epochs)), loss_train_list, loss_valid_list
        )
    return model, adv_loss_list, cls_loss_list, snr_loss_list
def train_gan(
    system_model_params: SystemModelParams,
    training_parameters: TrainingParams,
    model_name: str,
    plot_curves: bool = True,
    saving_path: Path = None,
):
    """
    Wrapper function for training the model.

    Args:
    ----------
    - training_params (TrainingParams): An instance of TrainingParams containing the training parameters.
    - model_name (str): The name of the model.
    - plot_curves (bool): Flag to indicate whether to plot learning and validation loss curves. Defaults to True.
    - saving_path (Path): The directory to save the trained model.

    Returns:
    ----------
    model: The trained model.
    loss_train_list: List of training loss values.
    loss_valid_list: List of validation loss values.

    Raises:
    ----------
    Exception: If the model type is not defined.
    Exception: If the optimizer type is not defined.
    """
    # Set the seed for all available random operations
    set_unified_seed()
    # Current date and time
    print("\n----------------------\n")
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    dt_string_for_save = now.strftime("%d_%m_%Y_%H_%M")
    print("date and time =", dt_string)
    # Train the model
    model, adv_loss_list, cls_loss_list, snr_loss_list = train_gan_model(system_model_params,
        training_parameters, model_name=model_name, checkpoint_path=saving_path
    )
    # Save models best weights
    torch.save(model.state_dict(), saving_path / Path(dt_string_for_save))
    # Plot learning and validation loss curves
    if plot_curves:
        plot_gan_learning_curve(adv_loss_list, cls_loss_list, snr_loss_list)
    return model, adv_loss_list, cls_loss_list, snr_loss_list

def train_gan_model(system_model_params: SystemModelParams, 
                 training_params: TrainingParams, 
                 model_name: str, 
                 checkpoint_path=None,
                 num_angles=180,
                 target_snr=10.0):
    """
    Function for training the GAN model.

    Args:
    -----
        training_params (TrainingParams): An instance of TrainingParams containing the training parameters.
        model_name (str): The name of the model.
        checkpoint_path (str): The path to save the checkpoint.

    Returns:
    --------
        model: The trained model.
        adv_loss_list (list): List of adversarial losses per batch.
        cls_loss_list (list): List of classification losses per batch.
        snr_loss_list (list): List of SNR losses per batch.
    """
    # Initialize model and optimizers
    model = training_params.model
    # Generator optimizer with lower learning rate (0.1x)
    gen_optimizer = optim.Adam(
        model.generator.parameters(), 
        lr=training_params.learning_rate * 0.1,
        weight_decay=training_params.weight_decay
    )
    # Discriminator optimizer with original learning rate
    disc_optimizer = training_params.optimizer
    # Initialize losses
    loss_train_list=[]
    adv_loss_list = []
    cls_loss_list = []
    snr_loss_list = []
    min_train_loss = np.inf
    # Set initial time for start training
    since = time.time()
    print("\n---Start Training Stage ---\n")
    # Run over all epochs
    for epoch in range(training_params.epochs):
        train_length = 0
        overall_train_loss = 0.0
        # Set model to train mode
        model.train()
        # 先检查GPU数量再决定设备分配

        model = model.to(device)  # 将模型移到主设备
        for data in tqdm(training_params.train_dataset):
            low_snr_list, high_snr_cov, angle_labels = data
            low_snr_cov1 = low_snr_list[0][0].to(device)
            low_snr_cov2 = low_snr_list[1][0].to(device)
            low_snr_cov3 = low_snr_list[2][0].to(device)
            low_snr_cov4 = low_snr_list[3][0].to(device)
            low_snr_cov5 = low_snr_list[4][0].to(device)
            high_snr_cov = high_snr_cov.to(device)
            angle_labels = angle_labels.to(device)
            train_length += angle_labels.shape[0]  # 使用angle_labels的形状代替未定义的DOA
            
            # 训练生成器5次，每次使用不同的低信噪比数据
            for low_snr_cov in [low_snr_cov1, low_snr_cov2, low_snr_cov3, low_snr_cov4, low_snr_cov5]:
                gen_high_snr = model.generator(low_snr_cov)
                # 生成器损失 - Wasserstein损失
                d_fake_valid, d_fake_angle, d_fake_snr = model.discriminator(gen_high_snr)
                gen_loss = -d_fake_valid.mean()  # Wasserstein损失
                
                # 角度分类损失
                cls_loss = nn.CrossEntropyLoss()(d_fake_angle, angle_labels)
                
                # SNR回归损失
                snr_loss = nn.MSELoss()(d_fake_snr, torch.full_like(d_fake_snr, target_snr))
                
                # 总损失
                total_loss = 0.5*gen_loss +50*cls_loss + 0.001*snr_loss
                total_loss.backward()
                gen_optimizer.step()
                model.generator.zero_grad()
            
            # 训练判别器1次
            gen_high_snr = model.generator(low_snr_cov1)  # 使用第一个低信噪比数据生成样本
            d_real_valid, d_real_angle, d_real_snr = model.discriminator(high_snr_cov)
            d_fake_valid, d_fake_angle, d_fake_snr = model.discriminator(gen_high_snr.detach())# 使用detach()来防止梯度传播到生成器
            
            # 计算三支路损失
            # Wasserstein损失
            adv_loss = d_fake_valid.mean() - d_real_valid.mean()
            
            # 梯度惩罚
            alpha = torch.rand(high_snr_cov.size(0), 1, 1, 1, device=device)
            interpolates = (alpha * high_snr_cov + (1 - alpha) * gen_high_snr).requires_grad_(True)
            d_interpolates, _, _ = model.discriminator(interpolates)
            gradients = torch.autograd.grad(
                outputs=d_interpolates,
                inputs=interpolates,
                grad_outputs=torch.ones_like(d_interpolates),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            gradients = gradients.view(gradients.size(0), -1)
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            
            # 添加梯度惩罚
            adv_loss += 8 * gradient_penalty
            
            # 角度分类损失（真实样本和生成样本）
            cls_loss_real = nn.CrossEntropyLoss()(d_real_angle, angle_labels)
            cls_loss_fake = nn.CrossEntropyLoss()(d_fake_angle, angle_labels)
            cls_loss = (cls_loss_real + cls_loss_fake) / 2
            
            # SNR回归损失（真实样本和生成样本）
            snr_loss_real = nn.MSELoss()(d_real_snr, torch.full_like(d_real_snr, target_snr))
            snr_loss_fake = nn.MSELoss()(d_fake_snr, torch.full_like(d_fake_snr, target_snr))
            snr_loss = (snr_loss_real + snr_loss_fake) / 2
            # Compute training loss
            train_loss = 0.5*adv_loss + 50*cls_loss + 0.001*snr_loss
            
            # Record individual losses
            adv_loss_list.append(adv_loss.item())
            cls_loss_list.append(cls_loss.item())
            snr_loss_list.append(snr_loss.item())
            # Back-propagation stage
            try:
                train_loss.backward()
            except RuntimeError:
                print("linalg error")
            # optimizer update
            disc_optimizer.step()
            # reset gradients
            model.discriminator.zero_grad()
            # add batch loss to overall epoch loss
            if training_params.model_type.startswith(("My_transform_Model", "DeepCNN","GAN_Model")):
                # BCE is averaged
                overall_train_loss += train_loss.item() * len(data[0])
            else:
                # RMSPE is summed
                overall_train_loss += train_loss.item()
        # Average the epoch training loss
        overall_train_loss = overall_train_loss / train_length
        loss_train_list.append(overall_train_loss)
        # Update schedular
        training_params.schedular.step()
        # Report results
        print(
            "epoch : {}/{}, Train loss = {:.6f}".format(
                epoch + 1, training_params.epochs, overall_train_loss
            )
        )
        print("Adv loss: {:.6f}, Cls loss: {:.6f}, SNR loss: {:.6f}".format(
            np.mean(adv_loss_list[-len(training_params.train_dataset):]),
            np.mean(cls_loss_list[-len(training_params.train_dataset):]),
            np.mean(snr_loss_list[-len(training_params.train_dataset):])
        ))
        print("lr {}".format(training_params.optimizer.param_groups[0]["lr"]))
        
        # Save best model weights based on train loss
        if overall_train_loss < min_train_loss:
            print(
                f"Train Loss Decreased({min_train_loss:.6f}--->{overall_train_loss:.6f}) \t Saving The Model"
            )
            min_train_loss = overall_train_loss
            best_epoch = epoch
            # Saving State Dict
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), checkpoint_path / model_name)

    time_elapsed = time.time() - since
    print("\n--- Training summary ---")
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), checkpoint_path / model_name)
    return model, adv_loss_list, cls_loss_list, snr_loss_list
def train_gan_model_pro(system_model_params: SystemModelParams, training_params: TrainingParams, model_name: str, checkpoint_path=None):
    """
    Function for training GAN model with WGAN-GP.

    Args:
    -----
        system_model_params: System model parameters
        training_params: Training parameters including model, optimizer, etc.
        model_name: Name of the model
        checkpoint_path: Path to save checkpoints

    Returns:
    --------
        model: Trained GAN model
        d_loss_list: List of discriminator losses
        g_loss_list: List of generator losses
    """
    # Initialize model and optimizers
    model = training_params.model
    # Separate optimizers for generator and discriminator
    d_optimizer = optim.Adam(
        model.module.discriminator.parameters(), 
        lr=training_params.learning_rate, 
        weight_decay=training_params.weight_decay
    )
    g_optimizer = optim.Adam(
        model.module.generator.parameters(), 
        lr=training_params.learning_rate * 0.1,  # Typically generator has lower learning rate
        weight_decay=training_params.weight_decay
    )
    
    # Initialize losses
    d_loss_list = []
    g_loss_list = []
    
    # Set main device to 5090D (device 0)
    torch.cuda.set_device(0)
    
    # Check available GPUs and setup parallel training
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training!")
        model = nn.DataParallel(model)
    
    # Set device
    model.to(device)
    
    # Gradient penalty coefficient
    lambda_gp = 10
    
    # Training loop
    since = time.time()
    print("\n---Start WGAN-GP Training Stage ---\n")
    
    for epoch in range(training_params.epochs):
        d_running_loss = 0.0
        g_running_loss = 0.0
        
        for i, (low_snr, high_snr) in enumerate(tqdm(training_params.train_dataset)):
            # Convert inputs to tensors if they are lists
            if isinstance(low_snr, list):
                low_snr = torch.stack(low_snr)
                high_snr = torch.stack(high_snr)
            
            # Get batch size
            batch_size = low_snr.size(0)
            
            # ---------------------
            # Train Discriminator
            # ---------------------
            d_optimizer.zero_grad()
            
            # Real data
            real_data = high_snr.to(device)
            real_outputs = model.module.discriminator(real_data)
            
            # Fake data
            fake_data = model.module.generator(low_snr.to(device))
            fake_outputs = model.module.discriminator(fake_data.detach())
            
            # Gradient penalty
            alpha = torch.rand(batch_size, 1, 1, 1, device=device)
            interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).requires_grad_(True)
            d_interpolates = model.module.discriminator(interpolates)
            
            gradients = torch.autograd.grad(
                outputs=d_interpolates,
                inputs=interpolates,
                grad_outputs=torch.ones_like(d_interpolates),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            
            gradients = gradients.view(gradients.size(0), -1)
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
            
            # Wasserstein loss
            d_loss = -torch.mean(real_outputs) + torch.mean(fake_outputs) + gradient_penalty
            d_loss.backward(retain_graph=True)
            d_optimizer.step()
            
            # -----------------
            # Train Generator
            # -----------------
            if i % 5 == 0:  # Train generator less frequently
                g_optimizer.zero_grad()
                # Detach fake_data to prevent inplace operation issues
                fake_outputs = model.module.discriminator(fake_data.detach())
                g_loss = -torch.mean(fake_outputs)
                g_loss.backward(retain_graph=True)
                g_optimizer.step()
                g_running_loss += g_loss.item()
            
            # Record discriminator loss
            d_running_loss += d_loss.item()
        
        # Calculate epoch losses
        d_epoch_loss = d_running_loss / len(training_params.train_dataset)
        g_epoch_loss = g_running_loss / len(training_params.train_dataset)
        d_loss_list.append(d_epoch_loss)
        g_loss_list.append(g_epoch_loss)
        
        # Print progress
        print(f"Epoch [{epoch+1}/{training_params.epochs}], "
              f"D Loss: {d_epoch_loss:.4f}, G Loss: {g_epoch_loss:.4f}")
    
    # Training summary
    time_elapsed = time.time() - since
    print("\n--- Training summary ---")
    print(f"Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
    
    # Save final model
    torch.save(model.state_dict(), checkpoint_path / model_name)
    return model, d_loss_list, g_loss_list

def plot_learning_curve(epoch_list, train_loss: list, validation_loss: list):
    """
    Plot the learning curve.

    Args:
    -----
        epoch_list (list): List of epochs.
        train_loss (list): List of training losses per epoch.
        validation_loss (list): List of validation losses per epoch.
    """
    plt.title("Learning Curve: Loss per Epoch")
    plt.plot(epoch_list, train_loss, label="Train")
    plt.plot(epoch_list, validation_loss, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.show()

def plot_gan_learning_curve(adv_loss_list: list, cls_loss_list: list, snr_loss_list: list):
    """
    Plot the GAN learning curve with three losses.

    Args:
    -----
        adv_loss_list (list): Adversarial loss per epoch.
        cls_loss_list (list): Classification loss per epoch.
        snr_loss_list (list): SNR regression loss per epoch.
    """
    epochs = range(len(adv_loss_list))
    plt.title("GAN Training Losses")
    plt.plot(epochs, adv_loss_list, label="Adversarial Loss")
    plt.plot(epochs, cls_loss_list, label="Classification Loss")
    plt.plot(epochs, snr_loss_list, label="SNR Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def simulation_summary(
    system_model_params: SystemModelParams,
    model_type: str,
    parameters: TrainingParams = None,
    phase="training",
):
    """
    Prints a summary of the simulation parameters.

    Args:
    -----
        model_type (str): The type of the model.
        M (int): The number of sources.
        N (int): The number of sensors.
        T (float): The number of observations.
        SNR (int): The signal-to-noise ratio.
        signal_type (str): The signal_type of the signals.
        mode (str): The nature of the sources.
        eta (float): The spacing deviation.
        bias (float): Value of bias deviation from nominal spacing.
        geo_noise_var (float): The geometry noise variance.
        parameters (TrainingParams): instance of the training parameters object
        phase (str, optional): The phase of the simulation. Defaults to "training", optional: "evaluation".
        tau (int, optional): The number of lags for auto-correlation (relevant only for SubspaceNet model).

    """
    print("\n--- New Simulation ---\n")
    print(f"Description: Simulation of {model_type}, {phase} stage")
    print("System model parameters:")
    print(f"Number of sources = {system_model_params.M}")
    print(f"Number of sensors = {system_model_params.N}")
    print(f"signal_type = {system_model_params.signal_type}")
    print(f"Observations = {system_model_params.T}")
    print(
        # f"SNR = {system_model_params.snr}, {system_model_params.signal_nature} sources"
        f"{system_model_params.signal_nature} sources" # by j  224
    )
    print(f"Spacing deviation (eta) = {system_model_params.eta}")
    print(f"Bias spacing deviation (eta) = {system_model_params.bias}")
    print(f"Geometry noise variance = {system_model_params.sv_noise_var}")
    print("Simulation parameters:")
    print(f"Model: {model_type}")
    print(f"test_snr: {system_model_params.snr}")
    if model_type.startswith("SubspaceNet"):
        print(f"SubspaceNet: tau = {parameters.tau}")
        print(
            f"SubspaceNet: differentiable subspace method  = {parameters.diff_method}"
        )
    if phase.startswith("training"):
        print(f"Epochs = {parameters.epochs}")
        print(f"Batch Size = {parameters.batch_size}")
        print(f"Learning Rate = {parameters.learning_rate}")
        print(f"Weight decay = {parameters.weight_decay}")
        print(f"Gamma Value = {parameters.gamma}")
        print(f"Step Value = {parameters.step_size}")


def get_simulation_filename(
    system_model_params: SystemModelParams, model_config: ModelGenerator
):
    return (
        f"{model_config.model_type}_M={system_model_params.M}_"
        + f"T={system_model_params.T}_SNR_{system_model_params.snr}_"
        + f"tau={model_config.tau}_{system_model_params.signal_type}_"
        + f"diff_method={model_config.diff_method}_"
        + f"{system_model_params.signal_nature}_eta={system_model_params.eta}_"
        + f"bias={system_model_params.bias}_"
        + f"sv_noise={system_model_params.sv_noise_var}"
    )


class OffgridDOA(nn.Module):
    def __init__(self, num_classes=241, N=16):
      
        # 定义您的模型结构
        super(OffgridDOA, self).__init__()
        
        # 分类网络
        self.cls_net  = nn.Sequential(
            nn.Linear(config['input_dim'], 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, config['cls_output']),
            nn.Sigmoid()
        )
        
        # 回归网络
        self.reg_net  = nn.Sequential(
            nn.Linear(400, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, config['reg_output'])
        )
        
    def forward(self, x):
        # 实现前向传播
         # 分类分支
        cls_feat1 = self.cls_net[:4](x)   # 前两层
        cls_feat2 = self.cls_net[4:8](cls_feat1)   # 第三层
        cls_feat3 = self.cls_net[8:12](cls_feat2)  # 第四层
        cls_output = self.cls_net[12:](cls_feat3)  # 第五层
        
        # 回归分支
        reg_input = torch.cat([x,  cls_feat3], dim=1)
        reg_output = self.reg_net(reg_input) 
        
        return cls_output, reg_output
