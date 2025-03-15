"""Subspace-Net 
Details
----------
    Name: data_handler.py
    Authors: D. H. Shmuel
    Created: 01/10/21
    Edited: 03/06/23

Purpose:
--------
    This scripts handle the creation and processing of synthetic datasets
    based on specified parameters and model types.
    It includes functions for generating datasets, reading data from files,
    computing autocorrelation matrices, and creating covariance tensors.

Attributes:
-----------
    Samples (from src.signal_creation): A class for creating samples used in dataset generation.

    The script defines the following functions:
    * create_dataset: Generates a synthetic dataset based on the specified parameters and model type.
    * read_data: Reads data from a file specified by the given path.
    * autocorrelation_matrix: Computes the autocorrelation matrix for a given lag of the input samples.
    * create_autocorrelation_tensor: Returns a tensor containing all the autocorrelation matrices for lags 0 to tau.
    * create_cov_tensor: Creates a 3D tensor containing the real part,
        imaginary part, and phase component of the covariance matrix.
    * set_dataset_filename: Returns the generic suffix of the datasets filename.

"""

# Imports
import torch
import numpy as np
import itertools
from tqdm import tqdm
from src.signal_creation import Samples
from pathlib import Path
from src.system_model import SystemModelParams

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config = {
    # 阵列参数
    'N': 12,                # 阵元数
    'K': 2,                 # 源数
    'theta_range': (-60, 60), # 角度范围
    'min_separation': 2,    # 最小源间隔(度)
    'T': 100,               # 快照数
    'snr_db': 10,           # 信噪比(dB)
    'sigma_n': 0.1**0.5,    # 噪声标准差
    
    # 模型参数
    'grid_resolution': 1,   # 网格分辨率(度)
    'input_dim': 144,       # 输入维度
    'cls_output': 121,      # 分类输出维度
    'reg_output': 2,        # 回归输出维度
    
    # 训练参数
    'total_samples': 6300000, # 总样本数
    'batch_size': 512,      
    'epochs': 300,
    'lr': 0.001,
    'loss_weights': (100, 0.1)
}
def create_dataset(
        system_model_params: SystemModelParams,
        samples_size: float,
        model_type: str,
        tau: int = None,
        save_datasets: bool = False,
        datasets_path: Path = None,
        true_doa: list = None,
        phase: str = None,
):
    """
    Generates a synthetic dataset based on the specified parameters and model type.

    Args:
    -----
        system_model_params (SystemModelParams): an instance of SystemModelParams
        samples_size (float): The size of the dataset
        model_type (str): The type of the model ('SubspaceNet', 'My_transform_Model', 'DeepCNN', etc.)
        tau (int): The number of lags for auto-correlation (relevant only for SubspaceNet model)
        save_datasets (bool, optional): Specifies whether to save the dataset. Defaults to False
        datasets_path (Path, optional): The path for saving the dataset. Defaults to None
        true_doa (list, optional): Predefined angles. Defaults to None
        phase (str, optional): The phase of the dataset (test or training phase). Defaults to None

    Returns:
    --------
        tuple: (model_dataset, generic_dataset, samples_model) containing the desired datasets
    """
    generic_dataset = []
    model_dataset = []
    samples_model = Samples(system_model_params)
    # if (model_type.startswith("OffgridDOA"))and phase.startswith("train"):
    #
    # Generate permutations for CNN-based model training datasets
    if (model_type.startswith(("My_transform_Model", "DeepCNN")) and
            phase.startswith("train")):
        doa_permutations = []
        angles_grid = np.linspace(start=-15, stop=15, num=system_model_params.grid_size)
        for comb in itertools.combinations(angles_grid, system_model_params.M):
            doa_permutations.append(list(comb))

    # Training phase for CNN-based models
    if (model_type.startswith(("My_transform_Model", "DeepCNN")) and
            phase.startswith("train")):
        for i, doa in tqdm(enumerate(doa_permutations)):
            samples_model.set_doa(doa)
            X = torch.tensor(
                samples_model.samples_creation(
                    noise_mean=0, noise_variance=1, signal_mean=0, signal_variance=1
                )[0],
                dtype=torch.complex64,
            )
            # Model-specific transformation
            if model_type.startswith("My_transform_Model"):
                X_model = create_rx_tensor(X)  # 4-channel for My_transform_Model
            elif model_type.startswith("DeepCNN"):
                X_model = create_cov_tensor(X)  # 3-channel for DeepCNN

            # Ground-truth creation (One-Hot encoding)
            Y = torch.zeros_like(torch.tensor(angles_grid))
            for angle in doa:
                Y[list(angles_grid).index(angle)] = 1

            # 使用高斯核函数生成软标签
            #     # Ground-truth creation (One-Hot encoding)
            # Y = torch.zeros_like(torch.tensor(angles_grid))
            # sigma = 2.0  # 可调整的核函数宽度参数
            # angles_grid_tensor = torch.tensor(angles_grid, dtype=torch.float16)
            #
            # for i, grid_angle in enumerate(angles_grid_tensor):
            #     # 对每个网格点,计算与所有真实角度的高斯核函数值之和
            #     kernel_sum = 0
            #     for theta in doa:
            #         # 将theta转换为tensor并确保类型匹配
            #         theta_tensor = torch.tensor(theta, dtype=torch.float16)
            #         # 计算高斯核函数
            #         kernel = torch.exp(-(grid_angle - theta_tensor) ** 2 / (2 * sigma ** 2))
            #         kernel_sum += kernel
            #     Y[i] = kernel_sum
            #
            # # 归一化处理(可选)
            # Y = Y / torch.max(Y)

            model_dataset.append((X_model, Y))
            generic_dataset.append((X, Y))

    # Test phase or non-CNN models
    else:
        for i in tqdm(range(samples_size)):
            # samples_model.set_doa(true_doa[i])
            if true_doa == None:
                samples_model.set_doa(true_doa)
            else:
                samples_model.set_doa(true_doa[i])
            X = torch.tensor(
                samples_model.samples_creation(
                    noise_mean=0, noise_variance=1, signal_mean=0, signal_variance=1
                )[0],
                dtype=torch.complex64,
            )

            # Model-specific transformations
            if model_type.startswith("SubspaceNet"):
                X_model = create_autocorrelation_tensor(X, tau).to(torch.float)
            elif model_type.startswith("My_transform_Model") and phase.startswith("test"):
                X_model = create_rx_tensor(X)  # 4-channel for testing
            elif model_type.startswith("DeepCNN") and phase.startswith("test"):
                X_model = create_cov_tensor(X)  # 3-channel for testing
            else:
                X_model = X

            # Ground-truth creation (raw DOA values)
            Y = torch.tensor(samples_model.doa, dtype=torch.float64)
            generic_dataset.append((X, Y))
            model_dataset.append((X_model, Y))

    # Save datasets if requested
    if save_datasets:
        model_dataset_filename = f"{model_type}_DataSet" + set_dataset_filename(
            system_model_params, samples_size
        )
        generic_dataset_filename = f"Generic_DataSet" + set_dataset_filename(
            system_model_params, samples_size
        )
        samples_model_filename = f"samples_model" + set_dataset_filename(
            system_model_params, samples_size
        )

        torch.save(obj=model_dataset, f=datasets_path / phase / model_dataset_filename)
        torch.save(
            obj=generic_dataset, f=datasets_path / phase / generic_dataset_filename
        )
        if phase.startswith("test"):
            torch.save(
                obj=samples_model, f=datasets_path / phase / samples_model_filename
            )

    return model_dataset, generic_dataset, samples_model

# def read_data(Data_path: str) -> torch.Tensor:
def read_data(path: str):
    """
    Reads data from a file specified by the given path.

    Args:
    -----
        path (str): The path to the data file.

    Returns:
    --------
        torch.Tensor: The loaded data.

    Raises:
    -------
        None

    Examples:
    ---------
        >>> path = "data.pt"
        >>> read_data(path)

    """
    assert isinstance(path, (str, Path))
    data = torch.load(path)
    return data


# def autocorrelation_matrix(X: torch.Tensor, lag: int) -> torch.Tensor:
def autocorrelation_matrix(X: torch.Tensor, lag: int):
    """
    Computes the autocorrelation matrix for a given lag of the input samples.

    Args:
    -----
        X (torch.Tensor): Samples matrix input with shape [N, T].
        lag (int): The requested delay of the autocorrelation calculation.

    Returns:
    --------
        torch.Tensor: The autocorrelation matrix for the given lag.

    """
    Rx_lag = torch.zeros(X.shape[0], X.shape[0], dtype=torch.complex128).to(device)
    for t in range(X.shape[1] - lag):
        # meu = torch.mean(X,1)
        x1 = torch.unsqueeze(X[:, t], 1).to(device)
        x2 = torch.t(torch.unsqueeze(torch.conj(X[:, t + lag]), 1)).to(device)
        Rx_lag += torch.matmul(x1 - torch.mean(X), x2 - torch.mean(X)).to(device)
    Rx_lag = Rx_lag / (X.shape[-1] - lag)
    Rx_lag = torch.cat((torch.real(Rx_lag), torch.imag(Rx_lag)), 0)
    return Rx_lag


# def create_autocorrelation_tensor(X: torch.Tensor, tau: int) -> torch.Tensor:
def create_autocorrelation_tensor(X: torch.Tensor, tau: int):
    """
    Returns a tensor containing all the autocorrelation matrices for lags 0 to tau.

    Args:
    -----
        X (torch.Tensor): Observation matrix input with size (BS, N, T).
        tau (int): Maximal time difference for the autocorrelation tensor.

    Returns:
    --------
        torch.Tensor: Tensor containing all the autocorrelation matrices,
                    with size (Batch size, tau, 2N, N).

    Raises:
    -------
        None

    """
    Rx_tau = []
    for i in range(tau):
        Rx_tau.append(autocorrelation_matrix(X, lag=i))
    Rx_autocorr = torch.stack(Rx_tau, dim=0)
    return Rx_autocorr


# def create_cov_tensor(X: torch.Tensor) -> torch.Tensor:
def create_cov_tensor(X: torch.Tensor):
    """
    Creates a 3D tensor of size (NxNx3) containing the real part, imaginary part, and phase component of the covariance matrix.

    Args:
    -----
        X (torch.Tensor): Observation matrix input with size (N, T).

    Returns:
    --------
        Rx_tensor (torch.Tensor): Tensor containing the auto-correlation matrices, with size (Batch size, N, N, 3).

    Raises:
    -------
        None

    """
    Rx = torch.cov(X)
    # Rx = torch.cov(X.T)
    # Rx=X
    Rx_tensor = torch.stack((torch.real(Rx), torch.imag(Rx), torch.angle(Rx)), 2)
    return Rx_tensor


def create_rx_tensor(X):
    """
    Creates a 4-channel tensor of size (NxNx4) for a single sample.

    Args:
        X (torch.Tensor): Input observation matrix of shape (N, T).

    Returns:
        Rx_tensor (torch.Tensor): Output tensor of shape (N, N, 4).
    """
    # Compute covariance matrix (shape: N x N)
    Rx = torch.cov(X)  # X is complex-valued, Rx will also be complex

    # Channel 1: Real lower + Imag upper (excluding diagonal)
    real_part = torch.real(Rx)
    imag_part = torch.imag(Rx)
    channel1 = torch.tril(real_part) + torch.triu(imag_part, diagonal=1)

    # Channel 2: Magnitude lower + Phase upper (excluding diagonal)
    mag = torch.abs(Rx)
    phase = torch.angle(Rx)
    channel2 = torch.tril(mag) + torch.triu(phase, diagonal=1)

    # Signal subspace processing
    # Eigen decomposition for Hermitian matrix
    eigenvalues, eigenvectors = torch.linalg.eigh(Rx)

    # Select top 2 eigenvectors (signal subspace)
    U_s = eigenvectors[:, -2:]  # shape (N, 2)

    # Compute signal subspace matrix
    S = U_s @ U_s.conj().T  # shape (N, N)

    # Channel 3: Signal subspace real/imag fusion
    s_real = torch.real(S)
    s_imag = torch.imag(S)
    channel3 = torch.tril(s_real) + torch.triu(s_imag, diagonal=1)

    # Channel 4: Signal subspace mag/phase fusion
    s_mag = torch.abs(S)
    s_phase = torch.angle(S)
    channel4 = torch.tril(s_mag) + torch.triu(s_phase, diagonal=1)

    # Combine channels
    Rx_tensor = torch.stack([channel1, channel2, channel3, channel4], dim=-1)

    return Rx_tensor

def load_datasets(
    system_model_params: SystemModelParams,
    model_type: str,
    samples_size: float,
    datasets_path: Path,
    train_test_ratio: float,
    is_training: bool = False,
):
    """
    Load different datasets based on the specified parameters and phase.

    Args:
    -----
        system_model_params (SystemModelParams): an instance of SystemModelParams.
        model_type (str): The type of the model.
        samples_size (float): The size of the overall dataset.
        datasets_path (Path): The path to the datasets.
        train_test_ratio (float): The ration between train and test datasets.
        is_training (bool): Specifies whether to load the training dataset.

    Returns:
    --------
        List: A list containing the loaded datasets.

    """
    datasets = []
    # Define test set size
    test_samples_size = int(train_test_ratio * samples_size)
    # Generate datasets filenames
    model_dataset_filename = f"{model_type}_DataSet" + set_dataset_filename(
        system_model_params, test_samples_size
    )
    generic_dataset_filename = f"Generic_DataSet" + set_dataset_filename(
        system_model_params, test_samples_size
    )
    samples_model_filename = f"samples_model" + set_dataset_filename(
        system_model_params, test_samples_size
    )

    # Whether to load the training dataset
    if is_training:
        # Load training dataset
        try:
            model_trainingset_filename = f"{model_type}_DataSet" + set_dataset_filename(
                system_model_params, samples_size
            )
            train_dataset = read_data(
                datasets_path / "train" / model_trainingset_filename
            )
            datasets.append(train_dataset)
        except:
            raise Exception("load_datasets: Training dataset doesn't exist")
    # Load test dataset
    try:
        test_dataset = read_data(datasets_path / "test" / model_dataset_filename)
        datasets.append(test_dataset)
    except:
        raise Exception("load_datasets: Test dataset doesn't exist")
    # Load generic test dataset
    try:
        generic_test_dataset = read_data(
            datasets_path / "test" / generic_dataset_filename
        )
        datasets.append(generic_test_dataset)
    except:
        raise Exception("load_datasets: Generic test dataset doesn't exist")
    # Load samples models
    try:
        samples_model = read_data(datasets_path / "test" / samples_model_filename)
        datasets.append(samples_model)
    except:
        raise Exception("load_datasets: Samples model dataset doesn't exist")
    return datasets


def set_dataset_filename(system_model_params: SystemModelParams, samples_size: float):
    """Returns the generic suffix of the datasets filename.

    Args:
    -----
        system_model_params (SystemModelParams): an instance of SystemModelParams.
        samples_size (float): The size of the overall dataset.

    Returns:
    --------
        str: Suffix dataset filename
    """
    suffix_filename = (
        f"_{system_model_params.signal_type}_"
        + f"{system_model_params.signal_nature}_{samples_size}_M={system_model_params.M}_"
        + f"N={system_model_params.N}_T={system_model_params.T}_SNR={system_model_params.snr}_"
        + f"eta={system_model_params.eta}_sv_noise_var{system_model_params.sv_noise_var}_"
        + f"bias={system_model_params.bias}_"
        + ".h5"
    )
    return suffix_filename
