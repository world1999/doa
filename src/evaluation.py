"""
Subspace-Net

Details
----------
Name: evaluation.py
Authors: D. H. Shmuel
Created: 01/10/21
Edited: 17/03/23

Purpose
----------
This module provides functions for evaluating the performance of Subspace-Net and others Deep learning benchmarks,
add for conventional subspace methods. 
This scripts also defines function for plotting the methods spectrums.
In addition, 


Functions:
----------
evaluate_dnn_model: Evaluate the DNN model on a given dataset.
evaluate_augmented_model: Evaluate an augmented model that combines a SubspaceNet model.
evaluate_model_based: Evaluate different model-based algorithms on a given dataset.
add_random_predictions: Add random predictions if the number of predictions
    is less than the number of sources.
evaluate: Wrapper function for model and algorithm evaluations.


"""
# Imports
import torch.nn as nn
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema

from src.utils import device
from src.criterions import RMSPELoss, MSPELoss
from src.criterions import RMSPE, MSPE
from src.methods import MUSIC, RootMUSIC, Esprit, MVDR
from src.utils import *
from src.models import SubspaceNet
from src.plotting import plot_spectrum, detect_top_peaks


def evaluate_dnn_model(
    model,
    dataset: list,
    criterion: nn.Module,
    plot_spec: bool = False,
    figures: dict = None,
    model_type: str = "SubspaceNet",
):
    """
    Evaluate the DNN model on a given dataset.

    Args:
        model (nn.Module): The trained model to evaluate.
        dataset (list): The evaluation dataset.
        criterion (nn.Module): The loss criterion for evaluation.
        plot_spec (bool, optional): Whether to plot the spectrum for SubspaceNet model. Defaults to False.
        figures (dict, optional): Dictionary containing figure objects for plotting. Defaults to None.
        model_type (str, optional): The type of the model. Defaults to "SubspaceNet".

    Returns:
        float: The overall evaluation loss.

    Raises:
        Exception: If the loss criterion is not defined for the specified model type.
        Exception: If the model type is not defined.
    """

    # Initialize values
    overall_loss = 0.0
    test_length = 0
    # Set model to eval mode   model.eval() 使模型进入评估模式，影响 BatchNorm 和 Dropout 层的行为。
    model.eval()
    # Gradients calculation isn't required for evaluation
    with torch.no_grad():
        for i, data in enumerate(dataset):
            X, DOA = data
            test_length += DOA.shape[0]
            # Convert observations and DoA to device
            X = X.to(device)
            DOA = DOA.to(device)
            # Get model output
            model_output = model(X)
            if model_type.startswith("DA-MUSIC"):
                # Deep Augmented MUSIC
                DOA_predictions = model_output
            elif model_type.startswith("DeepCNN"):
                # Deep CNN02
                if isinstance(criterion, nn.BCELoss):
                    # If evaluation performed over validation set, loss is BCE
                    DOA_predictions = model_output

                    # # find peaks in the pseudo spectrum of probabilities
                    # DOA_predictions = (
                    #     get_k_peaks(361, DOA.shape[1], DOA_predictions[0]) * D2R
                    # )
                    # DOA_predictions = DOA_predictions.view(1, DOA_predictions.shape[0])
                elif isinstance(criterion, (RMSPELoss, MSPELoss)):
                    # If evaluation performed over testset, loss is RMSPE / MSPE
                    # DOA_predictions = model_output
                    DOA_predictions = model_output[0].cpu().numpy()


                    # 生成角度坐标
                    angles = np.linspace(-15, 15, 241)
                    predictions_norm = DOA_predictions / np.max(DOA_predictions)
                    selected_peaks, peak_angles = detect_top_peaks(
                        predictions_norm,
                        angles,
                        min_distance=5,
                        top_k=2
                    )
                    DOA_predictions=peak_angles*D2R


                    # # # find peaks in the pseudo spectrum of probabilities
                    # DOA_predictions = (
                    #         get_k_peaks(121, DOA.shape[1], DOA_predictions[0]) * D2R
                    # )
                    DOA_predictions = torch.tensor(DOA_predictions).view(1, DOA_predictions.shape[0])
                    # print(f"[DeepCNN] Predicted DOAs: {DOA_predictions}")


                else:
                    raise Exception(
                        f"evaluate_dnn_model: Loss criterion is not defined for {model_type} model"
                    )
            elif model_type.startswith("SubspaceNet"):
                # Default - SubSpaceNet
                DOA_predictions = model_output[0]
            else:
                raise Exception(
                    f"evaluate_dnn_model: Model type {model_type} is not defined"
                )
            # Compute prediction loss

            if model_type.startswith("DeepCNN") and isinstance(criterion, RMSPELoss):
                eval_loss = criterion(torch.tensor(DOA_predictions).float(), DOA.float())  # by j
            else:
                eval_loss = criterion(torch.tensor(DOA_predictions).float(), DOA.float())
            # add the batch evaluation loss to epoch loss
            overall_loss += eval_loss.item()
        overall_loss = overall_loss / test_length
    # Plot spectrum for SubspaceNet model
    if plot_spec and model_type.startswith("SubspaceNet"):
        DOA_all = model_output[1]
        roots = model_output[2]
        plot_spectrum(
            predictions=DOA_all * R2D,
            true_DOA=DOA[0] * R2D,
            roots=roots,
            algorithm="SubNet+R-MUSIC",
            figures=figures,
        )
    if plot_spec and model_type.startswith("DeepCNN"):
        # 获取DeepCNN模型输出的概率谱
        spectrum = model_output[0].cpu().numpy()  # 假设模型输出为(batch_size, 361)的概率分布
        true_DOA_deg = DOA[0].cpu().numpy() * R2D  # 转换为度数
        # 提取预测的峰值角度
        predicted_peaks_deg = DOA_predictions[0].numpy() * R2D

        # 调用绘图函数
        if plot_spec and i == len(dataset.dataset) - 1:
            plot_spectrum(
                predictions=spectrum,  # 概率谱数据
                true_DOA=true_DOA_deg,  # 真实角度
                roots=predicted_peaks_deg,  # 预测的峰值角度（用roots参数传递）
                algorithm="My_transform_Model",  # 算法标识
                figures=figures  # 图形容器
            )
    # if plot_spec and model_type.startswith("DeepCNN"):
    #     DOA_all = model_output[1]
    #     roots = model_output[2]
    #     plot_spectrum(
    #         predictions=DOA_all * R2D,
    #         true_DOA=DOA[0] * R2D,
    #         roots=roots,
    #         algorithm="SubNet+R-MUSIC",
    #         figures=figures,
    #     )
    return overall_loss

def evaluate_transformer_model(
        model,
        dataset: list,
        criterion: nn.Module,
        plot_spec: bool = False,
        figures: dict = None,
        model_type: str = "My_transform_Model",
):
    """
    Evaluate the DNN model on a given dataset, with loss and accuracy metrics.

    Args:
        model (nn.Module): The trained model to evaluate.
        dataset (list): The evaluation dataset.
        criterion (nn.Module): The loss criterion for evaluation.
        plot_spec (bool, optional): Whether to plot the spectrum for SubspaceNet model. Defaults to False.
        figures (dict, optional): Dictionary containing figure objects for plotting. Defaults to None.
        model_type (str, optional): The type of the model. Defaults to "My_transform_Model".

    Returns:
        tuple: (overall_loss, accuracy) - Average evaluation loss and prediction accuracy.

    Raises:
        Exception: If the loss criterion is not defined for the specified model type.
        Exception: If the model type is not defined.
    """
    # 初始化
    overall_loss = 0.0
    test_length = 0
    correct_predictions = 0  # 正确预测的样本数
    model.eval()
    D2R = np.pi / 180  # 度到弧度的转换常数，假设未定义时在此定义

    with torch.no_grad():
        for i, data in enumerate(dataset):
            X, DOA = data
            batch_size = DOA.shape[0]
            test_length += batch_size
            X = X.to(device)
            DOA = DOA.to(device)

            # 获取模型输出
            model_output = model(X)

            if model_type.startswith("DA-MUSIC"):
                DOA_predictions = model_output
            elif model_type.startswith("My_transform_Model"):
                if isinstance(criterion, nn.BCELoss):
                    DOA_predictions = model_output
                elif isinstance(criterion, (RMSPELoss, MSPELoss)):
                    DOA_predictions = model_output[0].cpu().numpy()
                    angles = np.linspace(-15, 15, 241)
                    predictions_norm = DOA_predictions / np.max(DOA_predictions)
                    selected_peaks, peak_angles = detect_top_peaks(
                        predictions_norm, angles, min_distance=5, top_k=2
                    )
                    DOA_predictions = peak_angles * D2R
                    DOA_predictions = torch.tensor(DOA_predictions, device=device).view(1, -1)
                else:
                    raise Exception(
                        f"evaluate_dnn_model: Loss criterion is not defined for {model_type} model"
                    )
            elif model_type.startswith("SubspaceNet"):
                DOA_predictions = model_output[0]
            else:
                raise Exception(
                    f"evaluate_dnn_model: Model type {model_type} is not defined"
                )

            # 计算损失
            if model_type.startswith("My_transform_Model") and isinstance(criterion, RMSPELoss):
                eval_loss = criterion(DOA_predictions.float(), DOA.float())
            else:
                eval_loss = criterion(DOA_predictions.float(), DOA.float())
            overall_loss += eval_loss.item() * batch_size  # 按样本数加权

            # 计算正确率
            DOA_pred = DOA_predictions.cpu().numpy()  # [1, 2] or [batch, 2]
            DOA_true = DOA.cpu().numpy()  # [batch, 2]
            for b in range(batch_size):
                pred_angles = DOA_pred[b] if DOA_pred.ndim > 1 else DOA_pred  # 取当前样本预测
                true_angles = DOA_true[b]  # 取当前样本真实值

                # 转换为度数比较（假设DOA是以弧度存储）
                pred_angles_deg = pred_angles / D2R
                true_angles_deg = true_angles / D2R

                # 只考虑有效角度（假设无效角度标记为-1）
                valid_true = true_angles_deg[true_angles_deg != -1]
                valid_pred = pred_angles_deg[:len(valid_true)]  # 截取匹配数量的预测角度
                if len(valid_pred) < 2:
                    continue  # 如果预测角度少于2个，跳过当前样本

                if len(valid_true) == 0:  # 无有效目标，跳过
                    continue
                elif len(valid_true) == 1:  # 单目标
                    if abs(valid_true[0] - valid_pred[0]) <= 2:
                        correct_predictions += 1
                else:  # 双目标，检查两种顺序
                    # 计算两种顺序的每个角度差值
                    diff1_0 = abs(valid_pred[0] - valid_true[0])  # 顺序1：pred[0] vs true[0]
                    diff1_1 = abs(valid_pred[1] - valid_true[1])  # 顺序1：pred[1] vs true[1]
                    diff2_0 = abs(valid_pred[0] - valid_true[1])  # 顺序2：pred[0] vs true[1]
                    diff2_1 = abs(valid_pred[1] - valid_true[0])  # 顺序2：pred[1] vs true[0]

                    # 检查是否存在一种顺序，每对差值都小于2°
                    if (diff1_0 <= 1 and diff1_1 <= 1) or (diff2_0 <= 1 and diff2_1 <= 1):
                        correct_predictions += 1.0
                if plot_spec and model_type.startswith("My_transform_Model"):
                    # 获取DeepCNN模型输出的概率谱
                    spectrum = model_output[0].cpu().numpy()  # 假设模型输出为(batch_size, 361)的概率分布
                    true_DOA_deg = DOA[0].cpu().numpy() * R2D  # 转换为度数
                    # 提取预测的峰值角度
                    predicted_peaks_deg = DOA_predictions[0].cpu().numpy() * R2D

                    # 调用绘图函数
                    if plot_spec:
                        plot_spectrum(
                            predictions=spectrum,  # 概率谱数据
                            true_DOA=true_DOA_deg,  # 真实角度
                            roots=predicted_peaks_deg,  # 预测的峰值角度（用roots参数传递）
                            algorithm="My_transform_Model",  # 算法标识
                            figures=figures,  # 图形容器
                            sample_idx=i,  # 新增参数：样本索引   by j 224
                        )

        # 计算平均损失和正确率
        overall_loss = overall_loss / test_length
        accuracy = correct_predictions / test_length if test_length > 0 else 0.0

    return overall_loss, accuracy

# def evaluate_transformer_model(
#     model,
#     dataset: list,
#     criterion: nn.Module,
#     plot_spec: bool = False,
#     figures: dict = None,
#     model_type: str = "My_transform_Model",
# ):
#     """
#     Evaluate the DNN model on a given dataset.
#
#     Args:
#         model (nn.Module): The trained model to evaluate.
#         dataset (list): The evaluation dataset.
#         criterion (nn.Module): The loss criterion for evaluation.
#         plot_spec (bool, optional): Whether to plot the spectrum for SubspaceNet model. Defaults to False.
#         figures (dict, optional): Dictionary containing figure objects for plotting. Defaults to None.
#         model_type (str, optional): The type of the model. Defaults to "SubspaceNet".
#
#     Returns:
#         float: The overall evaluation loss.
#
#     Raises:
#         Exception: If the loss criterion is not defined for the specified model type.
#         Exception: If the model type is not defined.
#     """

    # Initialize values
    # overall_loss = 0.0
    # test_length = 0
    # # Set model to eval mode   model.eval() 使模型进入评估模式，影响 BatchNorm 和 Dropout 层的行为。
    # model.eval()
    # # Gradients calculation isn't required for evaluation
    # with torch.no_grad():
    #     for i, data in enumerate(dataset):
    #         X, DOA = data
    #         test_length += DOA.shape[0]
    #         # Convert observations and DoA to device
    #         X = X.to(device)
    #         DOA = DOA.to(device)
    #         # Get model output
    #         model_output = model(X)
    #         if model_type.startswith("DA-MUSIC"):
    #             # Deep Augmented MUSIC
    #             DOA_predictions = model_output
    #         elif model_type.startswith("My_transform_Model"):
    #             # Deep CNN02
    #             if isinstance(criterion, nn.BCELoss):
    #                 # If evaluation performed over validation set, loss is BCE
    #                 DOA_predictions = model_output
    #
    #                 # # find peaks in the pseudo spectrum of probabilities
    #                 # DOA_predictions = (
    #                 #     get_k_peaks(361, DOA.shape[1], DOA_predictions[0]) * D2R
    #                 # )
    #                 # DOA_predictions = DOA_predictions.view(1, DOA_predictions.shape[0])
    #             elif isinstance(criterion, (RMSPELoss, MSPELoss)):
    #                 # If evaluation performed over testset, loss is RMSPE / MSPE
    #                 # DOA_predictions = model_output
    #                 DOA_predictions = model_output[0].cpu().numpy()
    #
    #
    #                 # 生成角度坐标
    #                 angles = np.linspace(-15, 15, 241)
    #                 predictions_norm = DOA_predictions / np.max(DOA_predictions)
    #                 selected_peaks, peak_angles = detect_top_peaks(
    #                     predictions_norm,
    #                     angles,
    #                     min_distance=5,
    #                     top_k=2
    #                 )
    #                 DOA_predictions=peak_angles*D2R
    #
    #
    #                 # # # find peaks in the pseudo spectrum of probabilities
    #                 # DOA_predictions = (
    #                 #         get_k_peaks(121, DOA.shape[1], DOA_predictions[0]) * D2R
    #                 # )
    #                 DOA_predictions = torch.tensor(DOA_predictions).view(1, DOA_predictions.shape[0])
    #                 # print(f"[My_transform_Model] Predicted DOAs: {DOA_predictions}")
    #
    #
    #             else:
    #                 raise Exception(
    #                     f"evaluate_dnn_model: Loss criterion is not defined for {model_type} model"
    #                 )
    #         elif model_type.startswith("SubspaceNet"):
    #             # Default - SubSpaceNet
    #             DOA_predictions = model_output[0]
    #         else:
    #             raise Exception(
    #                 f"evaluate_dnn_model: Model type {model_type} is not defined"
    #             )
    #         # Compute prediction loss
    #
    #         if model_type.startswith("My_transform_Model") and isinstance(criterion, RMSPELoss):
    #             eval_loss = criterion(torch.tensor(DOA_predictions).float(), DOA.float())  # by j
    #         else:
    #             eval_loss = criterion(torch.tensor(DOA_predictions).float(), DOA.float())
    #         # add the batch evaluation loss to epoch loss
    #         overall_loss += eval_loss.item()
    #     overall_loss = overall_loss / test_length
    # # Plot spectrum for SubspaceNet model
    # if plot_spec and model_type.startswith("SubspaceNet"):
    #     DOA_all = model_output[1]
    #     roots = model_output[2]
    #     plot_spectrum(
    #         predictions=DOA_all * R2D,
    #         true_DOA=DOA[0] * R2D,
    #         roots=roots,
    #         algorithm="SubNet+R-MUSIC",
    #         figures=figures,
    #     )
    # if plot_spec and model_type.startswith("My_transform_Model"):
    #     # 获取DeepCNN模型输出的概率谱
    #     spectrum = model_output[0].cpu().numpy()  # 假设模型输出为(batch_size, 361)的概率分布
    #     true_DOA_deg = DOA[0].cpu().numpy() * R2D  # 转换为度数
    #     # 提取预测的峰值角度
    #     predicted_peaks_deg = DOA_predictions[0].numpy() * R2D
    #
    #     # 调用绘图函数
    #     if plot_spec and i == len(dataset.dataset) - 1:
    #         plot_spectrum(
    #             predictions=spectrum,  # 概率谱数据
    #             true_DOA=true_DOA_deg,  # 真实角度
    #             roots=predicted_peaks_deg,  # 预测的峰值角度（用roots参数传递）
    #             algorithm="My_transform_Model",  # 算法标识
    #             figures=figures  # 图形容器
    #         )
    # # if plot_spec and model_type.startswith("DeepCNN"):
    # #     DOA_all = model_output[1]
    # #     roots = model_output[2]
    # #     plot_spectrum(
    # #         predictions=DOA_all * R2D,
    # #         true_DOA=DOA[0] * R2D,
    # #         roots=roots,
    # #         algorithm="SubNet+R-MUSIC",
    # #         figures=figures,
    # #     )
    # return overall_loss

def evaluate_augmented_model(
    model: SubspaceNet,
    dataset,
    system_model,
    criterion=RMSPE,
    algorithm: str = "music",
    plot_spec: bool = False,
    figures: dict = None,
):
    """
    Evaluate an augmented model that combines a SubspaceNet model with another subspace method on a given dataset.

    Args:
    -----
        model (nn.Module): The trained SubspaceNet model.
        dataset: The evaluation dataset.
        system_model (SystemModel): The system model for the hybrid algorithm.
        criterion: The loss criterion for evaluation. Defaults to RMSPE.
        algorithm (str): The hybrid algorithm to use (e.g., "music", "mvdr", "esprit"). Defaults to "music".
        plot_spec (bool): Whether to plot the spectrum for the hybrid algorithm. Defaults to False.
        figures (dict): Dictionary containing figure objects for plotting. Defaults to None.

    Returns:
    --------
        float: The average evaluation loss.

    Raises:
    -------
        Exception: If the algorithm is not supported.
        Exception: If the algorithm is not supported
    """
    # Initialize parameters for evaluation
    hybrid_loss = []
    if not isinstance(model, SubspaceNet):
        raise Exception("evaluate_augmented_model: model is not from type SubspaceNet")
    # Set model to eval mode
    model.eval()
    # Initialize instances of subspace methods
    methods = {
        "mvdr": MVDR(system_model),
        "music": MUSIC(system_model),
        "esprit": Esprit(system_model),
        "r-music": RootMUSIC(system_model),
    }
    # If algorithm is not in methods
    if methods.get(algorithm) is None:
        raise Exception(
            f"evaluate_augmented_model: Algorithm {algorithm} is not supported."
        )
    # Gradients calculation isn't required for evaluation
    with torch.no_grad():
        for i, data in enumerate(dataset):
            X, DOA = data
            # Convert observations and DoA to device
            X = X.to(device)
            DOA = DOA.to(device)
            # Apply method with SubspaceNet augmentation
            method_output = methods[algorithm].narrowband(
                X=X, mode="SubspaceNet", model=model
            )
            # Calculate loss, if algorithm is "music" or "esprit"
            if not algorithm.startswith("mvdr"):
                predictions, M = method_output[0], method_output[-1]
                # If the amount of predictions is less than the amount of sources
                predictions = add_random_predictions(M, predictions, algorithm)
                # Calculate loss criterion
                loss = criterion(predictions, DOA * R2D)
                hybrid_loss.append(loss)
            else:
                hybrid_loss.append(0)
            # Plot spectrum, if algorithm is "music" or "mvdr"
            if not algorithm.startswith("esprit"):
                if plot_spec and i == len(dataset.dataset) - 1:
                    predictions, spectrum = method_output[0], method_output[1]
                    figures[algorithm]["norm factor"] = np.max(spectrum)
                    plot_spectrum(
                        predictions=predictions,
                        true_DOA=DOA * R2D,
                        system_model=system_model,
                        spectrum=spectrum,
                        algorithm="SubNet+" + algorithm.upper(),
                        figures=figures,
                    )
    return np.mean(hybrid_loss)


# def evaluate_model_based(
#     dataset: list,
#     system_model,
#     criterion: RMSPE,
#     plot_spec=False,
#     algorithm: str = "music",
#     figures: dict = None,
# ):
#     """
#     Evaluate different model-based algorithms on a given dataset.
#
#     Args:
#         dataset (list): The evaluation dataset.
#         system_model (SystemModel): The system model for the algorithms.
#         criterion: The loss criterion for evaluation. Defaults to RMSPE.
#         plot_spec (bool): Whether to plot the spectrum for the algorithms. Defaults to False.
#         algorithm (str): The algorithm to use (e.g., "music", "mvdr", "esprit", "r-music"). Defaults to "music".
#         figures (dict): Dictionary containing figure objects for plotting. Defaults to None.
#
#     Returns:
#         float: The average evaluation loss.
#
#     Raises:
#         Exception: If the algorithm is not supported.
#     """
#     # Initialize parameters for evaluation
#     loss_list = []
#     for i, data in enumerate(dataset):
#         # if i != len(dataset) - 1:
#         #     continue  # 跳过其他循环，直接进入最后一个数据
#         X, doa = data
#         X = X[0]
#         # Root-MUSIC algorithms
#         if "r-music" in algorithm:
#             root_music = RootMUSIC(system_model)
#             if algorithm.startswith("sps"):
#                 # Spatial smoothing
#                 predictions, roots, predictions_all, _, M = root_music.narrowband(
#                     X=X, mode="spatial_smoothing"
#                 )
#             else:
#                 # Conventional
#                 predictions, roots, predictions_all, _, M = root_music.narrowband(
#                     X=X, mode="sample"
#                 )
#             # If the amount of predictions is less than the amount of sources
#             predictions = add_random_predictions(M, predictions, algorithm)
#             # Calculate loss criterion
#             loss = criterion(predictions, doa * R2D)
#             loss_list.append(loss)
#             # Plot spectrum
#             if plot_spec and i == len(dataset.dataset) - 1:
#                 plot_spectrum(
#                     predictions=predictions_all,
#                     true_DOA=doa[0] * R2D,
#                     roots=roots,
#                     algorithm=algorithm.upper(),
#                     figures=figures,
#                 )
#         # MUSIC algorithms
#         elif "music" in algorithm:
#             music = MUSIC(system_model)
#             if algorithm.startswith("bb"):
#                 # Broadband MUSIC
#                 predictions, spectrum, M = music.broadband(X=X)
#             elif algorithm.startswith("sps"):
#                 # Spatial smoothing
#                 predictions, spectrum, M = music.narrowband(
#                     X=X, mode="spatial_smoothing"
#                 )
#             elif algorithm.startswith("music"):
#                 # Conventional
#                 predictions, spectrum, M = music.narrowband(X=X, mode="sample")
#             # If the amount of predictions is less than the amount of sources
#             predictions = add_random_predictions(M, predictions, algorithm)
#             # Calculate loss criterion
#             loss = criterion(predictions, doa * R2D)
#             loss_list.append(loss)
#             # Plot spectrum
#             if plot_spec and i == len(dataset.dataset) - 1:
#                 plot_spectrum(
#                     predictions=predictions,
#                     true_DOA=doa * R2D,
#                     system_model=system_model,
#                     spectrum=spectrum,
#                     algorithm=algorithm.upper(),
#                     figures=figures,
#                 )
#
#         # ESPRIT algorithms
#         elif "esprit" in algorithm:
#             esprit = Esprit(system_model)
#             if algorithm.startswith("sps"):
#                 # Spatial smoothing
#                 predictions, M = esprit.narrowband(X=X, mode="spatial_smoothing")
#             else:
#                 # Conventional
#                 predictions, M = esprit.narrowband(X=X, mode="sample")
#             # If the amount of predictions is less than the amount of sources
#             predictions = add_random_predictions(M, predictions, algorithm)
#             # Calculate loss criterion
#             loss = criterion(predictions, doa * R2D)
#             loss_list.append(loss)
#
#         # MVDR algorithm
#         elif algorithm.startswith("mvdr"):
#             mvdr = MVDR(system_model)
#             # 获取角度信息
#
#             angels_deg = np.rad2deg(mvdr._angels)  # 转换为度数
#             # Conventional
#             _, spectrum = mvdr.narrowband(X=X, mode="sample")#by j 让mvdr的比较更公平
#             spectrum_norm = spectrum[7500:10500] / np.max(spectrum[7500:10500]) #by j
#             peaks = argrelextrema(spectrum_norm, np.greater)[0]
#             peak_values = spectrum_norm[peaks]
#             sorted_indices = np.argsort(peak_values)[::-1]  # 降序
#             sorted_peaks = peaks[sorted_indices]
#             top_peaks = sorted_peaks[:2]+7500  # 取前两个最大峰值  by j
#             predicted_doas = angels_deg[top_peaks]
#             loss = criterion(predicted_doas, doa * R2D)  # Assume predictions are in degrees, multiply doa by R2D to match
#             loss_list.append(loss)
#             # Plot spectrum
#             if plot_spec and i == len(dataset.dataset) - 1:
#                 plot_spectrum(
#                     predictions=None,
#                     true_DOA=doa * R2D,
#                     system_model=system_model,
#                     spectrum=spectrum_norm,
#                     algorithm=algorithm.upper(),
#                     figures=figures,
#                 )
#         else:
#             raise Exception(
#                 f"evaluate_augmented_model: Algorithm {algorithm} is not supported."
#             )
#     return np.mean(loss_list)

def evaluate_model_based(
    dataset: list,
    system_model,
    criterion,  # RMSPE or similar
    plot_spec=False,
    algorithm: str = "music",
    figures: dict = None,
):
    """
    Evaluate different model-based algorithms on a given dataset, with loss and accuracy metrics.

    Args:
        dataset (list): The evaluation dataset.
        system_model (SystemModel): The system model for the algorithms.
        criterion: The loss criterion for evaluation (e.g., RMSPE).
        plot_spec (bool): Whether to plot the spectrum for the algorithms. Defaults to False.
        algorithm (str): The algorithm to use (e.g., "music", "mvdr", "esprit", "r-music"). Defaults to "music".
        figures (dict): Dictionary containing figure objects for plotting. Defaults to None.

    Returns:
        tuple: (average_loss, accuracy) - Average evaluation loss and prediction accuracy.

    Raises:
        Exception: If the algorithm is not supported.
    """
    # 初始化参数
    loss_list = []
    correct_predictions = 0  # 正确预测的样本数
    total_samples = 0
    R2D = 180 / np.pi  # 弧度转度数（假设未定义时在此定义）

    for i, data in enumerate(dataset):
        X, doa = data
        X = X[0]  # 取第一个样本
        total_samples += 1  # 单样本假设，doa为[1, num_sources]

        # Root-MUSIC算法
        if "r-music" in algorithm:
            root_music = RootMUSIC(system_model)
            if algorithm.startswith("sps"):
                predictions, roots, predictions_all, _, M = root_music.narrowband(X=X, mode="spatial_smoothing")
            else:
                predictions, roots, predictions_all, _, M = root_music.narrowband(X=X, mode="sample")
            predictions = add_random_predictions(M, predictions, algorithm)
            loss = criterion(predictions, doa * R2D)
            loss_list.append(loss)

            if plot_spec:
                plot_spectrum(predictions=predictions_all, true_DOA=doa[0] * R2D, roots=roots, algorithm=algorithm.upper(), figures=figures,sample_idx = i)

        # MUSIC算法
        elif "music" in algorithm:
            music = MUSIC(system_model)
            if algorithm.startswith("bb"):
                predictions, spectrum, M = music.broadband(X=X)
            elif algorithm.startswith("sps"):
                predictions, spectrum, M = music.narrowband(X=X, mode="spatial_smoothing")
            elif algorithm.startswith("music"):
                predictions, spectrum, M = music.narrowband(X=X, mode="sample")
            predictions = add_random_predictions(M, predictions, algorithm)
            loss = criterion(predictions, doa * R2D)
            loss_list.append(loss)

            if plot_spec and i == len(dataset) - 1:
                plot_spectrum(predictions=predictions, true_DOA=doa * R2D, system_model=system_model, spectrum=spectrum, algorithm=algorithm.upper(), figures=figures)

        # ESPRIT算法
        elif "esprit" in algorithm:
            esprit = Esprit(system_model)
            if algorithm.startswith("sps"):
                predictions, M = esprit.narrowband(X=X, mode="spatial_smoothing")
            else:
                predictions, M = esprit.narrowband(X=X, mode="sample")
            predictions = add_random_predictions(M, predictions, algorithm)
            loss = criterion(predictions, doa * R2D)
            loss_list.append(loss)

        # MVDR算法
        elif algorithm.startswith("mvdr"):
            mvdr = MVDR(system_model)
            angels_deg = np.rad2deg(mvdr._angels)  # 角度转为度数
            _, spectrum = mvdr.narrowband(X=X, mode="sample")
            spectrum_norm = spectrum[7500:10500] / np.max(spectrum[7500:10500])
            peaks = argrelextrema(spectrum_norm, np.greater)[0]
            peak_values = spectrum_norm[peaks]
            sorted_indices = np.argsort(peak_values)[::-1]
            sorted_peaks = peaks[sorted_indices]
            top_peaks = sorted_peaks[:2] + 7500  # 前两个峰值
            predicted_doas = angels_deg[top_peaks]
            loss = criterion(predicted_doas, doa * R2D)
            loss_list.append(loss)

        if plot_spec:
            plot_spectrum(predictions=None, true_DOA=doa * R2D, system_model=system_model, spectrum=spectrum_norm, algorithm=algorithm.upper(), figures=figures,sample_idx=i)

        # else:
        #     raise Exception(f"evaluate_augmented_model: Algorithm {algorithm} is not supported.")

        # 计算准确率
        pred_angles = predictions if "mvdr" not in algorithm else predicted_doas  # MVDR使用predicted_doas
        true_angles = (doa * R2D).flatten()  # 转换为度数，展平

        # 只考虑有效角度（假设无效角度为-1）
        valid_true = true_angles[true_angles != -1]
        valid_pred = pred_angles[:len(valid_true)]  # 截取匹配数量的预测角度
        if len(valid_pred) < 2:
            continue  # 如果预测角度少于2个，跳过当前样本

        if len(valid_true) == 0:  # 无有效目标
            continue
        elif len(valid_true) == 1:  # 单目标
            if abs(valid_true[0] - valid_pred[0]) <= 2:
                correct_predictions += 1
        else:  # 双目标，检查两种顺序
            # 计算两种顺序的每个角度差值
            diff1_0 = abs(valid_pred[0] - valid_true[0])  # 顺序1：pred[0] vs true[0]
            diff1_1 = abs(valid_pred[1] - valid_true[1])  # 顺序1：pred[1] vs true[1]
            diff2_0 = abs(valid_pred[0] - valid_true[1])  # 顺序2：pred[0] vs true[1]
            diff2_1 = abs(valid_pred[1] - valid_true[0])  # 顺序2：pred[1] vs true[0]

            # 检查是否存在一种顺序，每对差值都小于2°
            if (diff1_0 <= 1 and diff1_1 <= 1) or (diff2_0 <= 1 and diff2_1 <= 1):
                correct_predictions += 1.0

    # 计算平均损失和准确率
    average_loss = np.mean(loss_list) if loss_list else 0.0
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0

    return average_loss, accuracy
def add_random_predictions(M: int, predictions: np.ndarray, algorithm: str):
    """
    Add random predictions if the number of predictions is less than the number of sources.

    Args:
        M (int): The number of sources.
        predictions (np.ndarray): The predicted DOA values.
        algorithm (str): The algorithm used.

    Returns:
        np.ndarray: The updated predictions with random values.

    """
    # Convert to np.ndarray array
    if isinstance(predictions, list):
        predictions = np.array(predictions)
    while predictions.shape[0] < M:
        # print(f"{algorithm}: cant estimate M sources")
        predictions = np.insert(
            predictions, 0, np.round(np.random.rand(1) * 180, decimals=2) - 90.00
        )
    return predictions


def evaluate(
    model: nn.Module,
    model_type: str,
    model_test_dataset: list,
    generic_test_dataset: list,
    criterion: nn.Module,
    subspace_criterion,
    system_model,
    figures: dict,
    plot_spec: bool = True,
    augmented_methods: list = None,
    subspace_methods: list = None,
    training_params=None):
    """
    Wrapper function for model and algorithm evaluations.

    Parameters:
        model (nn.Module): The DNN model.
        model_type (str): Type of the model.
        model_test_dataset (list): Test dataset for the model.
        generic_test_dataset (list): Test dataset for generic subspace methods.
        criterion (nn.Module): Loss criterion for (DNN) model evaluation.
        subspace_criterion: Loss criterion for subspace method evaluation.
        system_model: instance of SystemModel.
        figures (dict): Dictionary to store figures.
        plot_spec (bool, optional): Whether to plot spectrums. Defaults to True.
        augmented_methods (list, optional): List of augmented methods for evaluation.
            Defaults to None.
        subspace_methods (list, optional): List of subspace methods for evaluation.
            Defaults to None.

    Returns:
        None
    """
    # Set default methods for SubspaceNet augmentation
    if not isinstance(augmented_methods, list) and model_type.startswith("SubspaceNet"):
        augmented_methods = [
            # "mvdr",
            "r-music",
            "esprit",
            # "music",
        ]
    # Set default model-based subspace methods
    if not isinstance(subspace_methods, list):
        subspace_methods = [
            # "esprit",
             # "music",
            # "r-music",
             "mvdr",
            #  "sps-r-music",
            #  "sps-esprit",
            # "sps-music"
            # "bb-music",
        ]
    # Evaluate SubspaceNet + differentiable algorithm performances
    # model_test_loss = evaluate_dnn_model(
    #     model=model,
    #     dataset=model_test_dataset,
    #     criterion=criterion,
    #     plot_spec=plot_spec,
    #     figures=figures,
    #     model_type=model_type,
    # )
    if training_params.model_type.startswith("My_transform_Model"):
        model_test_loss, acc = evaluate_transformer_model(
            model=model,
            dataset=model_test_dataset,
            criterion=criterion,
            plot_spec=plot_spec,
            figures=figures,
            model_type=model_type,
        )
    elif training_params.model_type.startswith("DeepCNN"):
        model_test_loss = evaluate_dnn_model(
            model=model,
            dataset=model_test_dataset,
            criterion=criterion,
            plot_spec=plot_spec,
            figures=figures,
            model_type=model_type,
        )

    print(f"{model_type} Test loss = {model_test_loss*R2D}")
    print(f"{model_type} accuracy = {acc}")
    # Evaluate SubspaceNet augmented methods      测试DNN 和DA 注释掉
    # for algorithm in augmented_methods:
    #     loss = evaluate_augmented_model(
    #         model=model,
    #         dataset=model_test_dataset,
    #         system_model=system_model,
    #         criterion=subspace_criterion,
    #         algorithm=algorithm,
    #         plot_spec=plot_spec,
    #         figures=figures,
    #     )
    #     print("augmented {} test loss = {}".format(algorithm, loss))
    # Evaluate classical subspace methods
    for algorithm in subspace_methods:
        loss, accuracy = evaluate_model_based(
            generic_test_dataset,
            system_model,
            criterion=subspace_criterion,
            plot_spec=plot_spec,
            algorithm=algorithm,
            figures=figures,
        )
        print("{} test loss = {}".format(algorithm.lower(), loss*R2D))
        print(f" {algorithm} accuracy = {accuracy}")
