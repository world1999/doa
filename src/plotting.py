"""
Subspace-Net

Details
----------
Name: plotting.py
Authors: D. H. Shmuel
Created: 01/10/21
Edited: 29/06/23

Purpose
----------
This module provides functions for plotting subspace methods spectrums,
like and RootMUSIC, MUSIC, and also beam patterns of MVDR.
 
Functions:
----------

plot_spectrum(predictions: np.ndarray, true_DOA: np.ndarray, system_model=None,
    spectrum: np.ndarray =None, roots: np.ndarray =None, algorithm:str ="music",
    figures:dict = None): Wrapper spectrum plotter based on the algorithm.
plot_music_spectrum(system_model, figures: dict, spectrum: np.ndarray, algorithm: str):
    Plot the MUSIC spectrum.
plot_root_music_spectrum(roots: np.ndarray, predictions: np.ndarray,
    true_DOA: np.ndarray, algorithm: str): Plot the Root-MUSIC spectrum.
plot_mvdr_spectrum(system_model, figures: dict, spectrum: np.ndarray,
    true_DOA: np.ndarray, algorithm: str): Plot the MVDR spectrum.
initialize_figures(void): Generates template dictionary containing figure objects for plotting multiple spectrums.


"""
import scipy.signal
# Imports
from matplotlib import pyplot as plt
import numpy as np
import torch
from scipy.signal import argrelextrema

from src.methods import MUSIC, RootMUSIC, MVDR
from src.system_model import SystemModelParams
from src.utils import R2D

def plot_spectrum(system_model_params: SystemModelParams, predictions: np.ndarray, true_DOA: np.ndarray, system_model=None,
    spectrum: np.ndarray =None, roots: np.ndarray =None, algorithm:str ="music",
    figures:dict = None, sample_idx: int = 0):
  """
  Wrapper spectrum plotter based on the algorithm.

  Args:
      predictions (np.ndarray): The predicted DOA values.
      true_DOA (np.ndarray): The true DOA values.
      system_model: The system model.
      spectrum (np.ndarray): The spectrum values.
      roots (np.ndarray): The roots for Root-MUSIC algorithm.
      algorithm (str): The algorithm used.
      figures (dict): Dictionary containing figure objects for plotting.

  Raises:
      Exception: If the algorithm is not supported.

  """
  # Convert predictions to 1D array
  if isinstance(predictions, (np.ndarray, list, torch.Tensor)):
    predictions = np.squeeze(np.array(predictions))
  # Plot MUSIC spectrums
  if "music" in algorithm.lower() and not ("r-music" in algorithm.lower()):
    plot_music_spectrum(system_model, figures, spectrum, algorithm)
  elif "mvdr" in algorithm.lower():
    plot_mvdr_spectrum(system_model, figures, spectrum, true_DOA, algorithm,sample_idx)
  elif "r-music" in algorithm.lower():
    plot_root_music_spectrum(roots, predictions, true_DOA, algorithm)
  elif "deepcnn" in algorithm.lower():
    # plot_DeepCNN_spectrum(predictions, true_DOA, roots, algorithm)  # 参数顺序调整
    plot_DeepCNN_spectrum1(system_model_params,predictions, true_DOA, roots, algorithm, figures, sample_idx)  # 直角坐标系
  elif "my_transform_model" in algorithm.lower():
    # plot_DeepCNN_spectrum(predictions, true_DOA, roots, algorithm)  # 参数顺序调整
    plot_My_transform_Model_spectrum(system_model_params,predictions, true_DOA, roots, algorithm, figures, sample_idx)  # 直角坐标系
  else:
    raise Exception(f"evaluate_augmented_model: Algorithm {algorithm} is not supported.")


def plot_My_transform_Model_spectrum(system_model_params: SystemModelParams,predictions: np.ndarray, true_DOA: np.ndarray,
                           roots: np.ndarray, algorithm: str, figures: dict = None,sample_idx: int = 0):
    """
    在直角坐标系绘制DeepCNN谱图并与MVDR比较
    """
    # 打印真实DOA
    unique_true_doa = np.unique(true_DOA)
    print(f"[transform] True DOAs: {unique_true_doa}")
    if figures is None:
        figures = {}
    # 为每个样本创建独立的 comparison_key 键
    comparison_key = f"comparison_{sample_idx}"
    if comparison_key not in figures:
        figures[comparison_key] = {'fig': None, 'ax': None}

    # # 初始化比较图容器
    # if "comparison_key" not in figures:
    #     figures["comparison_key"] = {'fig': None, 'ax': None}
    # 创建图形对象
    if figures[comparison_key]["fig"] is None:
        figures[comparison_key]["fig"] = plt.figure(figsize=(4, 2.5))
        figures[comparison_key]["ax"] = figures[comparison_key]["fig"].add_subplot(111)
    ax = figures[comparison_key]["ax"]
    predictions_norm=predictions/ np.max(predictions)


    # 生成角度坐标
    angles = np.linspace(-60, 60, system_model_params.grid_size)

    # 绘制DeepCNN谱线
    line_cnn, = ax.plot(angles, predictions_norm ,
                        color='#1f77b4',
                        linewidth=2,
                        alpha=0.8,
                        label='transform Spectrum')

    # 使用改进后的峰值检测函数
    selected_peaks, peak_angles = detect_top_peaks(
        predictions_norm,
        angles,
        min_distance=2,
        top_k=2
    )
    print(f"[transform] Predicted DOAs: {peak_angles}")
    # 绘制筛选后的峰值点
    for i, idx in enumerate(selected_peaks):
        ax.scatter(
            angles[idx],  # 峰对应的角度
            predictions_norm[idx],  # 峰对应的强度值
            color=line_cnn.get_color(),
            marker='^',
            s=100,
            edgecolor='k',
            zorder=5,
            label='transform Predictions' if i == 0 else None  # 仅第一个点添加图例
        )
    # 标记最大的两个值
    # top_two_indices = np.argsort(predictions_norm)[::-1][:2]
    # peak_angles = angles[top_two_indices]
    # for i, p in enumerate(peak_angles):
    #     ax.scatter(p, predictions_norm[np.argmin(np.abs(angles - p))],
    #                color=line_cnn.get_color(),
    #                marker='^',
    #                s=100,
    #                edgecolor='k',
    #                zorder=5,
    #                label='CNN Predictions' if i == 0 else None)

    # # 标记预测峰值
    # peaks = argrelextrema(predictions_norm, np.greater)[0]
    # peak_angles = angles[peaks]  # 将索引转换为角度值
    # for p in peak_angles[:2]:  # 假设最多两个信号
    #     ax.scatter(p, predictions_norm[np.argmin(np.abs(angles - p))],
    #                color=line_cnn.get_color(),
    #                marker='^',
    #                s=100,
    #                edgecolor='k',
    #                zorder=5,
    #                label='CNN Predictions' if p == peak_angles[0] else None)

    # if roots is not None:
    #     print(f"[DeepCNN] Predicted DOAs: {roots}")
    #     for peak in roots:
    #         idx = np.abs(angles - peak).argmin()
    #         ax.scatter(angles[idx], predictions_norm[idx],
    #                    color=line_cnn.get_color(),
    #                    marker='^',
    #                    s=100,
    #                    edgecolor='k',
    #                    zorder=5,
    #                    label='CNN Predictions')

    # 标记真实DOA（只添加一次图例）
    for i, doa in enumerate(np.unique(true_DOA)):
        ax.axvline(doa, color='#d62728', linestyle='--', linewidth=2,
                   label='True DOA' if i == 0 else None)




    # 配置坐标轴

    ax.set_xlim(-20, 20)
    ax.set_xticks(np.arange(-20, 21, 5))
    ax.set_xlabel("Azimuth Angle [deg]", fontsize=12)
    ax.set_ylabel("Normalized Amplitude", fontsize=12)
    ax.grid(True, alpha=0.4)
    # 保存当前样本的图
    if figures[comparison_key]["fig"] is not None:
        # ax = figures[comparison_key]["ax"]
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax.set_ylim(0, 1.2)
        ax.legend(unique_labels.values(), unique_labels.keys(),
                  loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        figures[comparison_key]["fig"].subplots_adjust(right=0.75)

        from pathlib import Path
        save_dir = Path("data/spectrums")
        save_dir.mkdir(parents=True, exist_ok=True)
        # figures[comparison_key]["fig"].savefig(
        #     save_dir / f"cnn_mvdr_comparison_{sample_idx}.png",
        #     bbox_inches='tight',
        #     dpi=300
        # )
        # plt.close(figures[comparison_key]["fig"])



    return figures


def plot_DeepCNN_spectrum1(system_model_params: SystemModelParams,predictions: np.ndarray, true_DOA: np.ndarray,
                           roots: np.ndarray, algorithm: str, figures: dict = None, sample_idx: int = 0):
    """
    在直角坐标系绘制DeepCNN谱图并与MVDR比较（升级版）
    新增功能：
    - 支持多样本独立绘图容器
    - 自动保存标准化谱图
    - 增强坐标轴和标签一致性
    """
    # 数据预处理
    unique_true_doa = np.unique(true_DOA)
    print(f"[DeepCNN] True DOAs: {unique_true_doa}")

    # 初始化图形容器
    if figures is None:
        figures = {}
    comparison_key = f"comparison_{sample_idx}"
    if comparison_key not in figures:
        figures[comparison_key] = {'fig': None, 'ax': None}

    # 创建图形对象
    if figures[comparison_key]["fig"] is None:
        figures[comparison_key]["fig"] = plt.figure(figsize=(4, 2.5))
        figures[comparison_key]["ax"] = figures[comparison_key]["fig"].add_subplot(111)
    ax = figures[comparison_key]["ax"]

    # 数据规范化处理
    predictions_norm = predictions / np.max(predictions)
    angles = np.linspace(-60, 60, system_model_params.grid_size)

    # 核心绘图逻辑
    line_cnn, = ax.plot(angles, predictions_norm,
                        color='#2ca02c',  # 修改颜色以示区分
                        linewidth=2,
                        alpha=0.8,
                        label='DeepCNN Spectrum')

    # 改进版峰值检测
    selected_peaks, peak_angles = detect_top_peaks(
        predictions_norm,
        angles,
        min_distance=2,
        top_k=2
    )
    print(f"[DeepCNN] Predicted DOAs: {peak_angles}")

    # 可视化增强
    for i, idx in enumerate(selected_peaks):
        ax.scatter(
            angles[idx],
            predictions_norm[idx],
            color=line_cnn.get_color(),
            marker='s',  # 改用方形标记
            s=80,
            edgecolor='w',
            zorder=5,
            label='DeepCNN Predictions' if i == 0 else None
        )

    # 真实DOA标注
    for i, doa in enumerate(np.unique(true_DOA)):
        ax.axvline(doa, color='#ff7f0e',  # 修改警示色
                   linestyle=':',  # 修改线型
                   linewidth=2.5,
                   label='Ground Truth' if i == 0 else None)

    # 布局标准化
    ax.set_xlim(-20, 20)
    ax.set_xticks(np.arange(-20, 21, 5))
    ax.set_xlabel("Azimuth  Angle [deg]", fontsize=12)
    ax.set_ylabel("Normalized  Power", fontsize=12)  # 修改坐标轴标签
    ax.grid(True, alpha=0.4)
    ax.set_ylim(0, 1.2)

    # 图例和保存逻辑
    if figures[comparison_key]["fig"] is not None:
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax.legend(unique_labels.values(), unique_labels.keys(),
                  loc='upper left',
                  bbox_to_anchor=(1.05, 1),
                  borderaxespad=0.,
                  frameon=False)

        figures[comparison_key]["fig"].subplots_adjust(right=0.75)

        # 文件保存路径管理
        from pathlib import Path
        save_dir = Path("data/spectrum_comparisons")
        save_dir.mkdir(parents=True, exist_ok=True)
        figures[comparison_key]["fig"].savefig(
            save_dir / f"DeepCNN_spectrum_{sample_idx}.png",  # 修改文件名
            bbox_inches='tight',
            dpi=300,
            transparent=True  # 添加透明背景
        )

    return figures
# def plot_DeepCNN_spectrum1(predictions: np.ndarray, true_DOA: np.ndarray,
#                            roots: np.ndarray, algorithm: str, figures: dict = None,sample_idx: int = 0):
#     """
#     在直角坐标系绘制DeepCNN谱图并与MVDR比较
#     """
#     # 打印真实DOA
#     unique_true_doa = np.unique(true_DOA)
#     print(f"[DeepCNN] True DOAs: {unique_true_doa}")
#     if figures is None:
#         figures = {}
#     # 为每个样本创建独立的 comparison_key 键
#     comparison_key = f"comparison_{sample_idx}"
#     # 初始化比较图容器
#     if comparison_key not in figures:
#         figures[comparison_key] = {'fig': None, 'ax': None}
#     # 创建图形对象
#     if figures[comparison_key]["fig"] is None:
#         figures[comparison_key]["fig"] = plt.figure(figsize=(10, 6))
#         figures[comparison_key]["ax"] = figures[comparison_key]["fig"].add_subplot(111)
#     ax = figures[comparison_key]["ax"]
#     predictions_norm=predictions/ np.max(predictions)
#
#
#     # 生成角度坐标
#     angles = np.linspace(-60, 60, 241)
#
#     # 绘制DeepCNN谱线
#     line_cnn, = ax.plot(angles, predictions_norm ,
#                         color='#1f77b4',
#                         linewidth=2,
#                         alpha=0.8,
#                         label='DeepCNN Spectrum')
#
#     # 使用改进后的峰值检测函数
#     selected_peaks, peak_angles = detect_top_peaks(
#         predictions_norm,
#         angles,
#         min_distance=2,
#         top_k=2
#     )
#     print(f"[DeepCNN] Predicted DOAs: {peak_angles}")
#     # 绘制筛选后的峰值点
#     for i, idx in enumerate(selected_peaks):
#         ax.scatter(
#             angles[idx],  # 峰对应的角度
#             predictions_norm[idx],  # 峰对应的强度值
#             color=line_cnn.get_color(),
#             marker='^',
#             s=100,
#             edgecolor='k',
#             zorder=5,
#             label='CNN Predictions' if i == 0 else None  # 仅第一个点添加图例
#         )
#     # 标记最大的两个值
#     # top_two_indices = np.argsort(predictions_norm)[::-1][:2]
#     # peak_angles = angles[top_two_indices]
#     # for i, p in enumerate(peak_angles):
#     #     ax.scatter(p, predictions_norm[np.argmin(np.abs(angles - p))],
#     #                color=line_cnn.get_color(),
#     #                marker='^',
#     #                s=100,
#     #                edgecolor='k',
#     #                zorder=5,
#     #                label='CNN Predictions' if i == 0 else None)
#
#     # # 标记预测峰值
#     # peaks = argrelextrema(predictions_norm, np.greater)[0]
#     # peak_angles = angles[peaks]  # 将索引转换为角度值
#     # for p in peak_angles[:2]:  # 假设最多两个信号
#     #     ax.scatter(p, predictions_norm[np.argmin(np.abs(angles - p))],
#     #                color=line_cnn.get_color(),
#     #                marker='^',
#     #                s=100,
#     #                edgecolor='k',
#     #                zorder=5,
#     #                label='CNN Predictions' if p == peak_angles[0] else None)
#
#     # if roots is not None:
#     #     print(f"[DeepCNN] Predicted DOAs: {roots}")
#     #     for peak in roots:
#     #         idx = np.abs(angles - peak).argmin()
#     #         ax.scatter(angles[idx], predictions_norm[idx],
#     #                    color=line_cnn.get_color(),
#     #                    marker='^',
#     #                    s=100,
#     #                    edgecolor='k',
#     #                    zorder=5,
#     #                    label='CNN Predictions')
#
#     # 标记真实DOA（只添加一次图例）
#     for i, doa in enumerate(np.unique(true_DOA)):
#         ax.axvline(doa, color='#d62728', linestyle='--', linewidth=2,
#                    label='True DOA' if i == 0 else None)
#
#
#
#
#     # 配置坐标轴
#
#     ax.set_xlim(-90, 90)
#     ax.set_xticks(np.arange(-90, 91, 30))
#     ax.set_xlabel("Azimuth Angle [deg]", fontsize=12)
#     ax.set_ylabel("Normalized Amplitude", fontsize=12)
#     ax.grid(True, alpha=0.4)
#
#
#     return figures
def plot_mvdr_spectrum(system_model, figures: dict, spectrum: np.ndarray,
                       true_DOA: np.ndarray, algorithm: str,sample_idx: int = 0):
    """
    在直角坐标系绘制MVDR谱图
    """
    # 打印真实DOA
    unique_true_doa = np.unique(true_DOA)
    print(f"[MVDR] True DOAs: {unique_true_doa}")
    # 获取角度信息
    mvdr = MVDR(system_model)
    angels_deg = np.rad2deg(mvdr._angels)[7500:10500]  # 转换为度数
    # 为每个样本创建独立的 comparison 键
    comparison_key = f"comparison_{sample_idx}"
    # 初始化比较图容器
    if comparison_key not in figures:
        figures[comparison_key] = {'fig': None, 'ax': None}
    # 创建图形对象
    if figures[comparison_key]["fig"] is None:
        figures[comparison_key]["fig"] = plt.figure(figsize=(4, 2.5))
        figures[comparison_key]["ax"] = figures[comparison_key]["fig"].add_subplot(111)
    ax = figures[comparison_key]["ax"]

    # 绘制MVDR谱线
    spectrum_norm = spectrum / np.max(spectrum)
    line_mvdr, = ax.plot(angels_deg, spectrum_norm,
                         color='#ff7f0e',
                         linewidth=2,
                         alpha=0.8,
                         label='MVDR Spectrum')

    # 标记谱峰（示例方法，需根据实际峰值检测逻辑调整）
    peaks = argrelextrema(spectrum_norm, np.greater)[0]
    if len(peaks) > 0:
        peak_values = spectrum_norm[peaks]
        sorted_indices = np.argsort(peak_values)[::-1]  # 降序
        sorted_peaks = peaks[sorted_indices]
        top_peaks = sorted_peaks[:2]  # 取前两个最大峰值
        predicted_doas = angels_deg[top_peaks]
        print(f"[MVDR] Predicted DOAs: {predicted_doas}")

        # 绘制峰值点
        for i, p in enumerate(top_peaks):
            ax.scatter(angels_deg[p], spectrum_norm[p],
                       color=line_mvdr.get_color(),
                       marker='o',
                       s=80,
                       edgecolor='k',
                       zorder=5,
                       label='MVDR Peaks'if i == 0 else None)
    else:
        print("[MVDR] No peaks detected.")






    #
    #     # 调整图例位置到外部
    # ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    # # 调整图形布局，确保图例不会被裁剪
    # plt.tight_layout(rect=[0, 0, 0.85, 1])



    # 同步坐标轴设置（避免重复设置）
    ax.set_xlim(-20, 20)
    ax.set_xticks(np.arange(-20, 21, 5))
    ax.set_ylabel("Normalized Amplitude", fontsize=12)
    # 保存当前样本的图
    if figures[comparison_key]["fig"] is not None:
        ax = figures[comparison_key]["ax"]
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax.set_ylim(0, 1.2)
        ax.legend(unique_labels.values(), unique_labels.keys(),
                  loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        figures[comparison_key]["fig"].subplots_adjust(right=0.75)

        from pathlib import Path
        save_dir = Path("data/spectrums")
        save_dir.mkdir(parents=True, exist_ok=True)
        figures[comparison_key]["fig"].savefig(
            save_dir / f"cnn_mvdr_comparison_{sample_idx}.png",
            bbox_inches='tight',
            dpi=300
        )
        plt.close(figures[comparison_key]["fig"])
        figures[comparison_key]["fig"] = None  # 清空以避免内存累积

    return figures
# def plot_DeepCNN_spectrum1(predictions: np.ndarray, true_DOA: np.ndarray,
#                           roots: np.ndarray, algorithm: str):
#     """
#     Plot DeepCNN probability spectrum with annotations.
#
#     Args:
#         predictions (np.ndarray):  Probability distribution over angles (361 points)
#         true_DOA (np.ndarray):  True DOA angles in degrees
#         roots (np.ndarray):  Predicted peak angles in degrees
#         algorithm (str): Algorithm identifier for naming
#     """
#     # 创建图形和坐标系
#     plt.style.use('seaborn')
#     fig = plt.figure(figsize=(10, 4))
#     ax = fig.add_subplot(111)
#
#     # 生成角度坐标轴
#     angles = np.linspace(-90, 90, 361)
#
#     # 绘制概率谱曲线
#     ax.plot(angles, predictions, linewidth=1.5, color='#1f77b4',
#             label='Probability Spectrum')
#
#     # 标记真实DOA（红色虚线）
#     for doa in np.unique(true_DOA):
#         ax.axvline(doa, color='#d62728', linestyle='--', linewidth=1.5,
#                    label='True DOA' if doa == true_DOA[0] else None)
#
#     # 标记预测峰值（绿色三角形）
#     if roots is not None:
#         for peak in roots:
#             idx = np.abs(angles - peak).argmin()  # 寻找最近的角度索引
#             ax.scatter(angles[idx], predictions[idx], color='#2ca02c',
#                        marker='^', s=80, zorder=5,
#                        label='Predicted Peaks' if peak == roots[0] else None)
#
#     # 图形美化
#     ax.set_xlim(-90, 90)
#     ax.set_xticks(np.arange(-90, 91, 30))
#     ax.set_xlabel("Azimuth  Angle [deg]", fontsize=12)
#     ax.set_ylabel("Probability", fontsize=12)
#     ax.grid(True, alpha=0.4)
#
#     # 智能图例处理
#     handles, labels = ax.get_legend_handles_labels()
#     unique_labels = dict(zip(labels, handles))
#     ax.legend(unique_labels.values(), unique_labels.keys(),
#               loc='upper right', frameon=True)
#     plt.show()
#     from pathlib import Path
#     save_dir = Path("data/spectrums")
#     save_dir.mkdir(parents=True, exist_ok=True)
#
#     plt.savefig(save_dir / f"{algorithm}_linear_spectrum.png",
#                 bbox_inches='tight',
#                 dpi=300)
#     plt.close()
#
#     # # 保存图形
#     # plt.tight_layout()
#     # plt.savefig(f"data/spectrums/{algorithm}_spectrum.pdf",
#     #             bbox_inches='tight', dpi=300)
#     # plt.close()
# def plot_DeepCNN_spectrum(predictions: np.ndarray, true_DOA: np.ndarray,
#                           roots: np.ndarray, algorithm: str):
#     """
#     极坐标版DeepCNN概率谱可视化
#
#     主要改造点：
#     - 坐标系改为极坐标
#     - 角度映射优化
#     - 可视化元素极坐标适配
#     """
#     # ==================== 坐标系初始化 ====================
#     plt.style.use('seaborn')
#     fig = plt.figure(figsize=(10, 8))  # 增大画布尺寸
#     ax = fig.add_subplot(111, projection='polar')  # 核心修改：极坐标投影
#
#     # ==================== 角度映射逻辑 ====================
#     # 将-90~90度映射到270°~90°（保持雷达图传统布局）
#     theta = np.deg2rad(np.linspace(-90, 90, 361) + 90)  # 转换为0~180度后再转弧度
#
#     # ==================== 极坐标绘图逻辑 ====================
#     # 绘制概率谱曲线（调整颜色增强对比度）
#     ax.plot(theta, predictions,
#             linewidth=2,
#             color='#1f77b4',
#             alpha=0.8,
#             label='Probability Spectrum')
#
#     # 标记真实DOA（改为径向线）
#     for doa in np.unique(true_DOA):
#         radian = np.deg2rad(doa + 90)  # 角度映射转换
#         ax.axvline(radian,
#                    color='#d62728',
#                    linestyle='--',
#                    linewidth=2.2,
#                    alpha=0.9,
#                    label='True DOA' if doa == true_DOA[0] else None)
#
#     # 标记预测峰值（极坐标散点）
#     if roots is not None:
#         for peak in roots:
#             peak_rad = np.deg2rad(peak + 90)  # 角度转换
#             idx = np.abs(np.linspace(-90, 90, 361) - peak).argmin()
#             ax.scatter(peak_rad, predictions[idx],
#                        color='#2ca02c',
#                        marker='^',
#                        s=120,  # 增大标记尺寸
#                        edgecolor='k',
#                        zorder=10,
#                        label='Predicted Peaks' if peak == roots[0] else None)
#
#     # ==================== 极坐标美化 ====================
#     # 角度轴设置
#     ax.set_theta_offset(np.pi / 2)  # 0度指向正上方
#     ax.set_theta_direction(-1)  # 顺时针方向增加角度
#
#     # 刻度标签设置
#     ax.set_xticks(np.deg2rad(np.arange(-90, 91, 30) + 90))  # 转换为0~180度对应的弧度
#     ax.set_xticklabels([f'{ang}°' for ang in np.arange(-90, 91, 30)])
#
#     # 半径轴设置
#     ax.set_ylim(0, np.max(predictions) * 1.1)  # 自动缩放半径轴
#     ax.set_rlabel_position(45)  # 半径标签位置
#     ax.yaxis.grid(True, alpha=0.4, linestyle=':')
#
#     # 图例优化
#     handles, labels = ax.get_legend_handles_labels()
#     unique_labels = dict(zip(labels, handles))
#     ax.legend(unique_labels.values(), unique_labels.keys(),
#               loc='upper right',
#               bbox_to_anchor=(1.60, 1.60),  # 外置图例位置调整
#               frameon=True)
#
#     # ==================== 保存与清理 ====================
#     plt.title(f"DeepCNN  DOA Estimation - {algorithm}", pad=20)
#     plt.tight_layout()
#     plt.show()
#
#     # 确保保存路径存在
#     from pathlib import Path
#     save_dir = Path("data/spectrums")
#     save_dir.mkdir(parents=True, exist_ok=True)
#
#     plt.savefig(save_dir / f"{algorithm}_polar_spectrum.png",
#                 bbox_inches='tight',
#                 dpi=300)
#     plt.close()
def plot_music_spectrum(system_model, figures: dict, spectrum: np.ndarray, algorithm: str):
    """
    Plot the MUSIC spectrum.

    Args:
        system_model (SystemModel): The system model.
        figures (dict): Dictionary containing figure objects for plotting.
        spectrum (np.ndarray): The spectrum values.
        algorithm (str): The algorithm used.

    """
    # Initialize MUSIC instance
    music = MUSIC(system_model)
    angels_grid = music._angels * R2D
    # Initialize plot for spectrum
    if figures["music"]["fig"] == None:
      plt.style.use('default')
      figures["music"]["fig"] = plt.figure(figsize=(8, 6))
      # plt.style.use('plot_style.txt')
    if figures["music"]["ax"] == None:
      figures["music"]["ax"] = figures["music"]["fig"].add_subplot(111)
    # Set labels titles and limits
    figures["music"]["ax"].set_xlabel("Angels [deg]")
    figures["music"]["ax"].set_ylabel("Amplitude")
    figures["music"]["ax"].set_ylim([0.0, 1.01])
    # Apply normalization factor for multiple plots
    figures["music"]["norm factor"] = None
    if figures["music"]["norm factor"] != None:
      # Plot music spectrum
      figures["music"]["ax"].plot(angels_grid , spectrum / figures["music"]["norm factor"], label=algorithm)
    else:
      # Plot normalized music spectrum
      figures["music"]["ax"].plot(angels_grid , spectrum / np.max(spectrum), label=algorithm)
    # Set legend
    figures["music"]["ax"].legend()

# def plot_mvdr_spectrum(system_model, figures: dict, spectrum: np.ndarray,
#     true_DOA: np.ndarray, algorithm: str):
#     """
#     Plot the MVDR spectrum.
#
#     Args:
#         system_model (SystemModel): The system model.
#         figures (dict): Dictionary containing figure objects for plotting.
#         spectrum (np.ndarray): The spectrum values.
#         algorithm (str): The algorithm used.
#         true_DOA (np.ndarray): The true DOA values.
#
#     """
#     # Initialize MVDR instance
#     mvdr = MVDR(system_model)
#     # Initialize plot for spectrum
#     if figures["mvdr"]["fig"] == None:
#       plt.style.use('default')
#       figures["mvdr"]["fig"] = plt.figure(figsize=(8, 6))
#     if figures["mvdr"]["ax"] == None:
#       figures["mvdr"]["ax"] = figures["mvdr"]["fig"].add_subplot(111, polar=True)
#     # Set axis location and limits
#     figures["mvdr"]["ax"].set_theta_zero_location('N')
#     figures["mvdr"]["ax"].set_theta_direction(-1)
#     figures["mvdr"]["ax"].set_thetamin(-90)
#     figures["mvdr"]["ax"].set_thetamax(90)
#     figures["mvdr"]["ax"].set_ylim([0.0, 1.01])
#     # Plot normalized mvdr beam pattern
#     figures["mvdr"]["ax"].plot(mvdr._angels , spectrum / np.max(spectrum), label=algorithm)
#     # marker in "x" true DoA's
#     for doa in true_DOA[0]:
#       figures["mvdr"]["ax"].plot([doa * np.pi / 180], [1], marker='x', color="r", markersize=14)
#     # Set leagend
#     figures["mvdr"]["ax"].legend()

def plot_root_music_spectrum(roots: np.ndarray, predictions: np.ndarray,
    true_DOA: np.ndarray, algorithm: str):
    """
    Plot the Root-MUSIC spectrum.

    Args:
        roots (np.ndarray): The roots for Root-MUSIC polynomyal.
        predictions (np.ndarray): The predicted DOA values.
        true_DOA (np.ndarray): The true DOA values.
        algorithm (str): The algorithm used.

    """
    # Initialize figure
    plt.style.use('default')
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, polar=True)
    # Set axis location and limits
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetamin(90)
    ax.set_thetamax(-90)
    # plot roots ang angles 
    for i in range(len(predictions)):
      angle = predictions[i]
      r = np.abs(roots[i])
      ax.set_ylim([0, 1.2])
      ax.set_yticks([0, 1])
      ax.plot([0, angle * np.pi / 180], [0, r], marker='o')
    for doa in true_DOA:
      ax.plot([doa * np.pi / 180], [1], marker='x', color="r", markersize=14)
    ax.set_xlabel("Angels [deg]")
    ax.set_ylabel("Amplitude")
    plt.savefig("data/spectrums/{}_spectrum.pdf".format(algorithm), bbox_inches='tight')
    
def initialize_figures():
  """Generates template dictionary containing figure objects for plotting multiple spectrums.

  Returns:
      (dict): The figures dictionary
  """  
  figures = {"music"  : {"fig" : None, "ax" : None, "norm factor" : None},
            "r-music": {"fig" : None, "ax" : None},
            "esprit" : {"fig" : None, "ax" : None},
            "mvdr"   : {"fig" : None, "ax" : None, "norm factor" : None},
            "comparison_key": {"fig": None,"ax" : None, "norm factor" : None}}

  return figures
def detect_top_peaks(spectrum: np.ndarray, angles: np.ndarray, min_distance: float = 5, top_k: int = 2):
    """
    识别谱图中的峰值，并确保相邻峰值至少间隔 `min_distance` 度，同时返回前 `top_k` 个最强峰值。

    参数：
    - spectrum: np.ndarray，输入的谱图数据（功率或归一化谱）。
    - angles: np.ndarray，对应的角度数组（与spectrum等长）。
    - min_distance: float，相邻峰之间的最小角度间隔（单位：度）。
    - top_k: int，返回最大的 `top_k` 个峰值。

    返回：
    - selected_peaks: np.ndarray，最终筛选后的峰值索引。
    - peak_angles: np.ndarray，筛选后峰值对应的角度。
    """
    # 找出所有局部最大值
    peak_indices, _ = scipy.signal.find_peaks(spectrum)
    peak_angles = angles[peak_indices]

    # 按峰值强度排序（从大到小）
    sorted_indices = np.argsort(spectrum[peak_indices])[::-1]
    sorted_peaks = peak_indices[sorted_indices]
    sorted_angles = peak_angles[sorted_indices]

    # 筛选相邻至少相距 min_distance 的峰
    selected_peaks = []
    for i, peak in enumerate(sorted_peaks):
        peak_angle = sorted_angles[i]
        # 仅当新峰与已选峰相距大于 min_distance 时才添加
        if all(abs(peak_angle - angles[p]) > min_distance for p in selected_peaks):
            selected_peaks.append(peak)
        # 达到 top_k 个峰后退出
        if len(selected_peaks) >= top_k:
            break

    # 返回最终峰值索引及其角度
    return np.array(selected_peaks), angles[selected_peaks]