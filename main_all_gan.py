"""Subspace-Net main script 
    Details
    -------
    Name: main.py
    Authors: D. H. Shmuel
    Created: 01/10/21
    Edited: 30/06/23

    Purpose
    --------
    This script allows the user to apply the proposed algorithms,
    by wrapping all the required procedures and parameters for the simulation.
    This scripts calls the following functions:
        * create_dataset: For creating training and testing datasets 
        * training: For training DR-MUSIC model
        * evaluate_dnn_model: For evaluating subspace hybrid models

    This script requires that requirements.txt will be installed within the Python
    environment you are running this script in.

"""
# Imports
import sys
import torch
import os
import copy
import matplotlib.pyplot as plt
import warnings
from src.system_model import SystemModelParams
from src.signal_creation import *
from src.data_handler import *
from src.criterions import set_criterions
from src.training import *
from src.evaluation import evaluate
from src.plotting import initialize_figures
from pathlib import Path
from src.models import ModelGenerator, GAN_Model
from src.model_transform import My_transform_Model
from torch.utils.data import DataLoader, TensorDataset
# Initialization
warnings.simplefilter("ignore")
os.system("cls||clear")
plt.close("all")

if __name__ == "__main__":
    # Initialize paths
    external_data_path = Path.cwd() / "data"
    scenario_data_path = "uniform_bias_spacing"
    datasets_path = external_data_path / "datasets" / scenario_data_path
    simulations_path = external_data_path / "simulations"
    saving_path = external_data_path / "weights"
    # create folders if not exists
    datasets_path.mkdir(parents=True, exist_ok=True)
    (datasets_path / "train").mkdir(parents=True, exist_ok=True)
    (datasets_path / "test").mkdir(parents=True, exist_ok=True)
    datasets_path.mkdir(parents=True, exist_ok=True)
    simulations_path.mkdir(parents=True, exist_ok=True)
    saving_path.mkdir(parents=True, exist_ok=True)
    # Initialize time and date
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    dt_string_for_save = now.strftime("%d_%m_%Y_%H_%M")
    # Operations commands
    #评估
    commands = {
        "SAVE_TO_FILE": True,  # Saving results to file or present them over CMD
        "CREATE_DATA": True,  # Creating new dataset
        "LOAD_DATA": False,  # Loading data from exist dataset
        "LOAD_MODEL": False,  # Load specific model for training
        "TRAIN_MODEL": False,  # Applying training operation
        "SAVE_MODEL": True,  # Saving tuned model
        "EVALUATE_MODE": False,  # Evaluating desired algorithms
        "LOAD_GAN_MODEL": False,  # 新增GAN模型训练标志
        "TRAIN_GAN_MODEL": True,  # 新增GAN模型训练标志
        "SAVE_GAN_MODEL": True,  # 新增GAN模型保存标志
        "EVALUATE_GAN_MODE":False,  # 新增GAN模型评估标志
    }
    #训练
    # commands = {
    #     "SAVE_TO_FILE": True,  # Saving results to file or present them over CMD
    #     "CREATE_DATA": True,  # Creating new dataset
    #     "LOAD_DATA": False,  # Loading data from exist dataset
    #     "LOAD_MODEL": False,  # Load specific model for training
    #     "TRAIN_MODEL": True,  # Applying training operation
    #     "SAVE_MODEL": True,  # Saving tuned model
    #     "EVALUATE_MODE": True,  # Evaluating desired algorithms
    # }

    # # Saving simulation scores to external file
    # if commands["SAVE_TO_FILE"]:
    #     file_path = (
    #         simulations_path / "results" / "train_scores" / Path(dt_string_for_save + ".txt")
    #     )
    #     sys.stdout = open(file_path, "w")
    # Define system model parameters

    base_params = (  #by j1 start    训练模型用
        SystemModelParams()
        .set_parameter("N", 32)
        .set_parameter("M", 2)
        .set_parameter("T", 100)
        .set_parameter("grid_size", 121)  # 添加网格点参数
        .set_parameter("signal_type", "NarrowBand")
        .set_parameter("signal_nature", "non-coherent")
        .set_parameter("eta", 0)
        .set_parameter("bias", 0)
        .set_parameter("sv_noise_var", 0)
        .set_parameter("gap", 10)
    )
    system_model_params1= copy.deepcopy(base_params).set_parameter("snr", 0)  # 测试集生成
    test_mode = [4132351]#训练模型用，同时设置多个训练轮数
    grid_use = [121]
    # snr_values_use = [[-10, -9, -8, -7], [-10, -9, -8, -7, -6], [-10, -9, -8, -7, -3], [-10, -9, -8, -7, -6, -3]]
    low_snr_values_use = [[9]]#条件输入
    high_snr_values_use = [[10]]# 目标SNR
    for i, test_mode_snr in enumerate(test_mode):
        base_params = copy.deepcopy(base_params).set_parameter("grid_size", grid_use[i])
        system_model_params = copy.deepcopy(base_params).set_parameter("snr",
                                                                       test_mode_snr)  ##多个数据集的训练过程日志和模型记录的参数 -20 只做记录用
        # 定义需要遍历的snr值列表
        low_snr_values = low_snr_values_use[i] #[[-10]]
        high_snr_values = high_snr_values_use[i] #[[10]]

        # Generate model configuration
        model_config = (
            ModelGenerator()
            .set_model_type("GAN_Model")  #SubspaceNet  DeepCNN DA-MUSIC   DeepRootMUSIC My_transform_Model
            .set_diff_method("root_music")  # root_music esprit
            .set_tau(8)
            .set_model(system_model_params)
        )
        # Define samples size
        samples_size = 30000  # Overall dateset size
        train_test_ratio = 0.002  # training and testing datasets ratio
        # Sets simulation filename
        simulation_filename = get_simulation_filename(
            system_model_params=system_model_params, model_config=model_config
        )
        # Saving simulation scores to external file
        if commands["SAVE_TO_FILE"]:
            file_path = (
                    simulations_path / "results" / "train_scores" / Path(
                dt_string_for_save + f"train_{model_config.model_type}_{system_model_params.grid_size}_{samples_size}_model={system_model_params.snr}_gap={system_model_params.gap}.txt")
            )
            sys.stdout = open(file_path, "w")
        # Print new simulation intro
        print("------------------------------------")
        print("---------- New Simulation ----------")
        print("------------------------------------")
        print("date and time =", dt_string)
        print(f"train_snr_values = {low_snr_values}")
        print(f"modelname_SNR 值: {system_model_params.snr}")
        print(f'modelgrid_size: {system_model_params.grid_size}')
        # Initialize seed
        set_unified_seed()
        # Datasets creation
        if commands["CREATE_DATA"]:
            # Define which datasets to generate
            create_training_data = True  # Flag for creating training data
            create_testing_data = False  # Flag for creating test data
            print("Creating Data...")
            if create_training_data:
                # Generate training dataset
                # 生成参数组列表
                low_param_groups = generate_param_groups(low_snr_values, base_params)# base_params的复制替代品
                high_param_groups = generate_param_groups(high_snr_values, base_params)
                # 生成合并数据集
                # 生成合并数据集
                combined_dataset = generate_combined_data(low_param_groups + high_param_groups, base_params, model_config, samples_size, datasets_path)
                # train_dataset, _, _ = create_dataset(
                #     system_model_params=system_model_params,
                #     samples_size=samples_size,
                #     model_type=model_config.model_type,
                #     tau=model_config.tau,
                #     save_datasets=True,
                #     datasets_path=datasets_path,
                #     true_doa=None,
                #     phase="train",
                # )
            if create_testing_data:
                # Generate test dataset
                test_dataset, generic_test_dataset, samples_model = create_dataset(
                    system_model_params=system_model_params1,
                    samples_size=int(train_test_ratio * samples_size),
                    model_type=model_config.model_type,
                    tau=model_config.tau,
                    save_datasets=True,
                    datasets_path=datasets_path,
                    true_doa=None,
                    phase="test",
                )
        # Datasets loading
        elif commands["LOAD_DATA"]:
            (
                train_dataset,
                test_dataset,
                generic_test_dataset,
                samples_model,
            ) = load_datasets(
                system_model_params=system_model_params,
                model_type=model_config.model_type,
                samples_size=samples_size,
                datasets_path=datasets_path,
                train_test_ratio=train_test_ratio,
                is_training=True,  # 训练时记得在设置
            )

        # Training stage
        if commands["TRAIN_MODEL"] or commands["TRAIN_GAN_MODEL"]:
            # Assign the training parameters object
            if commands["TRAIN_MODEL"]:
                simulation_parameters = (
                    TrainingParams()
                    .set_batch_size(256)
                    .set_epochs(50)
                    .set_model(model=model_config)
                    .set_optimizer(optimizer="Adam", learning_rate=0.0001,
                                   weight_decay=1e-7)  #learning_rate=0.00001, weight_decay=1e-9
                    .set_training_dataset(combined_dataset)  #by j
                    .set_schedular(step_size=20, gamma=0.5)
                    .set_criterion()  #自动设置成nn.BCELoss()  非常关键，训练的时候要设置
                )
            
            if commands["TRAIN_GAN_MODEL"]:
              
                simulation_parameters = (
                    TrainingParams()
                    .set_batch_size(128)
                    .set_epochs(300)
                    .set_model(model=model_config)
                     .set_optimizer(optimizer="Adam", learning_rate=0.0001,
                                   weight_decay=1e-5)
                    .set_training_dataset(combined_dataset)  #by j
                    .set_schedular(step_size=30, gamma=0.5)
                    .set_criterion()
                )
            if commands["LOAD_MODEL"]:
                simulation_parameters.load_model(
                    loading_path=saving_path / "final_models" / simulation_filename
                )
            # Print training simulation details
            simulation_summary(
                system_model_params=base_params,
                model_type=model_config.model_type,
                parameters=simulation_parameters,
                phase="training",
            )
            # Perform simulation training and evaluation stages
            model, adv_loss_list, cls_loss_list, snr_loss_list = train_gan(
                system_model_params=base_params,
                training_parameters=simulation_parameters,
                model_name=simulation_filename,
                saving_path=saving_path,
            )
            # Save model weights
            if commands["SAVE_MODEL"]:
                # 确保目标目录存在 by j
                final_models_dir = saving_path / 'final_models'
                final_models_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    model.state_dict(),
                    final_models_dir / Path(simulation_filename),
                )
            # Plots saving
            if commands["SAVE_TO_FILE"]:
                # 确保路径存在
                outplot_path = Path(simulations_path) / 'results' / 'plot'
                outplot_path.mkdir(parents=True, exist_ok=True)
                plt.savefig(
                    outplot_path / Path(dt_string_for_save + r".png")
                )
            else:
                plt.show()

        # Evaluation stage
        if commands["EVALUATE_MODE"]:
            # Initialize figures dict for plotting
            figures = initialize_figures()
            # figures='comparison'#进行不同方法的对比
            # Define loss measure for evaluation
            criterion, subspace_criterion = set_criterions("rmse")  #训练的是时候是交叉熵，只有测试的时候用rmse
            # Load datasets for evaluation
            if not (commands["CREATE_DATA"] or commands["LOAD_DATA"]):
                test_dataset, generic_test_dataset, samples_model = load_datasets(
                    system_model_params=system_model_params,
                    model_type=model_config.model_type,
                    samples_size=samples_size,
                    datasets_path=datasets_path,
                    train_test_ratio=train_test_ratio,
                )

            # Generate DataLoader objects
            model_test_dataset = torch.utils.data.DataLoader(
                test_dataset, batch_size=1, shuffle=False, drop_last=False
            )
            generic_test_dataset = torch.utils.data.DataLoader(
                generic_test_dataset, batch_size=1, shuffle=False, drop_last=False
            )
            # Load pre-trained model
            if not commands["TRAIN_MODEL"]:
                # Define an evaluation parameters instance
                simulation_parameters = (
                    TrainingParams()
                    .set_model(model=model_config)
                    .load_model(
                        loading_path=saving_path
                                     / "final_models"
                                     / simulation_filename
                    )
                )
                model = simulation_parameters.model
            # print simulation summary details
            simulation_summary(
                system_model_params=system_model_params1,  ###总结训练过程
                model_type=model_config.model_type,
                phase="evaluation",
                parameters=simulation_parameters,
            )
            # Evaluate DNN models, augmented and subspace methods
            evaluate(
                system_model_params=system_model_params1,
                model=model,
                model_type=model_config.model_type,
                model_test_dataset=model_test_dataset,
                generic_test_dataset=generic_test_dataset,
                criterion=criterion,
                subspace_criterion=subspace_criterion,
                system_model=samples_model,
                figures=figures,
                # plot_spec=True,
                plot_spec=False,
                # augmented_methods='mvdr'
                training_params=simulation_parameters,
            )
            # 在主要评估代码最后添加：
            if "comparison" in figures and figures["comparison"]["fig"] is not None:
                ax = figures["comparison"]["ax"]
                # 去重图例
                handles, labels = ax.get_legend_handles_labels()
                unique_labels = dict(zip(labels, handles))
                # 设置 y 轴范围（可选）
                ax.set_ylim(0, 1.2)  # 归一化数据，一般最大值为 1，稍微放宽 20%
                # **将图例放在图像右侧外部**
                ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
                # **调整布局，确保图例不会被裁剪**
                plt.subplots_adjust(right=0.75)  # 让图像腾出右侧空间

                # 保存图像
                from pathlib import Path

                save_dir = Path("data/spectrums")
                save_dir.mkdir(parents=True, exist_ok=True)
                figures["comparison"]["fig"].savefig(save_dir / "cnn_mvdr_comparison.png",
                                                     bbox_inches='tight',
                                                     dpi=300)
                plt.close(figures["comparison"]["fig"])

        plt.show()
        print("end")
