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
from src.models import ModelGenerator
from src.model_transform import My_transform_Model
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
        "TRAIN_MODEL":True,  # Applying training operation
        "SAVE_MODEL": True ,  # Saving tuned model
        "EVALUATE_MODE": False,  # Evaluating desired algorithms
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

    # Saving simulation scores to external file
    if commands["SAVE_TO_FILE"]:
        file_path = (
            simulations_path / "results" / "train_scores" / Path(dt_string_for_save + ".txt")
        )
        sys.stdout = open(file_path, "w")
    # Define system model parameters

    base_params = (   #by j1 start    训练模型用
        SystemModelParams()
        .set_parameter("N", 16)
        .set_parameter("M", 2)
        .set_parameter("T", 1000)
        .set_parameter("signal_type", "NarrowBand")
        .set_parameter("signal_nature", "non-coherent")
        .set_parameter("eta", 0)
        .set_parameter("bias", 0)
        .set_parameter("sv_noise_var", 0)
    )
    system_model_params = copy.deepcopy(base_params).set_parameter("snr", 3121501)##多个数据集的训练过程日志和模型记录的参数 -20 只做记录用
    # 定义需要遍历的snr值列表
    snr_values = [0]


    #测试机加噪声
    system_model_params1 = copy.deepcopy(base_params).set_parameter("snr", 0) # 只做测试用
    # Generate model configuration
    model_config = (
        ModelGenerator()
        .set_model_type("DeepCNN")#SubspaceNet  DeepCNN DA-MUSIC   DeepRootMUSIC My_transform_Model
        .set_diff_method("root_music")# root_music esprit
        .set_tau(8)
        .set_model(system_model_params)
    )
    # Define samples size
    samples_size = 50000  # Overall dateset size
    train_test_ratio = 0.002  # training and testing datasets ratio
    # Sets simulation filename
    simulation_filename = get_simulation_filename(
        system_model_params=system_model_params, model_config=model_config
    )
    # Print new simulation intro
    print("------------------------------------")
    print("---------- New Simulation ----------")
    print("------------------------------------")
    print("date and time =", dt_string)
    print(f"train_snr_values = {snr_values}")
    print(f"modelname_SNR 值: {system_model_params.snr}")
    # Initialize seed
    set_unified_seed()
    # Datasets creation
    if commands["CREATE_DATA"]:
        # Define which datasets to generate
        create_training_data = True  # Flag for creating training data
        create_testing_data = True  # Flag for creating test data
        print("Creating Data...")
        if create_training_data:
            # Generate training dataset
            # 生成参数组列表
            param_groups = []
            for snr in snr_values:
                new_params = copy.deepcopy(base_params).set_parameter("snr", snr)
                param_groups.append({
                    "system_model_params": new_params,
                })
            # 生成合并数据集
            combined_dataset = []

            for group in param_groups:
                # 生成单个数据集
                train_data, _, _ = create_dataset(
                    system_model_params=group["system_model_params"],
                    samples_size=samples_size,
                    model_type=model_config.model_type,
                    tau=model_config.tau,
                    save_datasets=False,
                    datasets_path=datasets_path,
                    true_doa=None,
                    phase="train"
                )

                # 合并数据集
                combined_dataset.extend(train_data)

                # 内存清理
                del train_data
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
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
            is_training=True,# 训练时记得在设置
        )

    # Training stage
    if commands["TRAIN_MODEL"]:
        # Assign the training parameters object
        simulation_parameters = (
            TrainingParams()
            .set_batch_size(1024)
            .set_epochs(50)
            .set_model(model=model_config)
            .set_optimizer(optimizer="Adam", learning_rate=0.0001, weight_decay=1e-7)#learning_rate=0.00001, weight_decay=1e-9
            .set_training_dataset(combined_dataset)  #by j
            .set_schedular(step_size=15, gamma=0.5)
            .set_criterion()#自动设置成nn.BCELoss()  非常关键，训练的时候要设置
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
        model, loss_train_list, loss_valid_list = train(
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
        criterion, subspace_criterion = set_criterions("rmse")#训练的是时候是交叉熵，只有测试的时候用rmse
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
            system_model_params=system_model_params1,###总结训练过程
            model_type=model_config.model_type,
            phase="evaluation",
            parameters=simulation_parameters,
        )
        # Evaluate DNN models, augmented and subspace methods
        evaluate(
            model=model,
            model_type=model_config.model_type,
            model_test_dataset=model_test_dataset,
            generic_test_dataset=generic_test_dataset,
            criterion=criterion,
            subspace_criterion=subspace_criterion,
            system_model=samples_model,
            figures=figures,
            plot_spec=True,
            # augmented_methods='mvdr'
            training_params=simulation_parameters
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
