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
import matplotlib.pyplot as plt
import warnings
from src.system_model import SystemModelParams
from src.signal_creation import *
from src.data_handler import *
from src.criterions import set_criterions
from src.training import *
from src.evaluation import evaluate, evaluate_model_based
from src.plotting import initialize_figures
from pathlib import Path
from src.models import ModelGenerator
import copy

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
        "SAVE_MODEL": False,  # Saving tuned model
        "EVALUATE_MODE": True,  # Evaluating desired algorithms
        "LOAD_GAN_MODEL": False,  #加载gan模型,进行数据增强
        # "LOAD_GAN_MODEL": False,  # 不加载gan模型,进行数据增强
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

    # Define system model parameters
    base_params = (
        SystemModelParams()
        .set_parameter("N", 32)
        .set_parameter("M", 2)
        .set_parameter("T", 200)
        .set_parameter("grid_size", 121)  # 设置网格点数量
        .set_parameter("signal_type", "NarrowBand")
        .set_parameter("signal_nature", "non-coherent")
        .set_parameter("eta", 0)
        .set_parameter("bias", 0)
        .set_parameter("sv_noise_var", 0)
        .set_parameter("gap", 10)
    )
    # test_mode=[3161222,3161612,3161509,3161318]
    # grid_use=[31,61,121,241]
    test_mode = [511425]#目前模型可以直接预测，不需要配合原有的模型，集合了数据增强和原有的注意力模型 5final:4142251 -10：4151451
    test_gan_mode=[420951]# 420904 只有第一层是归一化  420951 编码层是归一化的
    grid_use = [121]
    snr_values_use = [[-10, -9, -8, -7], [-10, -9, -8, -7, -6], [-10, -9, -8, -7, -3], [-10, -9, -8, -7, -6, -3]]
    for i, test_mode_snr in enumerate(test_mode):
        for j, test_gan_mode_snr in enumerate(test_gan_mode):
            base_params = copy.deepcopy(base_params).set_parameter("grid_size", grid_use[i])
            system_model_params = copy.deepcopy(base_params).set_parameter("snr",test_mode_snr)  #评估时加载的模型d:3161222 3161612 3161509 3161318
            system_gan_model_params = copy.deepcopy(base_params).set_parameter("snr", test_gan_mode_snr)
            # 定义需要遍历的snr值列表 t:3161845  3162036 3162104 3162138
            # test_snr = range(-13,6,1)
            # test_snr = [-10,-5,0,10]
            test_snr = range(5, 6, 1)
            # test_snr=[10]
            # Define samples size
            samples_size = 30000  # Overall dateset size
            train_test_ratio = 0.0002  # training and testing datasets ratio
            # Generate model configuration
            model_config = (#CNN模型
                ModelGenerator()
                .set_model_type("GAN_Model")  # SubspaceNet  DeepCNN DA-MUSIC   DeepRootMUSIC  My_transform_Model
                .set_diff_method("root_music")  # root_music esprit
                .set_tau(8)
                .set_model(system_model_params)
            )
            model_gan_config = (#gan模型
                ModelGenerator()
                .set_model_type("GAN_Model")  # SubspaceNet  DeepCNN DA-MUSIC   DeepRootMUSIC  My_transform_Model
                .set_diff_method("root_music")  # root_music esprit
                .set_tau(8)
                .set_model(system_gan_model_params)
            )
            # Saving simulation scores to external file
            if commands["SAVE_TO_FILE"]:
                file_path = (
                        simulations_path / "results" / "scores" / Path(
                    dt_string_for_save + f"test_{model_config.model_type}_{system_model_params.grid_size}_{samples_size}_model={system_model_params.snr}_gap={system_model_params.gap}.txt")
                )
                sys.stdout = open(file_path, "w")

            for snr in test_snr:

                system_model_params1 = copy.deepcopy(base_params).set_parameter("snr", snr)  #测试集生成需要的参数

                # Sets simulation filename
                simulation_filename = get_simulation_filename(
                    system_model_params=system_model_params, model_config=model_config
                )

                # Print new simulation intro
                print("------------------------------------")
                print("---------- New Simulation ----------")
                print("------------------------------------")
                print("date and time =", dt_string)
                # print(f"SNR 值: {system_model_params.snr}")
                print(f'modelgrid_size: {system_model_params.grid_size}')
                # Initialize seed
                set_unified_seed()
                # Datasets creation
                if commands["CREATE_DATA"]:
                    # Define which datasets to generate
                    create_training_data = False  # Flag for creating training data
                    create_testing_data = True  # Flag for creating test data
                    print("Creating Data...")
                    if create_training_data:
                        # Generate training dataset
                        train_dataset, _, _ = create_dataset(
                            system_model_params=system_model_params1,
                            samples_size=samples_size,
                            model_type=model_config.model_type,
                            tau=model_config.tau,
                            save_datasets=True,
                            datasets_path=datasets_path,
                            true_doa=None,
                            phase="train",
                        )
                    if create_testing_data:
                        # # 生成第一个角度：14:0.01:11.01
                        angle1 = np.arange(60, -59, -1)  # 从 14 开始，间隔 -0.01，直到 11.00
                        angle1 = np.round(angle1, 2)  # 保留两位小数
                        # 生成第二个角度：固定间隔为 -5 度
                        angle2 = angle1 - 1  # 每个值减去 5 度
                        angle2 = np.round(angle2, 2)  # 保留两位小数
                        # 将两个角度列表合并为元组列表
                        paired_angles = list(zip(angle1, angle2))
                        # Generate test dataset
                        test_dataset, generic_test_dataset, samples_model = create_dataset(
                            system_model_params=system_model_params1,
                            samples_size=60,
                            # samples_size=int(train_test_ratio * samples_size),
                            model_type=model_config.model_type,
                            tau=model_config.tau,
                            save_datasets=True,
                            datasets_path=datasets_path,
                            true_doa=None,
                            # true_doa=paired_angles,  #生成测试集时需要指定角度
                            phase="test",
                        )
                        # if not commands["LOAD_GAN_MODEL"]:
                        #     # 加载GAN模型
                        #     simulation_parameters = (
                        #         TrainingParams()
                        #         .set_model(model=model_config1)
                        #         .load_model(
                        #             loading_path=saving_path / "final_models" / simulation_filename
                        #         )
                        #     )
                        #     model = simulation_parameters.model
                        #
                        #     # 使用GAN生成增强数据
                        #     print("Generating augmented data using GAN...")
                        #     augmented_data = []
                        #     for x_model, y in test_dataset:
                        #         with torch.no_grad():
                        #             generated_x_model = model.generator(x_model)
                        #             augmented_data.append((generated_x_model, y))
                        #     test_dataset = augmented_data


                            # 保存增强数据集
                            # augmented_filename = f"GAN_augmented_{model_dataset_filename}"
                            # torch.save(
                            #     obj=augmented_data,
                            #     f=datasets_path / "test" / augmented_filename
                            # )
                            # print(f"Saved augmented dataset to: {datasets_path / 'test' / augmented_filename}")
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
                if commands["TRAIN_MODEL"]:
                    # Assign the training parameters object
                    simulation_parameters = (
                        TrainingParams()
                        .set_batch_size(512)
                        .set_epochs(80)
                        .set_model(model=model_config)
                        .set_optimizer(optimizer="Adam", learning_rate=0.00001,
                                       weight_decay=1e-9)  #learning_rate=0.00001, weight_decay=1e-9
                        .set_training_dataset(train_dataset)
                        .set_schedular(step_size=80, gamma=0.2)
                        .set_criterion()  #自动设置成nn.BCELoss()
                    )
                    if commands["LOAD_MODEL"]:
                        simulation_parameters.load_model(
                            loading_path=saving_path / "final_models" / simulation_filename
                        )
                    # Print training simulation details
                    simulation_summary(
                        system_model_params=system_model_params1,
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
                        torch.save(
                            model.state_dict(),
                            saving_path / "final_models" / Path(simulation_filename),
                        )
                    # Plots saving
                    if commands["SAVE_TO_FILE"]:
                        plt.savefig(
                            simulations_path
                            / "results"
                            / "plots"
                            / Path(dt_string_for_save + r".png")
                        )
                    else:
                        plt.show()

                # Evaluation stage
                if commands["EVALUATE_MODE"]:
                    # Initialize figures dict for plotting
                    figures = initialize_figures()
                    # figures='comparison'#进行不同方法的对比
                    # Define loss measure for evaluation
                    criterion, subspace_criterion = set_criterions("bce")

                    # Load datasets for evaluation
                    if not (commands["CREATE_DATA"] or commands["LOAD_DATA"]):
                        test_dataset, generic_test_dataset, samples_model = load_datasets(
                            system_model_params=system_model_params1,
                            model_type=model_config.model_type,
                            samples_size=samples_size,
                            datasets_path=datasets_path,
                            train_test_ratio=train_test_ratio,
                        )
                    if commands["LOAD_GAN_MODEL"]:
                        simulation_gan_filename = get_simulation_filename(
                            system_model_params=system_gan_model_params, model_config=model_gan_config
                        )
                        # 加载GAN模型
                        simulation_gan_parameters = (
                            TrainingParams()
                            .set_model(model=model_gan_config)
                            .load_model(
                                loading_path=saving_path / "final_models" / simulation_gan_filename
                            )
                        )
                        model_gan = simulation_gan_parameters.model.discriminator

                        # 使用GAN生成增强数据
                        print("Generating augmented data using GAN...")
                        model_gan.to(device)
                        augmented_data = []
                        for x_model, y in test_dataset:
                            with torch.no_grad():
                                generated_x_model = model_gan.generator(x_model.permute(2, 0, 1).unsqueeze(0).to(device))
                                augmented_data.append((generated_x_model.squeeze(0).permute(1, 2, 0).to(device), y))
                        test_dataset = augmented_data

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
                        plot_spec=False,
                        # plot_spec=False,
                        training_params=simulation_parameters
                        # augmented_methods='mvdr'
                    )
                elif not commands["EVALUATE_MODE"]:
                    # Initialize figures dict for plotting
                    figures = initialize_figures()
                    # Load datasets for evaluation
                    if not (commands["CREATE_DATA"] or commands["LOAD_DATA"]):
                        test_dataset, generic_test_dataset, samples_model = load_datasets(
                            system_model_params=system_model_params1,
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
                        simulation_parameters = TrainingParams()

                        # model = simulation_parameters.model
                    # simulation_summary(
                    #     system_model_params=system_model_params1,  ###总结训练过程
                    #     model_type=model_config.model_type,
                    #     phase="evaluation",
                    #     parameters=simulation_parameters,
                    # )
                    # Define loss measure for evaluation
                    criterion, subspace_criterion = set_criterions("rmse")
                    loss, accuracy = evaluate_model_based(
                        system_model_params1,
                        generic_test_dataset,
                        samples_model,
                        criterion=subspace_criterion,
                        plot_spec=True,
                        algorithm='mvdr',
                        figures=figures,
                    )
                    print("{} test loss = {}".format('mvdr'.lower(), loss * R2D))
                    print(f" mvdr accuracy = {accuracy}")

                    # 在主要评估代码最后添加：
                    # if "comparison" in figures and figures["comparison"]["fig"] is not None:
                    #     ax = figures["comparison"]["ax"]
                    #     # 去重图例
                    #     handles, labels = ax.get_legend_handles_labels()
                    #     unique_labels = dict(zip(labels, handles))
                    #     # 设置 y 轴范围（可选）
                    #     ax.set_ylim(0, 1.2)  # 归一化数据，一般最大值为 1，稍微放宽 20%
                    #     # **将图例放在图像右侧外部**
                    #     ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
                    #     # **调整布局，确保图例不会被裁剪**
                    #     plt.subplots_adjust(right=0.75)  # 让图像腾出右侧空间
                    #
                    #     # 保存图像
                    #     from pathlib import Path
                    #
                    #     save_dir = Path("data/spectrums")
                    #     save_dir.mkdir(parents=True, exist_ok=True)
                    #     figures["comparison"]["fig"].savefig(save_dir / "cnn_mvdr_comparison.png",
                    #                                          bbox_inches='tight',
                    #                                          dpi=300)
                    #     plt.close(figures["comparison"]["fig"])

                plt.show()
                print("end")
