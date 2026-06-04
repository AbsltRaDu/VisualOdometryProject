import json
import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_history(json_path: str):
    """
    Строит графики истории обучения модели по JSON-файлу.
    """

    # Загружаем JSON-файл
    with open(json_path, "r", encoding="utf-8") as file:
        history = json.load(file)

    # Эпохи делаем с 1, а не с 0
    epochs = [epoch + 1 for epoch in history["epoch"]]

    # Настройка общего стиля графиков
    sns.set_theme(style="whitegrid", context="talk")

    # Создаем область из 4 графиков
    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(18, 12)
    )

    # -------------------------------
    # 1. Loss train / validation
    # -------------------------------
    axes[0, 0].plot(
        epochs,
        history["loss_train"],
        marker="o",
        linewidth=2.5,
        label="loss на тренировке"
    )

    axes[0, 0].plot(
        epochs,
        history["loss_test"],
        marker="o",
        linewidth=2.5,
        label="loss на валидации"
    )

    axes[0, 0].set_title("Общая ошибка модели")
    axes[0, 0].set_xlabel("Эпоха")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # -------------------------------
    # 2. Ошибка конечной позы
    # -------------------------------
    axes[0, 1].plot(
        epochs,
        history["loss_trajectory_train_gl"],
        marker="o",
        linewidth=2.5,
        label="Ошибка конечной позы на тренировке"
    )

    axes[0, 1].plot(
        epochs,
        history["loss_trajectory_test_gl"],
        marker="o",
        linewidth=2.5,
        label="Ошибка конечной позы на валидации"
    )

    axes[0, 1].set_title("Ошибка конечной позы")
    axes[0, 1].set_xlabel("Эпоха")
    axes[0, 1].set_ylabel("Loss trajectory")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # -------------------------------
    # 3. Average Translational RMSE drift
    # -------------------------------
    axes[1, 0].plot(
        epochs,
        history["Average Translational RMSE drift"],
        marker="o",
        linewidth=2.5,
        label="Average Translational RMSE drift"
    )

    axes[1, 0].set_title("Средний трансляционный RMSE drift")
    axes[1, 0].set_xlabel("Эпоха")
    axes[1, 0].set_ylabel("Translational RMSE drift")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # -------------------------------
    # 4. Average Rotational RMSE drift
    # -------------------------------
    axes[1, 1].plot(
        epochs,
        history["Average Rotational RMSE drift"],
        marker="o",
        linewidth=2.5,
        label="Average Rotational RMSE drift"
    )

    axes[1, 1].set_title("Средний rotational RMSE drift")
    axes[1, 1].set_xlabel("Эпоха")
    axes[1, 1].set_ylabel("Rotational RMSE drift")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # Делаем расположение графиков аккуратным
    plt.tight_layout()

    # Показываем графики
    plt.show()

plot_training_history('process_of_fitting/result_of_fitting/ViNet_AirSim.json')