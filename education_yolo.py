from ultralytics import YOLO
import torch
import os

# Параметры обучения
MODEL_PATH = 'yolo11n.pt'           # Путь к предобученной модели YOLO11n
DATA_CONFIG = 'dataset.yaml'          # Путь к YAML-конфигу с описанием датасета
EPOCHS = 100                        # Количество эпох обучения
BATCH_SIZE = 16                   # Размер батча (подбирайте под GPU)
IMG_SIZE = 640                    # Размер изображения
PROJECT_NAME = 'yolo11n_train'  # Название проекта для логов
EXPERIMENT_NAME = 'exp1'          # Название эксперимента

# Проверка доступности GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Используемое устройство: {device}")

# Загрузка модели
model = YOLO(MODEL_PATH)

# Настройка и запуск обучения
results = model.train(
    data=DATA_CONFIG,
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    imgsz=IMG_SIZE,
    device=device,
    project=PROJECT_NAME,
    name=EXPERIMENT_NAME,
    save=True,                    # Сохранять чекпоинты
    save_period=10,             # Сохранять каждые 10 эпох
    patience=20,               # Ранняя остановка при отсутствии улучшения
    optimizer='AdamW',          # Оптимизатор
    lr0=0.001,               # Начальная скорость обучения
    lrf=0.01,                # Конечная скорость обучения (в конце scheduler)
    momentum=0.937,          # Momentum для SGD/Adam
    weight_decay=0.0005,     # Весовая регуляризация
    warmup_epochs=3.0,       # Эпохи прогрева
    warmup_momentum=0.8,     # Momentum в прогреве
    warmup_bias_lr=0.1,     # LR для bias в прогреве
    box=0.05,                # Вес потери для bounding box
    cls=0.5,                 # Вес потери для классификации
    dfl=1.5,                 # Вес распределения для DFL
    label_smoothing=0.0,     # Smoothing меток
    close_mosaic=10,         # Отключить Mosaic после N эпох
    flipud=0.0,            # Вертикальный флип (вероятность)
    fliplr=0.5,            # Горизонтальный флип (вероятность)
    mixup=0.1,             # MixUp augmentation
    copy_paste=0.1,        # Copy-paste augmentation
)

# Сохранение финальной модели
final_model_path = os.path.join(PROJECT_NAME, EXPERIMENT_NAME, 'weights', 'best.pt')
print(f"Финальная модель сохранена: {final_model_path}")

# Дополнительно: экспорт в другой формат (например, ONNX)
model.export(format='onnx', imgsz=IMG_SIZE)
print("Модель экспортирована в ONNX")
