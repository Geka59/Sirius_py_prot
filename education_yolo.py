from ultralytics import YOLO

# Загружаем предобученную модель YOLO11n
# Если хочешь с нуля — используй yolo11n.pt с параметром pretrained=False
model = YOLO("yolo11n.pt")    # или "yolo11s.pt", "yolo11m.pt" и т.д.

# Запуск обучения
results = model.train(
    data="dataset/data.yaml",   # путь к твоему yaml
    epochs=100,                 # сколько эпох
    imgsz=640,                  # размер изображения
    batch=16,                   # подбери под свою видеокарту (8-32 обычно)
    device=0,                   # 0 = первая GPU, можно [0,1] или "cpu"
    project="runs/yolo11n_custom",  # куда сохранять
    name="exp",                 # имя эксперимента
    patience=20,                # early stopping
    pretrained=True,            # используем предобученные веса
    optimizer="AdamW",          # или SGD
    lr0=0.001,                  # начальный learning rate
    weight_decay=0.0005,
    warmup_epochs=3,
    box=7.5,                    # коэффициенты лоссов (можно не трогать)
    cls=0.5,
    dfl=1.5,
    # Аугментации (по желанию)
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.0,
    cache=True,                 # кэшировать изображения в RAM (ускоряет обучение)
    workers=8,                  # количество потоков для DataLoader
    amp=True,                   # Automatic Mixed Precision (очень ускоряет и экономит память)
)

# После обучения лучшая модель будет в runs/yolo11n_custom/exp/weights/best.pt
