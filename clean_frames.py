import cv2
import numpy as np

class AnalogNoiseFilter:
    """
    Детектор сильных аналоговых помех (чёрные полосы, снег, полная потеря сигнала и т.д.)
    Оптимизирован под реальные VHS/аналоговые артефакты.
    """
    def __init__(
        self,
        black_threshold=45,          # Порог "очень тёмного" пикселя
        min_black_ratio=0.18,         # >18% тёмных → плохой кадр
        min_black_area_ratio=0.10,    # Одна связная тёмная область >10% кадра → плохой
        blur_size=9                   # Размер гауссовского размытия перед анализом
    ):
        self.BLACK_THRESHOLD = black_threshold
        self.MIN_BLACK_RATIO = min_black_ratio
        self.MIN_BLACK_AREA_RATIO = min_black_area_ratio
        self.BLUR_SIZE = blur_size if blur_size % 2 == 1 else blur_size + 1

        self.last_good_frame = None   # Здесь хранится последний валидный кадр

    def is_frame_bad(self, frame: np.ndarray) -> bool:
        """
        Возвращает True, если кадр считается сильно зашумлённым / потерянным.
        """
        if frame is None:
            return True

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.BLUR_SIZE, self.BLUR_SIZE), 0)

        # 1. Общая доля очень тёмных пикселей
        black_pixels_ratio = np.mean(blurred < self.BLACK_THRESHOLD)
        if black_pixels_ratio > self.MIN_BLACK_RATIO:
            return True

        # 2. Одна огромная чёрная область (полоса, блок и т.п.)
        mask = (blurred < self.BLACK_THRESHOLD).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(mask)


        if num_labels > 1:
            # Самая большая связная область (исключаем нулевой фон)
            largest_area = np.max(np.bincount(labels.ravel())[1:])
            total_pixels = frame.shape[0] * frame.shape[1]
            if largest_area > total_pixels * self.MIN_BLACK_AREA_RATIO:
                return True

        return False

    def filter_frame(self, frame: np.ndarray):
        """
        Основная функция для интеграции.
        Возвращает "чистый" кадр:
         - если текущий хороший → возвращает его
         - если плохой → возвращает последний хороший (или чёрный кадр при первом запуске)
        """
        if self.is_frame_bad(frame):
            # Держим последний хороший
            if self.last_good_frame is None:
                # Первый раз — просто чёрный кадр
                clean = np.full_like(frame, 255)
                cv2.putText(clean, "NO SIGNAL", (50, frame.shape[0]//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            else:
                clean = self.last_good_frame.copy()
                cv2.putText(clean, "NO SIGNAL - HOLDING LAST", (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
        else:
            # Хороший кадр — сохраняем и отдаём
            self.last_good_frame = frame.copy()
            clean = frame.copy()
            cv2.putText(clean, "VALID", (80, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 3)

        return clean

    def reset(self):
        """Сбросить сохранённый последний хороший кадр (например, при смене видео)"""
        self.last_good_frame = None




# noise_filter = AnalogNoiseFilter(
#     black_threshold=45,
#     min_black_ratio=0.18,
#     min_black_area_ratio=0.10,
#     blur_size=9
# )
# clean_frame = noise_filter.filter_frame(frame)   # ← вот и всё!