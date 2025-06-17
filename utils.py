import io
import os
import sys
import tempfile

import imageio
import matplotlib.pyplot as plt
import numpy as np

from IPython.display import Image, display
from loguru import logger
from PIL import Image as PILImage


class LoggerHandler:
    def __init__(self, logger):
        self.logger = logger

    def del_logger(self):
        for logger_id in range(100):
            try:
                logger.remove(logger_id)
                return logger_id
            except:
                pass
        assert (
            logger_id > 100
        )  # You have probably created over than 100 loggers. Restart your environment

    def change_log_level(self, level):
        self.del_logger()
        self.logger.add(sink=sys.stdout, level=level)


class GifMaker:
    def __init__(self, duration=0.5, loop=0):
        """
        :param duration: Длительность каждого кадра (в секундах)
        :param loop: Кол-во повторов (0 — бесконечно)
        """
        self.images = []
        self.duration = duration
        self.loop = loop

    def add_frames(self, loss_history, loss_history_test=None, w=None):
        fig, axs = plt.subplots(3, 1, figsize=(10, 6))
        axs[0].plot(loss_history, label="Train")
        axs[0].legend()
        axs[0].grid()

        if loss_history_test:
            axs[1].plot(loss_history_test, label="Test")
            axs[1].legend()
            axs[1].grid()

        if w is not None:
            axs[2].bar([f"X{i}" for i in range(w.shape[0])], w)

        fig.tight_layout()

        # Сохраняем в буфер
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        img = PILImage.open(buf)
        self.images.append(img)
        plt.close(fig)

    def create_gif(self, gif_name="animation.gif"):
        """
        Создаёт и отображает GIF из изображений.

        :param images: список путей к изображениям или PIL.Image
        :param gif_name: имя gif файла
        """
        frames = []

        for img in self.images:
            if isinstance(img, str) and os.path.isfile(img):
                frames.append(imageio.imread(img))
            elif isinstance(img, PILImage.Image):
                frames.append(np.asarray(img.convert("RGB")))
            else:
                raise ValueError("Поддерживаются только пути к изображениям или PIL.Image объекты")

        with tempfile.TemporaryDirectory() as tmpdir:
            gif_path = os.path.join(tmpdir, gif_name)
            imageio.mimsave(gif_path, frames, duration=self.duration, loop=self.loop)
            with open(gif_path, "rb") as f:
                display(Image(data=f.read(), format="png"))
