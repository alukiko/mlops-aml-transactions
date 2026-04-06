from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from mlops_aml_transactions.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # При необходимости замените пути к данным и файлу графика
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
):
    # Заготовка: сюда — загрузка данных и построение графиков
    logger.info("Генерация графика из данных (заготовка)...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Итерация 5.")
    logger.success("Заготовка графика завершена.")


if __name__ == "__main__":
    app()
