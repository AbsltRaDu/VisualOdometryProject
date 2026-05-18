
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime 
# Отключаем интерактивные вопросы при apt install
ENV DEBIAN_FRONTEND=noninteractive

# Устанавливаем системные зависимости для OpenCV, git и сборки пакетов
RUN apt update && apt install -y \
    git \
    tmux \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Создаём рабочую директорию внутри контейнера
WORKDIR /workspace

# Копируем только requirements.txt отдельно,
COPY requirements.txt .

# Устанавливаем Python-зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код проекта внутрь контейнера
COPY . .

# Добавляем корень проекта в PYTHONPATH,
ENV PYTHONPATH=/workspace

CMD ["bash"]