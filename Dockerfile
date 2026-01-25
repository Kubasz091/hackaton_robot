# 1. Obraz bazowy z CUDA 12.1 (pasuje do RTX / B200)
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# 2. Instalacja zależności systemowych (zapobiega błędom cv2/ffmpeg)
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 ffmpeg

# 3. Kopiujemy cały folder hackathonu do kontenera
COPY . /workspace/lerobot_hackathon

# 4. Instalacja LeRobot (bez sprawdzania zależności, żeby nie zepsuć torchcodec)
RUN pip install -e /workspace/lerobot_hackathon --no-deps

# 5. KOPIOWANIE MODELU - TUTAJ PODMIEŃ DATĘ I GODZINĘ NA TWOJĄ!
# Ścieżkę weź z ostatniej linijki logów treningu
COPY ./outputs/train/2026-01-24/13_12_49_act/checkpoints/last/pretrained_model /workspace/model

EXPOSE 8080

# 6. Uruchomienie serwera na porcie 8080
CMD ["python", "/workspace/lerobot_hackathon/src/lerobot/async_inference/policy_server.py", "--model-path", "/workspace/model", "--port", "8080"]