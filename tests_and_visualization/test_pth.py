import torch
from src.models.modelsNN.DeepVOFRA import DeepVO

# Путь к твоему .pth файлу
path = "process_of_fitting/fitting_models/checkpoint_e190.pth"

# Загружаем .pth как Python-объект
# weights_only=False нужен, если внутри лежит полный checkpoint-словарь
checkpoint = torch.load(
    path,
    map_location="cpu",
    weights_only=False,
)



model = DeepVO()
model.load_state_dict(checkpoint['model_state_dict'])

print(model)