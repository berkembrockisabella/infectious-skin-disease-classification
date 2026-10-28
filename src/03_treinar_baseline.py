import os
import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import f1_score

SEED = 42
TAMANHO = 224
LOTE = 32
EPOCAS = 30
PACIENCIA = 5
TAXA = 1e-3
MEDIA = [0.485, 0.456, 0.406]
DESVIO = [0.229, 0.224, 0.225]

torch.manual_seed(SEED)
dispositivo = "cuda" if torch.cuda.is_available() else "cpu"

transf_treino = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(TAMANHO),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEDIA, DESVIO),
])
transf_avaliacao = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(TAMANHO),
    transforms.ToTensor(),
    transforms.Normalize(MEDIA, DESVIO),
])

treino = datasets.ImageFolder("data/split/train", transf_treino)
validacao = datasets.ImageFolder("data/split/val", transf_avaliacao)
# num_workers=0 porque no Windows valores maiores exigem if __name__ == "__main__"
carregador_treino = DataLoader(treino, batch_size=LOTE, shuffle=True, num_workers=0)
carregador_val = DataLoader(validacao, batch_size=LOTE, num_workers=0)
num_classes = len(treino.classes)

modelo = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3),
    nn.Linear(256, num_classes),
).to(dispositivo)

# pesos inversos a frequencia para nao favorecer a classe majoritaria
contagem = torch.bincount(torch.tensor(treino.targets), minlength=num_classes).float()
pesos = (len(treino) / (num_classes * contagem)).to(dispositivo)
criterio = nn.CrossEntropyLoss(weight=pesos)
otimizador = torch.optim.Adam(modelo.parameters(), lr=TAXA)

os.makedirs("models", exist_ok=True)
linhas = []
melhor_f1 = -1.0
sem_melhora = 0
for epoca in range(1, EPOCAS + 1):
    modelo.train()
    perda_total = 0.0
    for imagens, rotulos in carregador_treino:
        imagens, rotulos = imagens.to(dispositivo), rotulos.to(dispositivo)
        otimizador.zero_grad()
        perda = criterio(modelo(imagens), rotulos)
        perda.backward()
        otimizador.step()
        perda_total += perda.item() * len(rotulos)

    modelo.eval()
    previstos, reais = [], []
    perda_val = 0.0
    with torch.no_grad():
        for imagens, rotulos in carregador_val:
            imagens, rotulos = imagens.to(dispositivo), rotulos.to(dispositivo)
            saidas = modelo(imagens)
            perda_val += criterio(saidas, rotulos).item() * len(rotulos)
            previstos += saidas.argmax(1).tolist()
            reais += rotulos.tolist()

    f1_val = f1_score(reais, previstos, average="macro")
    linhas.append({
        "epoch": epoca,
        "train_loss": perda_total / len(treino),
        "val_loss": perda_val / len(validacao),
        "val_f1_macro": f1_val,
    })
    print(f"Epoca {epoca}: train_loss={linhas[-1]['train_loss']:.4f} val_loss={linhas[-1]['val_loss']:.4f} val_f1={f1_val:.4f}")

    if f1_val > melhor_f1:
        melhor_f1 = f1_val
        sem_melhora = 0
        torch.save(modelo, "models/baseline.pt")
    else:
        sem_melhora += 1
        if sem_melhora >= PACIENCIA:
            print("Parada antecipada.")
            break

historico = pd.DataFrame(linhas)
historico.to_csv("data/processed/historico_baseline.csv", index=False)

print(f"Melhor F1-macro na validacao: {melhor_f1:.4f}")
print("Salvo em: models/baseline.pt")
