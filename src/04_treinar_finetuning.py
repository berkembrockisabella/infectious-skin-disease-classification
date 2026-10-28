import os
import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import f1_score

SEED = 42
TAMANHO = 224
LOTE = 32
EPOCAS = 20
PACIENCIA = 5
# faixa usada nos artigos de referencia -- 1e-3 (CNN/ensemble), 1e-4 (hibrido), 1e-5 (VGG16)
TAXAS = [1e-3, 1e-4, 1e-5]
MEDIA = [0.485, 0.456, 0.406]
DESVIO = [0.229, 0.224, 0.225]

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

# pesos inversos a frequencia para nao favorecer a classe majoritaria
contagem = torch.bincount(torch.tensor(treino.targets), minlength=num_classes).float()
pesos = (len(treino) / (num_classes * contagem)).to(dispositivo)
criterio = nn.CrossEntropyLoss(weight=pesos)

os.makedirs("models", exist_ok=True)
busca = []
melhor_f1_geral = -1.0
for taxa in TAXAS:
    torch.manual_seed(SEED)
    # pesos do ImageNet; todas as camadas sao ajustadas, so a ultima e trocada
    modelo = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    modelo.fc = nn.Linear(modelo.fc.in_features, num_classes)
    modelo = modelo.to(dispositivo)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=taxa)

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
            "lr": taxa,
            "epoch": epoca,
            "train_loss": perda_total / len(treino),
            "val_loss": perda_val / len(validacao),
            "val_f1_macro": f1_val,
        })
        print(f"lr={taxa} epoca {epoca}: train_loss={linhas[-1]['train_loss']:.4f} val_loss={linhas[-1]['val_loss']:.4f} val_f1={f1_val:.4f}")

        if f1_val > melhor_f1:
            melhor_f1 = f1_val
            sem_melhora = 0
            if melhor_f1 > melhor_f1_geral:
                torch.save(modelo, "models/finetuning.pt")
        else:
            sem_melhora += 1
            if sem_melhora >= PACIENCIA:
                print("Parada antecipada.")
                break

    busca.append({"lr": taxa, "best_val_f1_macro": melhor_f1, "epochs_run": len(linhas)})
    if melhor_f1 > melhor_f1_geral:
        melhor_f1_geral = melhor_f1
        melhor_historico = linhas

resultado = pd.DataFrame(busca)
resultado.to_csv("data/processed/busca_lr_finetuning.csv", index=False)
historico = pd.DataFrame(melhor_historico)
historico.to_csv("data/processed/historico_finetuning.csv", index=False)

print(resultado)
print(f"Melhor lr: {historico['lr'].iloc[0]} (F1-macro na validacao: {melhor_f1_geral:.4f})")
print("Salvo em: models/finetuning.pt")
