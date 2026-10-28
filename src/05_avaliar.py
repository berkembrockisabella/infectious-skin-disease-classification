import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay

TAMANHO = 224
LOTE = 32
EXEMPLOS = 6
MEDIA = [0.485, 0.456, 0.406]
DESVIO = [0.229, 0.224, 0.225]
MODELOS = {
    "baseline": "models/baseline.pt",
    "finetuning": "models/finetuning.pt",
}

dispositivo = "cuda" if torch.cuda.is_available() else "cpu"

transf_avaliacao = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(TAMANHO),
    transforms.ToTensor(),
    transforms.Normalize(MEDIA, DESVIO),
])
teste = datasets.ImageFolder("data/split/test", transf_avaliacao)
carregador_teste = DataLoader(teste, batch_size=LOTE, num_workers=0)

os.makedirs("results", exist_ok=True)
linhas = []
for nome, caminho in MODELOS.items():
    modelo = torch.load(caminho, map_location=dispositivo, weights_only=False)
    modelo.eval()
    previstos, reais = [], []
    with torch.no_grad():
        for imagens, rotulos in carregador_teste:
            saidas = modelo(imagens.to(dispositivo))
            previstos += saidas.argmax(1).cpu().tolist()
            reais += rotulos.tolist()

    linhas.append({
        "model": nome,
        "accuracy": accuracy_score(reais, previstos),
        "f1_macro": f1_score(reais, previstos, average="macro"),
    })
    print(f"\n=== {nome} ===")
    print(classification_report(reais, previstos, target_names=teste.classes, digits=4, zero_division=0))

    figura, eixo = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay.from_predictions(
        reais, previstos, display_labels=teste.classes, xticks_rotation=45, ax=eixo, colorbar=False
    )
    eixo.set_title(f"Matriz de confusao - {nome}")
    figura.tight_layout()
    figura.savefig(f"results/matriz_confusao_{nome}.png", dpi=150)
    plt.close(figura)

    acertos = [i for i in range(len(reais)) if reais[i] == previstos[i]][:EXEMPLOS]
    erros = [i for i in range(len(reais)) if reais[i] != previstos[i]][:EXEMPLOS]
    figura, eixos = plt.subplots(2, EXEMPLOS, figsize=(2.5 * EXEMPLOS, 6), squeeze=False)
    for linha, indices, titulo in [(0, acertos, "Acerto"), (1, erros, "Erro")]:
        for j in range(EXEMPLOS):
            eixos[linha, j].axis("off")
            if j < len(indices):
                k = indices[j]
                eixos[linha, j].imshow(Image.open(teste.samples[k][0]).convert("RGB"))
                eixos[linha, j].set_title(
                    f"{titulo}\nreal: {teste.classes[reais[k]]}\nprev: {teste.classes[previstos[k]]}", fontsize=8
                )
    figura.tight_layout()
    figura.savefig(f"results/exemplos_predicoes_{nome}.png", dpi=150)
    plt.close(figura)

    historico = pd.read_csv(f"data/processed/historico_{nome}.csv")
    figura, eixos = plt.subplots(1, 2, figsize=(10, 4))
    eixos[0].plot(historico["epoch"], historico["train_loss"], label="treino")
    eixos[0].plot(historico["epoch"], historico["val_loss"], label="validacao")
    eixos[0].set_xlabel("Epoca")
    eixos[0].set_ylabel("Perda")
    eixos[0].legend()
    eixos[1].plot(historico["epoch"], historico["val_f1_macro"])
    eixos[1].set_xlabel("Epoca")
    eixos[1].set_ylabel("F1-macro (validacao)")
    figura.suptitle(f"Curvas de aprendizado - {nome}")
    figura.tight_layout()
    figura.savefig(f"results/curvas_{nome}.png", dpi=150)
    plt.close(figura)

metricas = pd.DataFrame(linhas)
metricas.to_csv("data/processed/metricas_teste.csv", index=False)

print(metricas)
print("Salvo em: data/processed/metricas_teste.csv e results/")
