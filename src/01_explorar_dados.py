import os
import hashlib
import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

EXTENSOES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
EXEMPLOS = 4
PARES = 6
# bits diferentes (de 256) para considerar a mesma foto -- calibrado no olho: ate 27 sempre a mesma foto, a partir de 31 fotos diferentes
LIMIAR = 27
RAIZ_MCVSLD = "data/raw/Skin Lesion Dataset"
CLASSES = ["Chickenpox", "Measles", "Monkeypox", "Normal"]
MAPA_ROTULOS = {"Healthy": "Normal"}

caminho = kagglehub.dataset_download("gupiii/infectious-skin-disease-dataset")
RAIZ = os.path.join(caminho, "Dataset_Gabungan")

entradas = []
for pasta, _, arquivos in os.walk(RAIZ):
    for arquivo in sorted(arquivos):
        if arquivo.lower().endswith(EXTENSOES):
            entradas.append((os.path.join(pasta, arquivo), os.path.basename(pasta), arquivo.split("_")[0]))

# no MCVSLD cada foto tem 15 versoes aumentadas (rotacao, espelho, ruido, cor) -- so a _ORIGINAL entra
for divisao in ["train", "val", "test"]:
    for classe in sorted(os.listdir(os.path.join(RAIZ_MCVSLD, divisao))):
        for arquivo in sorted(os.listdir(os.path.join(RAIZ_MCVSLD, divisao, classe))):
            if arquivo.endswith("_ORIGINAL.jpg"):
                entradas.append((os.path.join(RAIZ_MCVSLD, divisao, classe, arquivo), MAPA_ROTULOS.get(classe, classe), "mcvsld"))

linhas = []
hashes = []
for caminho_img, rotulo, fonte in entradas:
    arquivo = os.path.basename(caminho_img)
    if rotulo not in CLASSES:
        continue
    with open(caminho_img, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    with Image.open(caminho_img) as img:
        largura, altura = img.size
        modo = img.mode
        cinza = np.asarray(img.convert("L").resize((17, 16)), dtype=float)
        rgb = np.asarray(img.convert("RGB"), dtype=float)
    # dHash compara pixels vizinhos -- nao muda com resolucao, formato ou compressao
    hashes.append((cinza[:, 1:] > cinza[:, :-1]).flatten())
    linhas.append({
        "path": caminho_img,
        "file": arquivo,
        "label": rotulo,
        "source": fonte,
        "format": os.path.splitext(arquivo)[1].lower(),
        "mode": modo,
        "width": largura,
        "height": altura,
        "brightness": rgb.mean(),
        "red": rgb[:, :, 0].mean(),
        "green": rgb[:, :, 1].mean(),
        "blue": rgb[:, :, 2].mean(),
        "hash": md5,
    })

df = pd.DataFrame(linhas)
hashes = np.array(hashes)

vizinho = []
for i in range(len(df)):
    distancias = np.count_nonzero(hashes != hashes[i], axis=1)
    distancias[i] = hashes.shape[1]
    vizinho.append(int(distancias.min()))
df["nearest_dist"] = vizinho

# agrupamento guloso: cada imagem entra no grupo do primeiro representante proximo
representantes = []
grupos = []
for i in range(len(df)):
    if representantes:
        distancias = np.count_nonzero(hashes[representantes] != hashes[i], axis=1)
        j = int(distancias.argmin())
        if distancias[j] <= LIMIAR:
            grupos.append(j)
            continue
    representantes.append(i)
    grupos.append(len(representantes) - 1)
df["group"] = grupos

os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/imagens.csv", index=False)

print(f"{len(df)} imagens carregadas de {RAIZ} e {RAIZ_MCVSLD}")
print("\nImagens por classe e fonte:")
print(pd.crosstab(df["label"], df["source"], margins=True))
print("\nFormato e resolucao por fonte:")
print(df.groupby(["source", "format", "mode", "width", "height"]).size())

print(f"\nDuplicatas exatas (md5): {df.duplicated('hash').sum()}")
print(f"Grupos de quase-duplicatas (limiar {LIMIAR} bits): {df['group'].nunique()} imagens unicas de {len(df)}")
tamanhos = df.groupby("group").size()
print("Tamanho dos grupos:")
print(tamanhos.value_counts().sort_index())
fontes = df.groupby("group")["source"].nunique()
print(f"Grupos com a mesma foto em fontes diferentes: {(fontes > 1).sum()}")
conflitos = df.groupby("group")["label"].nunique()
print(f"Grupos com rotulos diferentes: {(conflitos > 1).sum()}")
print("\nImagens unicas por classe:")
print(df.drop_duplicates("group")["label"].value_counts())

# diferencas de cor por fonte podem virar atalho para o modelo
print("\nCor media (0-255) por classe e fonte:")
print(df.groupby(["label", "source"])[["brightness", "red", "green", "blue"]].mean().round(1))

os.makedirs("results", exist_ok=True)

figura, eixo = plt.subplots(figsize=(8, 4))
pd.crosstab(df["label"], df["source"]).plot.bar(stacked=True, ax=eixo)
eixo.set_ylabel("Imagens")
eixo.set_title("Distribuicao de classes por fonte")
figura.tight_layout()
figura.savefig("results/distribuicao_classes.png", dpi=150)
plt.close(figura)

figura, eixo = plt.subplots(figsize=(8, 4))
eixo.hist(df["nearest_dist"], bins=range(0, df["nearest_dist"].max() + 2))
eixo.axvline(LIMIAR + 0.5, color="red", linestyle="--", label=f"limiar = {LIMIAR}")
eixo.set_xlabel("Distancia ao vizinho mais proximo (bits de 256)")
eixo.set_ylabel("Imagens")
eixo.set_title("Distancia entre hashes perceptuais")
eixo.legend()
figura.tight_layout()
figura.savefig("results/distancia_hashes.png", dpi=150)
plt.close(figura)

contagem = df["label"].value_counts()
classes = sorted(df["label"].unique())
figura, eixos = plt.subplots(len(classes), EXEMPLOS, figsize=(2.5 * EXEMPLOS, 2.5 * len(classes)), squeeze=False)
for i, classe in enumerate(classes):
    amostra = df[df["label"] == classe].sample(min(EXEMPLOS, contagem[classe]), random_state=42)
    for j in range(EXEMPLOS):
        eixos[i, j].axis("off")
        if j < len(amostra):
            eixos[i, j].imshow(Image.open(amostra["path"].iloc[j]).convert("RGB"))
    eixos[i, 0].set_title(classe, loc="left")
figura.tight_layout()
figura.savefig("results/exemplos_classes.png", dpi=150)
plt.close(figura)

# pares mais distantes ainda agrupados -- servem para conferir o limiar no olho
pares = []
for i in range(len(df)):
    rep = representantes[df["group"].iloc[i]]
    if rep != i:
        pares.append((int(np.count_nonzero(hashes[rep] != hashes[i])), rep, i))
pares = sorted(pares, reverse=True)[:PARES]
if pares:
    figura, eixos = plt.subplots(len(pares), 2, figsize=(5, 2.5 * len(pares)), squeeze=False)
    for k, (distancia, a, b) in enumerate(pares):
        for col, idx in enumerate([a, b]):
            eixos[k, col].imshow(Image.open(df["path"].iloc[idx]).convert("RGB"))
            eixos[k, col].axis("off")
            eixos[k, col].set_title(f"{df['file'].iloc[idx]}\n{df['label'].iloc[idx]} (dist {distancia})", fontsize=7)
    figura.tight_layout()
    figura.savefig("results/quase_duplicatas.png", dpi=150)
    plt.close(figura)

print("Salvo em: data/processed/imagens.csv e results/")
