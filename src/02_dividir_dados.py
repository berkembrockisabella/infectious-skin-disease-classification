import os
import shutil
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

SEED = 42
TAMANHO = 256
DESTINO = "data/split"

df = pd.read_csv("data/processed/imagens.csv")
total = len(df)

# a mesma foto com rotulos diferentes e ambigua -- descartada
conflitos = df.groupby("group")["label"].nunique()
df = df[~df["group"].isin(conflitos[conflitos > 1].index)]

# uma imagem por grupo (a de maior resolucao) -- evita a mesma foto em treino e teste
df = df.sort_values(["width", "path"], ascending=[False, True]).drop_duplicates("group")

treino, resto = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=SEED)
validacao, teste = train_test_split(resto, test_size=0.5, stratify=resto["label"], random_state=SEED)

divisao = pd.concat([
    treino.assign(split="train"),
    validacao.assign(split="val"),
    teste.assign(split="test"),
])

if os.path.exists(DESTINO):
    shutil.rmtree(DESTINO)

for _, row in divisao.iterrows():
    pasta = os.path.join(DESTINO, row["split"], row["label"])
    os.makedirs(pasta, exist_ok=True)
    # mesmo tamanho e formato sem perda para todas -- a fonte nao fica identificavel pelo arquivo
    with Image.open(row["path"]) as img:
        img.convert("RGB").resize((TAMANHO, TAMANHO), Image.BICUBIC).save(os.path.join(pasta, row["hash"] + ".png"))

divisao.to_csv("data/processed/divisao.csv", index=False)

print(f"{total - len(divisao)} imagens removidas (copias e conflitos); {len(divisao)} imagens unicas divididas.")
print(pd.crosstab(divisao["label"], divisao["split"], margins=True))
print("\nFonte por conjunto:")
print(pd.crosstab(divisao["source"], divisao["split"]))
print(f"Salvo em: {DESTINO}")
