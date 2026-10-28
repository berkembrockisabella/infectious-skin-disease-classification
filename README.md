# Classificação de Doenças Infecciosas de Pele

Trabalho da disciplina de Visão Computacional (PUCPR). O objetivo é classificar fotos de pele em **4 classes** — Chickenpox (catapora), Measles (sarampo), Monkeypox (mpox) e Normal — comparando uma CNN treinada do zero com o fine-tuning de uma ResNet50 pré-treinada.

O ponto central do trabalho acabou sendo **a qualidade dos dados**: as bases públicas usadas têm a mesma foto repetida várias vezes, o que infla os resultados quando isso não é tratado.

**Resultado principal:** no conjunto de teste sem cópias, a ResNet50 com fine-tuning alcança **85,9% de acurácia e F1-macro de 0,836**, contra 45,8% e 0,430 da CNN treinada do zero. Antes do tratamento das cópias, o mesmo modelo marcava 98,1% — um número inflado.

## Sumário

1. [Problema](#1-problema)
2. [Ponto de partida: a literatura](#2-ponto-de-partida-a-literatura)
3. [Dados](#3-dados)
4. [Etapas do experimento](#4-etapas-do-experimento)
5. [Resultados](#5-resultados)
6. [Limitações e próximos passos](#6-limitações-e-próximos-passos)
7. [Como executar](#7-como-executar)

---

## 1. Problema

Catapora, sarampo e mpox são doenças virais que causam lesões de pele parecidas. Um classificador baseado em fotos comuns (não dermatoscópicas) pode apoiar a triagem de forma rápida e barata.

Há duas dificuldades:
- **Classes parecidas:** as diferenças entre as lesões são sutis.
- **Poucos dados:** as bases públicas têm algumas centenas de fotos por classe.

## 2. Ponto de partida: a literatura

| Artigo | Base | Modelo | Resultado |
|---|---|---|---|
| Pai et al. (2023) | ~500 fotos, 5 classes (bactérias/fungos) | VGG16 com fine-tuning | 86% acurácia |
| Vajpayee et al. (2023) | MSID, 782 fotos, mesmas 4 classes | 10 CNNs; melhor EfficientNetV2L | 92,5% acurácia |
| Ahmed et al. (2026) | 1159 fotos, 8 classes | Híbrido MaxViT + MobileNet | 99,6% acurácia |

O que tiramos daqui:
- **Transfer learning é o caminho** para bases pequenas — justifica comparar CNN do zero vs. ResNet50 pré-treinada.
- **Hiperparâmetros de referência:** entrada 224×224, normalização do ImageNet, lr entre 1e-3 e 1e-5, augmentation e pesos de classe.
- **Lacuna:** nenhum dos artigos verifica fotos duplicadas, e dois deles não têm um conjunto de teste independente.

## 3. Dados

### 3.1 Fontes

| Base | Origem | O que contém |
|---|---|---|
| `Dataset_Gabungan` | Kaggle (`gupiii/infectious-skin-disease-dataset`) | Junção de `Dataset_1` (Univ. of Jammu, 659 imagens, JPG 256×256) e `Dataset_2` (MSID, 770 imagens, PNG 224×224) |
| MCVSLD | Mendeley Data (DOI 10.17632/dfztdtfsxz.1) | 11.325 imagens em 6 classes; usamos as 4 em comum (`Healthy` → `Normal`) |

### 3.2 O problema encontrado: a mesma foto várias vezes

- **No `Dataset_Gabungan`**, 95% das imagens do `Dataset_1` são a mesma foto de uma imagem do MSID, apenas redimensionada e salva em outro formato. Na prática, a base é o MSID duas vezes.
- **No MCVSLD**, cada foto aparece em 15 versões aumentadas (rotação, espelhamento, ruído, desfoque, cor), espalhadas entre treino, validação e teste. **100% das imagens de teste** têm uma versão da mesma foto no treino.

**Por que isso importa:** se a mesma foto cai no treino e no teste, o modelo é avaliado em algo que já viu. Na primeira rodada, sem tratamento, **58% do teste tinha cópia no treino** e a acurácia chegou a 98%.

### 3.3 Como identificamos a mesma foto

Comparar os bytes do arquivo (MD5) não basta: a mesma foto em outro tamanho ou formato gera um arquivo diferente. Usamos dois critérios:

1. **Hash perceptual (dHash)** — para o `Dataset_Gabungan`.
   - A imagem é reduzida para 17×16 em tons de cinza.
   - Cada pixel é comparado com o vizinho da direita (mais claro ou mais escuro), gerando 256 bits.
   - Duas imagens são a mesma foto quando diferem em poucos bits. Isso resiste a mudança de tamanho, formato e compressão.
   - **Limiar de 27 bits, calibrado visualmente:** até 27 bits todos os pares conferidos eram a mesma foto; a partir de 31, fotos diferentes (figuras em `results/calibracao_limiar/`).

2. **Nome do arquivo** — para o MCVSLD.
   - Rotação e espelhamento mudam o hash, então as 15 versões são identificadas pelo nome (`MSL_01_01_1.jpg`, `MSL_01_01_ORIGINAL.jpg`…). Fica apenas a versão `_ORIGINAL`.

**Regra adotada: só é removida uma imagem quando ela é a mesma foto de outra.**

### 3.4 Da coleta à base final

| Passo | Imagens |
|---|---|
| `Dataset_1` + `Dataset_2` (Kaggle) | 1.429 |
| MCVSLD nas 4 classes: 7.920 → só a versão `_ORIGINAL` | + 528 |
| **Candidatas** | **1.957** |
| Cópias da mesma foto removidas (inclui 1 foto com rótulos conflitantes) | − 683 |
| **Base final (fotos únicas)** | **1.274** |

Nenhuma foto do MCVSLD é cópia de uma foto do Kaggle.

| Classe | Fotos únicas | Treino | Validação | Teste |
|---|---|---|---|---|
| Chickenpox | 175 | 122 | 27 | 26 |
| Measles | 141 | 99 | 21 | 21 |
| Monkeypox | 554 | 387 | 83 | 84 |
| Normal | 404 | 283 | 60 | 61 |
| **Total** | **1.274** | **891** | **191** | **192** |

## 4. Etapas do experimento

| Etapa | Script | Em uma frase |
|---|---|---|
| 1 | `src/01_explorar_dados.py` | Conhecer as bases e encontrar as cópias |
| 2 | `src/02_dividir_dados.py` | Montar treino, validação e teste sem vazamento |
| 3 | `src/03_treinar_baseline.py` | Treinar uma CNN do zero como referência |
| 4 | `src/04_treinar_finetuning.py` | Ajustar uma ResNet50 pré-treinada |
| 5 | `src/05_avaliar.py` | Medir o desempenho no teste e analisar os erros |

O conjunto de teste só é usado na etapa 5.

### Etapa 1 — Análise exploratória

**Objetivo:** conhecer a base antes de treinar.

- Lê as duas bases e alinha as classes.
- Mostra imagens por classe e por fonte, formato e resolução de cada fonte e cor média por classe e fonte.
- Calcula MD5 e dHash e agrupa as imagens que são a mesma foto.
- Gera `data/processed/imagens.csv` e as figuras em `results/`: distribuição de classes, exemplos, histograma de distâncias e pares agrupados.

### Etapa 2 — Pré-processamento

**Objetivo:** montar treino, validação e teste sem vazamento.

- **Rótulos conflitantes:** a mesma foto com duas classes é ambígua e sai da base (1 caso).
- **Uma imagem por foto:** garante que a mesma foto nunca esteja em dois conjuntos.
- **Divisão estratificada 70/15/15:** mantém a proporção das classes em cada conjunto (seed 42).
- **Padronização RGB · 256×256 · PNG:** todas as imagens ficam iguais em formato e tamanho, para que a fonte não seja reconhecível pelo arquivo.

### Etapas 3 e 4 — Treinamento

As duas redes recebem a mesma entrada e produzem a mesma saída; o que muda é como as características são extraídas.

![Arquiteturas comparadas](docs/figuras/fig3_arquiteturas.png)

- **Baseline (etapa 3):** 4 blocos `Conv → BatchNorm → ReLU → MaxPool` (32 → 256 filtros), pooling global, dropout de 0,3 e camada linear. Adam com lr 1e-3, até 30 épocas. Mostra até onde se chega aprendendo tudo do zero.
- **Fine-tuning (etapa 4):** ResNet50 com pesos do ImageNet; a última camada é trocada para 4 classes e **todas as camadas são ajustadas**. Reaproveita o que a rede já sabe sobre bordas, texturas e formas. A taxa de aprendizado é escolhida por **busca em {1e-3, 1e-4, 1e-5}** — a faixa usada nos artigos de referência.

**Configuração comum às duas redes:**

| Item | Escolha | Motivo |
|---|---|---|
| Entrada | 224×224, normalização do ImageNet | Padrão para modelos pré-treinados |
| Augmentation (só treino) | Crop aleatório, flip horizontal, rotação ±15°, variação de cor | Base pequena; reduz overfitting |
| Perda | Entropia cruzada com pesos de classe | Classes desbalanceadas (Monkeypox ≈ 4× Measles) |
| Seleção do modelo | Melhor F1-macro na validação + early stopping (paciência 5) | F1-macro trata todas as classes igualmente |

### Etapa 5 — Avaliação

**Objetivo:** medir o desempenho final e entender os erros.

- Acurácia, precisão, recall e F1 por classe no conjunto de teste.
- Matriz de confusão, curvas de aprendizado e exemplos de acertos e erros (`results/`).

## 5. Resultados

### 5.1 Visão geral

| Condição | Base | F1-macro (val.) | Acurácia (teste) | F1-macro (teste) |
|---|---|---|---|---|
| A. Baseline | Gabungan com vazamento | 0,56 | 61,5% | 0,534 |
| B. Fine-tuning | Gabungan com vazamento | 0,979 | 98,1% | 0,974 |
| C. Baseline | Gabungan + MCVSLD sem cópias | 0,515 | 45,8% | 0,430 |
| **D. Fine-tuning** | **Gabungan + MCVSLD sem cópias** | **0,881** | **85,9%** | **0,836** |

A e B usam só o `Dataset_Gabungan` (1.429 imagens, divisão aleatória). C e D usam a base final de 1.274 fotos únicas.

- **O vazamento inflava o resultado:** o fine-tuning caiu de 0,974 para 0,836 de F1-macro no teste quando as cópias foram removidas.
- **Transfer learning faz diferença:** com os mesmos dados, a ResNet50 quase dobra o F1-macro da CNN do zero (0,836 vs. 0,430).
- **Validação e teste estão próximos** em D (0,881 vs. 0,836), o que indica que a seleção pela validação não superestimou muito o modelo.

### 5.2 Convergência

| Modelo | Épocas | Melhor época | Perda treino | Perda validação | Leitura |
|---|---|---|---|---|---|
| Baseline | 16 | 11 | 1,16 | 1,15 | **Underfitting:** perdas altas e iguais; a rede não consegue aprender as classes |
| Fine-tuning (lr 1e-4) | 19 | 14 | 0,03 | 0,39 | **Overfitting:** decora o treino; o early stopping guarda o melhor ponto da validação |

Busca da taxa de aprendizado (fine-tuning):

| lr | F1-macro (val.) | Épocas |
|---|---|---|
| 1e-3 | 0,804 | 20 |
| **1e-4** | **0,881** | 19 |
| 1e-5 | 0,798 | 20 |

O lr 1e-4 é o mesmo usado por Ahmed et al. (2026). Com 1e-3 os pesos pré-treinados são alterados demais; com 1e-5 o ajuste é lento demais para 20 épocas.

### 5.3 Desempenho por classe (teste)

| Classe | Precisão | Recall | F1 (fine-tuning) | F1 (baseline) | Imagens |
|---|---|---|---|---|---|
| Chickenpox | 0,63 | 0,73 | **0,68** | 0,36 | 26 |
| Measles | 0,86 | 0,86 | 0,86 | 0,28 | 21 |
| Monkeypox | 0,86 | 0,86 | 0,86 | 0,47 | 84 |
| Normal | 0,98 | 0,92 | **0,95** | 0,62 | 61 |

Precisão e recall referem-se ao fine-tuning.

Matriz de confusão do fine-tuning (linhas = classe real, colunas = previsão):

| Real \ Previsto | Chickenpox | Measles | Monkeypox | Normal |
|---|---|---|---|---|
| **Chickenpox** | **19** | 1 | 6 | 0 |
| **Measles** | 0 | **18** | 3 | 0 |
| **Monkeypox** | 11 | 0 | **72** | 1 |
| **Normal** | 0 | 2 | 3 | **56** |

### 5.4 Análise dos erros

- **A confusão principal é entre Monkeypox e Chickenpox:** 17 dos 27 erros. 11 casos de Monkeypox foram previstos como Chickenpox e 6 no sentido contrário. As duas doenças produzem vesículas e pústulas parecidas; nos exemplos de erro, close-ups de uma única pústula de catapora são previstos como mpox.
- **Chickenpox é a classe mais difícil** (F1 0,68): recebe muitos falsos positivos de Monkeypox, que é a classe majoritária.
- **Normal é a mais fácil** (F1 0,95), mas parte disso pode vir do estilo da foto e não da pele: no MSID, as fotos normais são retratos de estúdio, bem diferentes das fotos de lesões (ver seção 6).
- **Measles** acerta 18 de 21; os 3 erros vão para Monkeypox.

### 5.5 Comparação com a literatura

Vajpayee et al. (2023) reportam 92,5% de acurácia no MSID, mas sem conjunto de teste independente e sem verificar duplicatas. Os 85,9% obtidos aqui vêm de um teste com fotos que o modelo nunca viu, em uma base com duas fontes. Os números não são diretamente comparáveis, mas a diferença é coerente com o efeito do vazamento medido na seção 5.1.

## 6. Limitações e próximos passos

| Limitação | Possível melhoria |
|---|---|
| A classe Normal reúne retratos de estúdio (MSID) e close-ups de pele (MCVSLD); o modelo pode aprender o estilo da foto | Avaliar o desempenho separado por fonte; usar Grad-CAM para ver onde o modelo olha |
| A divisão é por foto, não por paciente; o MCVSLD tem várias fotos do mesmo paciente | Dividir por paciente no MCVSLD |
| O hash pega a mesma foto redimensionada ou recomprimida, mas pode não pegar a mesma foto recortada de outro jeito | Comparar por embeddings de uma rede pré-treinada |
| Teste pequeno (21 imagens de Measles): cada erro muda o recall em quase 5 pontos | Validação cruzada estratificada |
| Overfitting no fine-tuning | Congelar as primeiras camadas; augmentation mais forte |

## 7. Como executar

1. Instale as dependências. Para usar GPU, instale antes o PyTorch com CUDA pelo comando indicado em pytorch.org.

   ```
   pip install -r requirements.txt
   ```

2. Configure o acesso ao Kaggle. O `kagglehub` baixa o `Dataset_Gabungan` automaticamente.
3. Baixe o MCVSLD no Mendeley Data e extraia em `data/raw/`, de forma que exista `data/raw/Skin Lesion Dataset/train`, `val` e `test`.
4. Rode os scripts a partir da raiz do repositório, nesta ordem:

   ```
   python src/01_explorar_dados.py
   python src/02_dividir_dados.py
   python src/03_treinar_baseline.py
   python src/04_treinar_finetuning.py
   python src/05_avaliar.py
   ```

### Estrutura

```
.
├── data/
│   ├── raw/            # MCVSLD extraído
│   ├── processed/      # CSVs gerados (imagens, divisão, históricos, métricas)
│   └── split/          # train/val/test gerados pela etapa 2
├── docs/               # artigos de referência
│   └── figuras/        # diagrama das arquiteturas (fonte editável: figuras.html)
├── models/             # baseline.pt e finetuning.pt
├── results/            # figuras geradas pelas etapas 1 e 5
├── src/                # scripts das etapas 1 a 5
└── requirements.txt
```

## Referências

- Pai, V. R. et al. (2023). *Identification and Classification of Skin Diseases using Deep Learning Techniques*. Research Square. DOI: 10.21203/rs.3.rs-2628782/v1
- Vajpayee, H. et al. (2023). *An effective classification of Skin Disease using Deep Learning Techniques*. Confluence 2023, IEEE. DOI: 10.1109/Confluence56041.2023.10048840
- Ahmed, S., Sakib, K. M. S., & Hany, U. (2026). *Hybrid deep learning architecture for skin disease classification*. Franklin Open, 15, 100556. DOI: 10.1016/j.fraope.2026.100556
- Bala, D., & Hossain, M. S. (2023). *Monkeypox Skin Images Dataset (MSID)*. Mendeley Data. DOI: 10.17632/r9bfpnvyxr.6
- Ahmed, K. R. et al. (2024). *Multi-Class Viral Skin Lesion Dataset (MCVSLD)*. Mendeley Data. DOI: 10.17632/dfztdtfsxz.1
- Prayoga, L. F. (2026). *Infectious Skin Disease Dataset & Classification using Hybrid MobileNetV2-Xception with CLAHE Image Enhancement*. Universitas Airlangga.
