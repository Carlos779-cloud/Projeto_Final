# Predição de Falhas em Máquinas com Machine Learning

## Sobre o projeto

Este projeto consiste no desenvolvimento de um sistema de **Machine Learning para a predição de falhas em máquinas industriais**, utilizando técnicas de análise de dados e aprendizado de máquina supervisionado.

O objetivo é analisar características operacionais das máquinas, como temperatura, velocidade de rotação, torque e desgaste da ferramenta, para identificar padrões associados à ocorrência de falhas.

Durante o projeto foram realizadas etapas de:

* Análise exploratória dos dados;
* Limpeza e preparação do dataset;
* Tratamento de variáveis categóricas;
* Análise de correlação;
* Separação entre dados de treino e teste;
* Balanceamento das classes utilizando SMOTE;
* Padronização dos dados com StandardScaler;
* Treinamento de diferentes modelos de Machine Learning;
* Comparação das métricas de desempenho;
* Análise de overfitting;
* Avaliação dos modelos por meio de matriz de confusão.

---

## Vídeo de demonstração

[Assista à demonstração do projeto](COLE_AQUI_O_LINK_DO_VIDEO)

---

---

## Como executar o projeto

### 1. Clonar o repositório

Abra o terminal e execute:

```bash
git clone https://github.com/Carlos779-cloud/Projeto_Final.git
```

Depois, entre na pasta do projeto:

```bash
cd Projeto_Final
```

---

### 2. Criar um ambiente virtual

No terminal:

```bash
python -m venv venv
```

Ative o ambiente virtual.

No Windows:

```bash
venv\Scripts\activate
```

---

### 3. Instalar as bibliotecas

Execute:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
```

---

### 4. Executar o projeto

Abra o arquivo:

```text
main.ipynb
```

no Jupyter Notebook ou no Visual Studio Code.

Depois, execute as células do notebook na sequência apresentada.

---

## Ferramentas utilizadas

* Visual Studio Code;
* Jupyter Notebook;
* Python;
* Git;
* GitHub.

---

---

## Estrutura do projeto

Projeto_Final/
│
├── imagens/
│   ├── 01_dataset_bruto_head.png
│   ├── 02_distribuicao_sensores.png
│   ├── 03_balanceamento.png
│   ├── 04_heatmap.png
│   ├── 05_boxplot_sensores_com_outliers.png
│   ├── 06_dataset_limpo_head.png
│   ├── 07_boxplot_sensores_sem_outliers.png
│   ├── 08_smote_balanceamento.png
│   ├── 09_resultados_knn.png
│   ├── 10_resultados_decision_tree.png
│   ├── 11_resultados_random_forest.png
│   ├── 12_comparativo_final_modelos.png
│   └── 13_matriz_confusao_rf.png
│
├── .gitignore
├── main.ipynb
├── manutencao_preditiva.csv
├── README.md
└── requirements.txt
```

---

## Tecnologias utilizadas

* Python
* Jupyter Notebook
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn
* Git
* GitHub

---

## Bibliotecas utilizadas

Principais bibliotecas utilizadas no projeto:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
```

## O que o sistema analisa

O projeto utiliza informações operacionais das máquinas para prever a ocorrência de falhas.

Entre as principais variáveis analisadas estão:

* Temperatura do ar;
* Temperatura do processo;
* Velocidade de rotação;
* Torque;
* Desgaste da ferramenta;
* Tipo da máquina.

A variável-alvo do projeto é:

```text
falha_maquina
```

Onde:

* `0` representa que não ocorreu falha;
* `1` representa que ocorreu falha.

---

## Objetivo

O objetivo principal é desenvolver e comparar modelos de classificação capazes de identificar máquinas com maior probabilidade de apresentar falhas.

Além disso, o projeto busca aplicar na prática os principais conceitos de:

* Python para Ciência de Dados;
* Análise exploratória de dados;
* Limpeza e preparação de dados;
* Engenharia de atributos;
* Aprendizado de Máquina supervisionado;
* Classificação binária;
* Balanceamento de classes;
* Avaliação de modelos;
* Git e GitHub.

---

## Modelos de Machine Learning utilizados

Foram treinados e avaliados diferentes algoritmos de classificação:

### K-Nearest Neighbors (KNN)

Foram testados diferentes valores de `k` para avaliar o impacto do número de vizinhos no desempenho do modelo.

### Decision Tree

Foram testadas diferentes profundidades da árvore:

```text
max_depth = 3
max_depth = 5
max_depth = None
```

Essa comparação permitiu analisar o comportamento do modelo e identificar possíveis problemas de overfitting.

### Random Forest

O modelo Random Forest foi testado utilizando diferentes profundidades máximas:

```text
max_depth = 3
max_depth = 5
max_depth = None
```

A configuração com `max_depth=None` apresentou maior complexidade e evidências de overfitting.

A configuração com `max_depth=5` apresentou melhor equilíbrio entre o desempenho nos dados de treinamento e nos dados de teste.

---

## Preparação dos dados

### Definição da variável-alvo

A variável `falha_maquina` foi definida como o alvo do modelo:

```python
y = df_limpo['falha_maquina']
```

As variáveis utilizadas como entrada foram selecionadas a partir das características operacionais da máquina.

Algumas colunas foram removidas por não serem adequadas para o treinamento, como identificadores e variáveis que poderiam causar vazamento de informação.

---

### Tratamento de variáveis categóricas

A variável `tipo`, que representa o tipo da máquina, foi convertida para um formato numérico utilizando One-Hot Encoding:

```python
pd.get_dummies(X, drop_first=True)
```

Essa técnica permite que variáveis categóricas sejam utilizadas pelos algoritmos de Machine Learning sem criar uma relação numérica artificial entre as categorias.

---

### Divisão dos dados

Os dados foram divididos em:

* 80% para treinamento;
* 20% para teste.

Foi utilizada divisão estratificada para manter a proporção das classes:

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

---

## Balanceamento das classes com SMOTE

O dataset apresentava uma quantidade muito menor de exemplos de máquinas com falha em comparação com máquinas sem falha.

Para reduzir esse desequilíbrio, foi utilizado o método **SMOTE**.

O balanceamento foi aplicado somente aos dados de treinamento:

```text
Dados originais
        ↓
Divisão treino/teste
        ↓
SMOTE aplicado apenas no treino
        ↓
Treinamento dos modelos
        ↓
Avaliação no conjunto de teste original
```

Essa abordagem evita que dados sintéticos sejam criados no conjunto de teste, mantendo uma avaliação mais próxima de um cenário real.

---

## Padronização dos dados

Foi utilizado o `StandardScaler` para padronizar as variáveis numéricas.

Essa etapa é importante principalmente para modelos sensíveis à escala dos dados, como:

* KNN;

A padronização permite que as variáveis tenham uma escala comparável durante o treinamento.

---

## Métricas utilizadas

Os modelos foram avaliados utilizando diferentes métricas:

### Accuracy

Indica a proporção geral de previsões corretas.

### Precisão

Indica, entre as máquinas classificadas como com falha, quantas realmente apresentaram falha.

### Recall

Indica a capacidade do modelo de identificar corretamente as máquinas que realmente apresentaram falha.

Neste projeto, o Recall é uma métrica especialmente importante, pois deixar de identificar uma máquina que realmente apresenta risco de falha pode gerar consequências operacionais.

### F1-Score

Representa um equilíbrio entre Precisão e Recall.

---

## Análise de Overfitting

Durante a comparação dos modelos, foi observado comportamento de overfitting em modelos muito complexos.

No Random Forest, por exemplo, a configuração:

```text
max_depth = None
```

permitiu que as árvores crescessem sem uma limitação de profundidade.

Como consequência, o modelo apresentou um desempenho muito elevado nos dados de treinamento, mas uma diferença maior em relação ao desempenho nos dados de teste.

A configuração:

```text
max_depth = 5
```

apresentou um melhor equilíbrio entre:

* Desempenho no treinamento;
* Capacidade de generalização;
* Recall;
* F1-Score.

Por esse motivo, essa configuração foi considerada mais estável para o projeto.

## Autor

**Carlos João Reinert**

Projeto desenvolvido como parte dos estudos de:

**Análise de Dados, Inteligência Artificial e Machine Learning.**
