# Projeto Final - Manutenção Preditiva

Este projeto faz a análise de manutenção preditiva usando um conjunto de dados de sensores de máquinas industriais.

## Estrutura do projeto

- `main.ipynb`: notebook com análise exploratória, pré-processamento, balanceamento de classes, treinamento de modelos e visualizações.
- `manutencao_preditiva.csv`: base de dados original.
- `requirements.txt`: dependências Python necessárias para reproduzir a análise.
- `imagens/`: gráficos gerados pelo notebook.

## Pontos de atenção

1. O conjunto é altamente desbalanceado. O notebook usa SMOTE para balancear a classe `falha_maquina` no conjunto de treinamento.
2. Para o modelo KNN, o conjunto numérico é escalonado com `StandardScaler` antes do treinamento.
3. Há comparações entre KNN, Decision Tree e Random Forest, sendo a Random Forest o modelo final com avaliação de matriz de confusão.

## Como executar

1. Crie e ative um ambiente virtual:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Abra o notebook:
   ```bash
   jupyter notebook main.ipynb
   ```
4. Execute as células na ordem para garantir que o pré-processamento e os modelos sejam carregados corretamente.

## Resumo Executivo

A análise compara três abordagens de classificação (KNN, Decision Tree, Random Forest) para detecção de falhas em equipamentos industriais. A melhor configuração encontrada foi uma Random Forest com `max_depth=5`, que apresentou bom recall (prioritário para reduzir falhas não detectadas) com um aumento de falsos positivos que deve ser avaliado no contexto operacional.

Veja o detalhamento das conclusões e recomendações operacionais em [CONCLUSION.md](CONCLUSION.md).
