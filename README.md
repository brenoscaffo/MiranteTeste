# Telco Customer Churn

Projeto de Ciência de Dados desenvolvido com o objetivo de identificar clientes com maior probabilidade de churn em uma empresa de telecomunicações.

A solução contempla análise exploratória, preparação dos dados, comparação de modelos de classificação, otimização de hiperparâmetros, rastreamento de experimentos com MLflow e interpretação do modelo final com SHAP.

---

## Objetivo

O objetivo principal é construir um modelo capaz de identificar clientes com maior risco de cancelamento, permitindo que a empresa direcione ações de retenção para os grupos mais relevantes.

O problema foi tratado como uma classificação binária:

- `0`: cliente sem churn
- `1`: cliente com churn

---

## Dataset

A base utilizada contém informações cadastrais, contratuais, financeiras e de utilização de serviços dos clientes.

Entre as principais variáveis estão:

- `gender`
- `SeniorCitizen`
- `Partner`
- `Dependents`
- `tenure`
- `Contract`
- `PaperlessBilling`
- `PaymentMethod`
- `MonthlyCharges`
- `TotalCharges`
- `Churn`

Durante a preparação dos dados, a variável `customerID` foi removida da modelagem por funcionar apenas como identificador.

Também foi criada a feature `qtd_servicos`, que representa a quantidade de serviços contratados por cada cliente.

As variáveis categóricas foram convertidas para representação numérica utilizando One-Hot Encoding.

---

## Estrutura do projeto

```text
telco-customer-churn/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   └── 02_Modeling.ipynb
│
├── src/
│   ├── data/
│   │   └── data_preprocessing.py
│   └── models/
│       ├── train.py
│       └── tune_model.py
│
├── scripts/
│   ├── data_preprocessed.py
│   └── eda.py
│
├── reports/
│   └── Decision_Log.md
│
├── requirements.txt
├── Dockerfile
├── Makefile
├── .gitignore
├── .dockerignore
└── README.md
```

---

## Tecnologias utilizadas

- Python
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- MLflow
- SHAP
- Matplotlib
- Seaborn
- Jupyter
- Docker
- Make

---

## Estratégia de modelagem

Foram comparados inicialmente dois algoritmos:

1. Decision Tree Classifier
2. LightGBM Classifier

A divisão dos dados foi realizada com `train_test_split`, utilizando 80% dos dados para treinamento e 20% para teste.

Foi utilizado `stratify=y` para preservar aproximadamente a mesma proporção de churn entre os conjuntos.

```text
Treino: 5.634 registros | Taxa de churn: 35,16%
Teste:  1.409 registros | Taxa de churn: 35,13%
```

As métricas utilizadas para comparação foram:

- Precision
- Recall
- F1-Score
- ROC-AUC

O Recall recebeu atenção especial devido ao contexto de negócio, pois um falso negativo representa um cliente que realmente apresenta churn, mas não é identificado pelo modelo como cliente em risco.

---

## Resultados iniciais

### Decision Tree

O primeiro modelo testado foi uma árvore de decisão sem otimização de hiperparâmetros.

| Métrica | Resultado |
|---|---:|
| Precision | 0.3713 |
| Recall | 0.4020 |
| F1-Score | 0.3860 |
| ROC-AUC | 0.5167 |

A ROC-AUC ficou próxima de 0.5, indicando baixa capacidade discriminativa para essa configuração.

### LightGBM

O segundo modelo testado foi o LightGBM utilizando inicialmente os parâmetros padrão.

| Métrica | Resultado |
|---|---:|
| Precision | 0.4171 |
| Recall | 0.5232 |
| F1-Score | 0.4642 |
| ROC-AUC | 0.5908 |

O LightGBM apresentou desempenho superior à Decision Tree em todas as métricas avaliadas e, por esse motivo, foi escolhido para a etapa de otimização.

---

## Otimização do LightGBM

A otimização de hiperparâmetros foi realizada com `GridSearchCV`, utilizando ROC-AUC como critério de seleção.

O espaço de busca incluiu:

```python
{
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [300, 500, 900],
    "max_depth": [3, 5, 6, 8],
    "subsample": [0.5, 0.75, 1]
}
```

A validação cruzada foi realizada com 3 folds.

A melhor configuração encontrada foi:

```python
LGBMClassifier(
    learning_rate=0.01,
    max_depth=3,
    n_estimators=300,
    subsample=0.5
)
```

---

## Resultado do modelo final

Após a otimização, o LightGBM apresentou:

| Métrica | LightGBM inicial | LightGBM otimizado |
|---|---:|---:|
| Precision | 0.4171 | **0.4300** |
| Recall | 0.5232 | **0.7192** |
| F1-Score | 0.4642 | **0.5382** |
| ROC-AUC | 0.5908 | **0.6272** |

O principal ganho ocorreu no Recall, que passou de aproximadamente 52,3% para 71,9%.

Isso significa que o modelo otimizado conseguiu identificar uma parcela significativamente maior dos clientes que efetivamente apresentaram churn.

---

## Escolha do modelo

O LightGBM otimizado foi selecionado como modelo final entre as alternativas avaliadas.

A decisão foi baseada principalmente em quatro pontos:

- apresentou desempenho superior à Decision Tree;
- obteve maior ROC-AUC;
- apresentou aumento relevante no Recall após a otimização;
- permite interpretar as previsões utilizando SHAP.

No contexto de churn, o aumento do Recall é especialmente relevante porque reduz a quantidade de clientes que cancelariam o serviço sem serem identificados previamente pelo modelo.

Existe, porém, um trade-off entre Recall e Precision. O modelo final alcançou Recall de aproximadamente 71,9%, enquanto a Precision permaneceu em aproximadamente 43,0%. Isso significa que a estratégia consegue capturar mais clientes em risco, mas também pode direcionar ações de retenção para clientes que não necessariamente apresentariam churn.

A escolha do ponto de corte da probabilidade deve, portanto, considerar o custo de uma ação de retenção em comparação com o custo de perder um cliente.

---

## Threshold de classificação

As probabilidades previstas pelo modelo foram transformadas em classes por meio de um threshold.

No experimento apresentado no notebook, foi utilizado aproximadamente:

```text
0.385
```

Valores de probabilidade iguais ou superiores ao threshold são classificados como churn.

```python
y_pred = (y_prob >= threshold).astype(int)
```

A alteração do threshold permite controlar o trade-off entre Precision e Recall.

> **Observação metodológica:** para uma avaliação final sem vazamento de informação, o threshold deve ser selecionado em um conjunto de validação ou por previsões out-of-fold no conjunto de treinamento, mantendo o conjunto de teste reservado para a avaliação final.

---

## Interpretabilidade com SHAP

O modelo final foi interpretado utilizando SHAP.

A análise indicou que as variáveis relacionadas ao tipo de contrato foram as que mais influenciaram as previsões do modelo.

Os principais padrões observados foram:

- contratos de dois anos apresentaram impacto predominantemente negativo sobre a previsão de churn;
- contratos de um ano também apresentaram efeito de redução do risco previsto;
- pagamento por `Electronic check` apresentou impacto positivo sobre a previsão de churn;
- `MonthlyCharges`, `tenure` e `TotalCharges` também contribuíram para as previsões, porém com magnitude menor.

Esses resultados indicam que clientes sem contratos de longo prazo e que utilizam `Electronic check` representam grupos que merecem atenção em estratégias de retenção.

Os valores SHAP representam o comportamento aprendido pelo modelo e não devem ser interpretados como relações causais.

---

## Rastreamento de experimentos

O MLflow foi utilizado para registrar:

- hiperparâmetros;
- métricas;
- modelos;
- experimentos de otimização.

Isso permite maior rastreabilidade e comparação entre diferentes versões do modelo.

---

## Como executar o projeto

### 1. Clonar o repositório

```bash
git clone <URL_DO_REPOSITORIO>
cd telco-customer-churn
```

### 2. Criar um ambiente virtual

```bash
python -m venv .venv
```

No Windows:

```bash
.venv\Scripts\activate
```

No Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

ou, caso o Make esteja disponível:

```bash
make install
```

---

## Execução com Makefile

Pré-processamento:

```bash
make preprocess
```

Treinamento:

```bash
make train
```

Otimização de hiperparâmetros:

```bash
make tune
```

Pipeline principal:

```bash
make all
```

Abrir Jupyter Lab:

```bash
make notebook
```

---

## Execução com Docker

Construir a imagem:

```bash
docker build -t telco-churn .
```

ou:

```bash
make docker-build
```

Executar o container:

```bash
docker run --rm -it -p 8888:8888 telco-churn
```

ou:

```bash
make docker-run
```

Após iniciar o container, o Jupyter Lab poderá ser acessado pelo endereço exibido no terminal.

---

## Notebooks

### `01_EDA.ipynb`

Contém a análise exploratória dos dados, incluindo:

- distribuição da variável-alvo;
- análise das variáveis numéricas;
- análise de contratos;
- métodos de pagamento;
- tenure;
- relações entre características dos clientes e churn.

### `02_Modeling.ipynb`

Contém:

- preparação das features;
- divisão treino/teste;
- treinamento da Decision Tree;
- treinamento do LightGBM;
- comparação das métricas;
- otimização com GridSearchCV;
- curva ROC;
- escolha do modelo final;
- interpretação com SHAP;
- rastreamento dos experimentos com MLflow.

---

## Principais conclusões

O LightGBM apresentou melhor desempenho que a Decision Tree no conjunto de experimentos realizado.

Após a otimização dos hiperparâmetros, houve um aumento relevante no Recall, passando de aproximadamente 52,3% para 71,9%.

A análise SHAP mostrou que o tipo de contrato é a informação de maior impacto nas previsões. Contratos de um e dois anos foram associados pelo modelo a menor risco previsto, enquanto o método de pagamento `Electronic check` esteve associado a maior risco previsto.

Do ponto de vista de negócio, os resultados sugerem que estratégias de retenção podem priorizar clientes identificados com maior probabilidade de churn, principalmente aqueles sem contratos de maior duração e que utilizam métodos de pagamento associados a maior risco previsto.

---

## Limitações e próximos passos

Alguns pontos podem ser explorados em evoluções futuras:

- selecionar o threshold usando apenas dados de treinamento/validação;
- ampliar a busca de hiperparâmetros;
- testar outros algoritmos de classificação;
- avaliar calibração das probabilidades;
- comparar a feature `qtd_servicos` com a utilização individual das variáveis de serviço;
- avaliar o modelo utilizando validação cruzada estratificada;
- definir o threshold com base no custo financeiro de falsos positivos e falsos negativos;
- monitorar drift e degradação das métricas após implantação.

---

## Autor

Projeto desenvolvido como estudo de classificação e previsão de churn utilizando técnicas de Ciência de Dados e Machine Learning.
