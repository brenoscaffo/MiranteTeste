# Decision Log — Telco Customer Churn

## 1. Objetivo

Este documento registra as principais decisões técnicas adotadas durante a etapa de modelagem do projeto **Telco Customer Churn**, incluindo preparação dos dados, escolha dos algoritmos, critérios de avaliação, otimização de hiperparâmetros, desafios encontrados e próximos passos.

O objetivo do modelo é identificar clientes com maior probabilidade de churn, permitindo que a empresa priorize ações de retenção.

---

## 2. Estratégia de modelagem

Foram comparados dois algoritmos de classificação:

- **Decision Tree Classifier**
- **LightGBM Classifier**

A estratégia adotada no notebook foi começar com modelos sem otimização de hiperparâmetros, avaliar o desempenho inicial e, posteriormente, otimizar o algoritmo que apresentou melhor resultado.

As métricas utilizadas foram:

- Precision
- Recall
- F1-Score
- ROC-AUC

A curva ROC também foi utilizada para avaliar a capacidade de discriminação dos modelos.

### Justificativa das métricas

Para o problema de churn, o **Recall** possui importância especial, pois representa a proporção de clientes que realmente apresentaram churn e foram corretamente identificados pelo modelo.

Um falso negativo representa um cliente que efetivamente cancela o serviço, mas não é identificado pelo modelo como cliente em risco. Dependendo da estratégia de retenção, esse tipo de erro pode ter custo relevante para o negócio.

A **Precision** também é importante porque indica quanto das ações direcionadas aos clientes classificados como churn está sendo destinado a clientes que realmente apresentam churn.

O **F1-Score** foi utilizado como medida de equilíbrio entre Precision e Recall.

A **ROC-AUC** foi utilizada como principal métrica independente de um único threshold, permitindo avaliar a capacidade do modelo de ordenar clientes de maior risco acima dos clientes de menor risco.

---

## 3. Preparação dos dados

### 3.1 Transformação da variável-alvo

A variável `Churn` foi convertida de texto para representação binária:

```python
No  -> 0
Yes -> 1
```

A transformação utilizada foi:

```python
df["Churn"] = df["Churn"].str.strip().map({
    "No": 0,
    "Yes": 1
})
```

Essa transformação permite utilizar diretamente os classificadores e métricas do Scikit-learn.

---

### 3.2 Remoção de `customerID`

A variável `customerID` foi removida da base utilizada na modelagem.

```python
df = df.drop("customerID", axis=1)
```

**Justificativa:** o identificador representa apenas uma chave de identificação do cliente e não deve ser tratado como informação preditiva.

---

### 3.3 Feature engineering: quantidade de serviços

Foi criada a variável:

```text
qtd_servicos
```

a partir das seguintes colunas:

- PhoneService
- MultipleLines
- InternetService
- OnlineSecurity
- OnlineBackup
- DeviceProtection
- TechSupport
- StreamingTV
- StreamingMovies

Valores equivalentes à ausência de serviço foram convertidos para `0`, enquanto os demais valores foram considerados `1`. Em seguida, os valores foram somados.

Exemplo da lógica:

```python
lambda x: 0 if x in [
    "No",
    "No internet service",
    "No phone service"
] else 1
```

Após a criação de `qtd_servicos`, as variáveis de serviço originais foram removidas.

**Objetivo:** criar uma variável agregada que represente aproximadamente a quantidade de serviços contratados por cliente.

### Ponto de atenção

Essa transformação reduz a dimensionalidade, porém também remove a identidade individual de cada serviço.

Por exemplo, dois clientes podem possuir a mesma quantidade de serviços, mas combinações completamente diferentes. Essa simplificação pode eliminar informação preditiva relevante.

Uma evolução recomendada é comparar:

1. modelo utilizando apenas `qtd_servicos`;
2. modelo utilizando os serviços individualmente via One-Hot Encoding;
3. modelo utilizando os serviços individuais e `qtd_servicos` simultaneamente.

A comparação deve ser feita por validação cruzada.

---

### 3.4 Codificação das variáveis categóricas

As variáveis categóricas restantes foram transformadas por meio de:

```python
pd.get_dummies(
    df,
    drop_first=True,
    dtype="int"
)
```

O parâmetro `drop_first=True` remove uma categoria de referência de cada variável.

Também foi realizada a padronização dos nomes das colunas:

```python
df.columns = df.columns.str.replace(" ", "_")
```

---

## 4. Divisão entre treino e teste

A base foi dividida utilizando:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Resultados observados:

| Conjunto | Registros | Taxa de churn |
|---|---:|---:|
| Treino | 5.634 | 35,16% |
| Teste | 1.409 | 35,13% |

O uso de `stratify=y` foi mantido para preservar aproximadamente a mesma proporção da variável-alvo nos dois conjuntos.

---

# 5. Modelo 1 — Decision Tree

## 5.1 Configuração inicial

O primeiro modelo treinado foi:

```python
DecisionTreeClassifier()
```

Nenhum hiperparâmetro foi definido explicitamente nesta primeira avaliação.

### Resultado

| Métrica | Resultado |
|---|---:|
| Precision | 0,3713 |
| Recall | 0,4020 |
| F1-Score | 0,3860 |
| ROC-AUC | 0,5167 |

A ROC-AUC ficou próxima de `0.5`, indicando baixa capacidade discriminativa nessa configuração.

Por esse motivo, a Decision Tree não foi selecionada para a etapa de otimização apresentada no notebook.

---

# 6. Modelo 2 — LightGBM

## 6.1 Justificativa

O LightGBM foi escolhido como segundo algoritmo por ser um método de gradient boosting baseado em árvores, capaz de capturar relações não lineares e interações entre variáveis.

A configuração inicial utilizada foi:

```python
lgb.LGBMClassifier()
```

### Resultado inicial

| Métrica | Resultado |
|---|---:|
| Precision | 0,4171 |
| Recall | 0,5232 |
| F1-Score | 0,4642 |
| ROC-AUC | 0,5908 |

Comparado à Decision Tree, o LightGBM apresentou melhora em todas as métricas observadas.

Por esse motivo, o LightGBM foi selecionado para a etapa seguinte de otimização.

---

# 7. Otimização de hiperparâmetros

Foi utilizado `GridSearchCV` para explorar diferentes configurações do LightGBM.

O espaço de busca utilizado foi:

```python
{
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [300, 500, 900],
    "max_depth": [3, 5, 6, 8],
    "subsample": [0.5, 0.75, 1]
}
```

A configuração do Grid Search foi:

```python
GridSearchCV(
    estimator=lgb.LGBMClassifier(),
    param_grid=params,
    cv=3,
    scoring="roc_auc"
)
```

### Decisão

O critério de seleção adotado foi a **ROC-AUC média durante a validação cruzada**.

O uso de validação cruzada permite comparar configurações utilizando apenas a base de treinamento, reduzindo a dependência de uma única divisão interna dos dados.

---

## 7.1 Melhor configuração encontrada

O melhor estimador retornado apresentou, entre seus parâmetros:

```python
learning_rate = 0.01
n_estimators = 300
max_depth = 3
subsample = 0.5
```

O modelo também manteve valores padrão do LightGBM para parâmetros que não foram incluídos na busca, como:

```text
num_leaves = 31
min_child_samples = 20
reg_alpha = 0.0
reg_lambda = 0.0
```

---

## 7.2 Resultado após otimização

| Métrica | LightGBM inicial | LightGBM otimizado |
|---|---:|---:|
| Precision | 0,4171 | **0,4300** |
| Recall | 0,5232 | **0,7192** |
| F1-Score | 0,4642 | **0,5382** |
| ROC-AUC | 0,5908 | **0,6272** |

A otimização apresentou ganho principalmente em **Recall**, que passou de aproximadamente `52,3%` para `71,9%`.

Também houve melhora no F1-Score e na ROC-AUC.

Com base nos resultados apresentados no notebook, o **LightGBM otimizado foi selecionado como o melhor modelo entre as configurações testadas**.

---

# 8. MLflow

O MLflow foi utilizado para monitoramento e versionamento dos experimentos.

O tracking server foi configurado em:

```python
mlflow.set_tracking_uri(
    "http://127.0.0.1:5000/"
)
```

Também foi utilizado:

```python
mlflow.lightgbm.autolog()
```

durante o Grid Search.

Os melhores hiperparâmetros foram registrados com:

```python
mlflow.log_params(
    grid.best_params_
)
```

e as métricas finais foram registradas com:

```python
mlflow.log_metrics({
    "Precisao_teste": precisao_teste,
    "Recall_teste": recall_teste,
    "F1_teste": f1_teste,
    "Curva_roc_teste": roc_auc_teste
})
```

Essa decisão permite manter rastreabilidade entre configurações de modelo, hiperparâmetros e desempenho.

---

# 9. Threshold de classificação

No notebook, o threshold utilizado para transformar probabilidades em classes foi calculado a partir da probabilidade média observada entre clientes da classe positiva:

```python
metricas_light = (
    teste_light
    .groupby("Churn")["Probabilidade"]
    .mean()
)
```

e posteriormente:

```python
y_pred_light = (
    y_probab_light >= metricas_light[1]
).astype(int)
```

O mesmo threshold foi reaproveitado na avaliação do LightGBM otimizado.

## Ponto de atenção metodológico

Esse threshold foi calculado utilizando `y_test`.

Como o valor real da variável-alvo do conjunto de teste participa da escolha do corte, existe **vazamento de informação do conjunto de teste para a regra de decisão**.

Por isso, as métricas de Precision, Recall e F1 apresentadas no notebook devem ser interpretadas com cautela.

### Decisão recomendada para a versão final

O threshold deve ser escolhido utilizando:

- conjunto de validação; ou
- previsões *out-of-fold* produzidas por validação cruzada no conjunto de treino.

Somente após a escolha definitiva do threshold o conjunto de teste deve ser utilizado para a avaliação final.

Um fluxo recomendado é:

```text
Treino
  |
  +--> treinamento / GridSearchCV
  |
  +--> probabilidades de validação
  |
  +--> escolha do threshold
  |
  +--> modelo + threshold congelados
  |
  +--> avaliação única no conjunto de teste
```

---

# 10. Reprodutibilidade

O `train_test_split` utiliza:

```python
random_state=42
```

garantindo reprodutibilidade da divisão entre treino e teste.

Entretanto, os modelos apresentados no notebook foram criados sem `random_state` explícito:

```python
DecisionTreeClassifier()
LGBMClassifier()
```

### Próxima melhoria

Definir explicitamente:

```python
random_state=42
```

nos classificadores utilizados na versão final.

Exemplo:

```python
lgb.LGBMClassifier(
    random_state=42
)
```

Isso deixa o pipeline mais consistente e reproduzível.

---

# 11. Principais desafios encontrados

## 11.1 Escolha do threshold

A classificação padrão utiliza normalmente o corte de `0.5`, mas esse valor não necessariamente representa o melhor equilíbrio para churn.

Reduzir o threshold tende a:

- aumentar Recall;
- aumentar falsos positivos;
- reduzir Precision em determinados cenários.

A escolha do corte deve considerar o custo de uma ação de retenção e o custo de perder um cliente.

---

## 11.2 Equilíbrio entre Precision e Recall

O modelo otimizado aumentou substancialmente o Recall, chegando a aproximadamente `71,9%`, mas a Precision permaneceu em aproximadamente `43,0%`.

Isso demonstra um trade-off relevante:

- mais clientes que realmente apresentam churn são identificados;
- porém também aumenta a quantidade de clientes classificados como risco que não necessariamente apresentarão churn.

A decisão final deve ser associada à capacidade operacional da estratégia de retenção.

---

## 11.3 Representação dos serviços contratados

A criação de `qtd_servicos` simplifica várias variáveis em uma única feature.

Essa transformação facilita a modelagem, mas pode remover informações relacionadas ao tipo específico de serviço contratado.

Uma comparação adicional deve avaliar se manter as variáveis originais melhora a capacidade discriminativa.

---

## 11.4 Capacidade discriminativa ainda moderada

Mesmo após otimização, a ROC-AUC observada no teste foi:

```text
0.6272
```

O resultado representa uma melhora em relação aos modelos iniciais, mas indica que ainda existe espaço para evolução da modelagem e da engenharia de features.

---

# 12. Lições aprendidas

1. Um modelo mais sofisticado não garante bom desempenho sem uma representação adequada das variáveis.

2. A escolha de hiperparâmetros deve ser baseada em validação cruzada, e não no desempenho repetido sobre o conjunto final de teste.

3. O threshold é uma decisão separada do treinamento do modelo e deve refletir o objetivo de negócio.

4. ROC-AUC não depende de um único threshold, enquanto Precision, Recall e F1 variam conforme o corte escolhido.

5. Para churn, Recall pode ser especialmente importante quando o custo de não identificar um cliente prestes a cancelar é alto.

6. Feature engineering deve ser validada empiricamente. Reduzir várias features a uma única variável pode ajudar na simplicidade, mas também pode remover informação útil.

7. MLflow melhora a rastreabilidade das decisões e facilita a comparação entre experimentos.

---

# 13. Próximos passos

Antes da entrega final, os próximos passos recomendados são:

- selecionar o threshold sem utilizar `y_test`;
- adicionar `random_state=42` aos modelos;
- comparar `qtd_servicos` com a utilização das variáveis de serviço originais;
- avaliar uma estratégia de validação cruzada estratificada;
- analisar matriz de confusão do modelo final;
- interpretar as features do LightGBM utilizando SHAP ou outra técnica de interpretabilidade;
- registrar no MLflow o threshold final utilizado;
- salvar o modelo final juntamente com todas as transformações necessárias para inferência;
- manter o conjunto de teste reservado para uma única avaliação final.

---

# 14. Decisão final registrada

Com base exclusivamente nos experimentos apresentados no notebook, o modelo selecionado foi o **LightGBM otimizado por GridSearchCV**, por apresentar o melhor desempenho entre os modelos testados.

Configuração principal encontrada:

```text
learning_rate = 0.01
n_estimators = 300
max_depth = 3
subsample = 0.5
```

Resultados apresentados:

```text
Precision = 0.4300
Recall    = 0.7192
F1-Score  = 0.5382
ROC-AUC   = 0.6272
```

A seleção deve ser considerada **provisória até a correção da estratégia de threshold**, pois o corte utilizado no notebook atual foi calculado utilizando informações do conjunto de teste.

Depois dessa correção, as métricas finais devem ser recalculadas e registradas como resultado oficial do projeto.
