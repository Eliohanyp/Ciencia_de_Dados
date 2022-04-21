
## Autores

- [GitHub: @Eliohanyp](https://github.com/Eliohanyp)
- [LinkedIn: Eliohan Yamauti Poiati](https://www.linkedin.com/in/eliohanyp/)


# Estudos de Machine Learning e Data Science

[Repositorio Criado em: 21/04/2022]

Este repositorio contará com notebooks de estudos e conhecimentos aplicados pelo Eliohan Yamauti Poiati, Todos os codigos e conhecimentos aplicados foram adquiridos no trabalho e de cursos que fiz com o passar dos anos.

O 1º Topico será sobre Machine Learning e Data Science com PYTHON (Estudos_MachineLearning_01.ipynb)

Dentro deste notebook citado teremos diversas categorias e uma ordenação do conteudo para um aprendizado completo.

# - [Estudos_MachineLearning_01.ipynb](https://colab.research.google.com/drive/1uyD39XFVo-tEezW6q58shNrX-BqAjCEU?usp=sharing)

👨‍💻 Notebook de estudos de Machine Learning com Python by Eliohan Y. Poiati 👨‍💻

Este notebook abordará primeiro sobre a tarefa de classificação da aprendizagem de máquina.

### CLASSIFICAÇÃO: 

🔸 1. Pré-processamento e preparação de bases de dados para classificação🔸

🔸 2. Aprendizagem bayesiana (algoritmo Naive Bayes). 🔸

🔸 3. Aprendizagem por árvores de decisão (algoritmo básico de árvores e Random Forest). 🔸

🔸 4. Aprendizagem por regras. 🔸

🔸 5. Aprendizagem baseada em instâncias (algoritmo kNN). 🔸

🔸 6. Regressão logística. 🔸

🔸 7. Máquinas de vetores de suporte (SVM).🔸

🔸 8. Redes neurais artificiais. 🔸

🔸 9. Avaliação de algoritmos de classificação. 🔸

🔸 10. Combinação e rejeição de classificadores. 🔸

📚Bons Estudos!  😉😉😉

# 🔸 1. Pré-processamento e preparação de bases de dados para classificação🔸
🔧 Pré-Processamento de Dados com Pandas e Scikit-Learn 🔧

🔸 O pré processamento serve para voce deixar as base de dados consistente para o machine learning.

🔸 Serão usados uma base de dados de Crédito e Censo.

🔸 Para o préprocessamento utilizaremos valores inconsistentes e valores faltantes em nossa base para simular o maximo possivel os dados da vida real.

🔸 Trabalharemos com escalonamento de atributos para deixar os valores numericos na mesma escala e tambem trabalharemos com transformação de variaveis categoricas.

🔸 No PANDAS iremos usar varios recursos como: Localizar, remover linhas e colunas, alterar valores... 

🔸 E para finalizar o modulo darei uma introdução de como avaliar o algoritmo e como separar base de treino e base de teste.

## Tipos de variaveis

![TIpos de variaveis](https://raw.githubusercontent.com/Eliohanyp/Ciencia_de_Dados/f512f68879c14b9d3013a8a9b9fef3b7ff07356d/Images_ML-DS/Tipos%20de%20Vari%C3%A1veis.PNG)

## Importação e instalação das bibliotecas
#### Instalação das bibliotecas utilizadas

```bash
!pip install pandas seaborn matplotlib numpy
!pip -q install plotly --upgrade
!pip -q install yellowbrick
```
#### Importação das bibliotecas utilizadas

```bash
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
```
### Exploração de Dados
#### Base de dados de crédito 

 * Fonte (adaptado): https://www.kaggle.com/laotse/credit-risk-dataset

Baixe a base de dados e faça upload para sua maquina ou Google Colab

Leia o arquivo .csv com o Pandas utilizando: 
```bash
base_credito = pd.read_csv('credit_risk_dataset.csv')
```
Para visualizar alguns registros, só chamar a base de credito:
```bash
base_credito
```
![Base de Crédito](https://raw.githubusercontent.com/Eliohanyp/Ciencia_de_Dados/main/Images_ML-DS/Base_credito.PNG)

Podemos observar que são 32.581 linhas e 12 colunas, agora iremos fazer uma analise básica da nossa base de dados.

Se quisermos ver o 10 primeiros podemos usar .head()
```bash
base_credito.head(10)
```
![Base de Crédito](https://raw.githubusercontent.com/Eliohanyp/Ciencia_de_Dados/main/Images_ML-DS/Base_credito_head.PNG)

Se quisermos ver o 10 ultimos podemos usar .tail()
```bash
base_credito.tail(10)
```
![Base de Crédito](https://raw.githubusercontent.com/Eliohanyp/Ciencia_de_Dados/main/Images_ML-DS/Base_credito_tail.PNG)

Se quisermos a informação das colunas e seus tipos .info()
```bash
base_credito.info()
```
![Base de Crédito](https://raw.githubusercontent.com/Eliohanyp/Ciencia_de_Dados/main/Images_ML-DS/Base_credito_info.PNG)

Se quisermos uma analise rapida podemos usar o .describe()
```bash
base_credito.describe()
```
![Base de Crédito](https://raw.githubusercontent.com/Eliohanyp/Ciencia_de_Dados/main/Images_ML-DS/Base_credito_describe.PNG)

Se queremos ativar um filtro especifico podemos fazer assim: 
```bash
base_credito[base_credito['person_age'] >= 90]
```
Dessa forma irá aparecer somente quem tiver com idade acima de 90
![Base de Crédito](https://raw.githubusercontent.com/Eliohanyp/Ciencia_de_Dados/main/Images_ML-DS/Base_credito_idade.PNG)

### Visualização de Dados
#### Base de dados de crédito 
Leia o arquivo .csv com o Pandas utilizando: 
```bash
base_credito = pd.read_csv('credit_risk_dataset.csv')
```
agora iremos usar a biblioteca NumPy e utilizaremos a função .unique([]) que mostra os valores unicos da coluna que for especificada
```bash
np.unique(base_credito['cb_person_default_on_file'])
```
![Base de Crédito](https://raw.githubusercontent.com/Eliohanyp/Ciencia_de_Dados/main/Images_ML-DS/np_unique.PNG)

E se queremos saber quantos foram 'N' e quantos foram 'Y' podemos adicionar um return_counts=True assim:
```bash
np.unique(base_credito['cb_person_default_on_file'], return_counts=True)
```
![Base de Crédito](https://raw.githubusercontent.com/Eliohanyp/Ciencia_de_Dados/main/Images_ML-DS/np_unique_count.PNG)

Assim conseguimos observar que temos 26.836 'N' e 5.745 'Y'.


Podemos plotar isso em um grafico com o Seaborn:
```bash
sns.countplot(x = base_credito['cb_person_default_on_file']);
```
![Base de Crédito](https://raw.githubusercontent.com/Eliohanyp/Ciencia_de_Dados/main/Images_ML-DS/Grafico.PNG)

Assim conseguimos observar em um histograma com o resultado
