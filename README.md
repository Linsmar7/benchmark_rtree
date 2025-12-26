# R*-Tree Benchmark & Validação

Benchmark para indexação espacial utilizando uma R*-Tree (via `libspatialindex`). Ele avalia performance de construção e consulta (k-NN e Range Query), comparando com uma busca linear (Ground Truth) para validação.

## Estrutura do Projeto

*   **`benchmark_rstar.cpp`**: Código principal para criação do índice e execução de benchmarks de performance.
*   **`validar.cpp`**: Código para verificar a corretude das consultas k-NN calculando o Recall em comparação com uma varredura linear exata.
*   **`detalhes_execucao.csv`**: Log gerado pelo benchmark com tempos e estatísticas.
*   **`validacao_detalhada.csv`**: Log gerado pela validação com métricas de Recall.

## Pré-requisitos

*   Compilador C++ (g++)
*   Biblioteca `libspatialindex`

## Compilação

Para compilar os arquivos, utilize os seguintes comandos:

```bash
# Compilar o Benchmark
g++ benchmark_rstar.cpp -o benchmark -lspatialindex -O3 -std=c++17

# Compilar a Validação
g++ validar.cpp -o validar -lspatialindex -O3 -std=c++17
```

## Execução

### 1. Rodar o Benchmark

O benchmark agora aceita parâmetros via linha de comando para suportar múltiplos datasets. O sistema gera automaticamente arquivos de consulta separados para k-NN e Range na pasta `queries/`.

**Sintaxe**:
```bash
./benchmark <caminho_dataset> <dimensao>
```

**Exemplo**:
```bash
./benchmark ../datasets_processed/Imagenet32_train/color_32.txt 32
```

O programa irá:
1. Usar o nome do arquivo do dataset (ex: `color_32`) para criar o índice `rtree_index_color_32`.
2. Gerar (se não existirem) os arquivos `queries/color_32_knn.csv` e `queries/color_32_range.csv`.
3. Executar as consultas e salvar os resultados em `results/benchmark_color_32.csv`.

### 2. Rodar a Validação de k-NN

A validação compara os resultados do k-NN com uma busca linear exata.

**Sintaxe**:
```bash
./validar <caminho_dataset> <dimensao>
```

**Exemplo**:
```bash
./validar ../datasets_processed/Imagenet32_train/color_32.txt 32
```

O programa irá ler o arquivo `queries/color_32_knn.csv` e comparar os resultados. Os resultados serão salvos em `results/validacao_color_32.csv`.
