#include <spatialindex/SpatialIndex.h>
#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <random>

using namespace SpatialIndex;
using namespace std;

// --- Estruturas de Dados ---

// Estrutura para armazenar dados da busca exaustiva (Ground Truth/Verdade de Chão)
struct PointEntry {
  uint64_t id;
  vector<double> coords;
  double dist;
};

// Gera arquivos de queries (k-NN e Range) usando Reservoir Sampling
void generateQueryFiles(const string& datasetPath, const string& datasetName, uint32_t dimension, int queriesPerType = 100) {
  string queriesDir = "queries";
  if (!fs::exists(queriesDir)) fs::create_directory(queriesDir);

  string knnPath = queriesDir + "/" + datasetName + "_knn.csv";
  string rangePath = queriesDir + "/" + datasetName + "_range.csv";

  if (fs::exists(knnPath) && fs::exists(rangePath)) return;

  cout << "Gerando arquivos de queries em " << queriesDir << " ..." << endl;
  
  ifstream file(datasetPath);
  if (!file.is_open()) {
    cerr << "Erro ao abrir dataset para gerar queries: " << datasetPath << endl;
    exit(1);
  }

  vector<string> reservoir;
  string line;
  int lineCount = 0;
  int totalNeeded = queriesPerType * 2;
  
  std::mt19937 gen(12345); // Seed fixa

  while (getline(file, line)) {
    if (lineCount < totalNeeded) {
      reservoir.push_back(line);
    } else {
      std::uniform_int_distribution<> dis(0, lineCount);
      int j = dis(gen);
      if (j < totalNeeded) {
        reservoir[j] = line;
      }
    }
    lineCount++;
  }

  // Salva arquivo k-NN
  ofstream knnFile(knnPath);
  for (int i = 0; i < queriesPerType && i < reservoir.size(); ++i) {
    knnFile << reservoir[i] << "\n";
  }
  knnFile.close();

  // Salva arquivo Range
  ofstream rangeFile(rangePath);
  for (int i = queriesPerType; i < totalNeeded && i < reservoir.size(); ++i) {
    rangeFile << reservoir[i] << "\n";
  }
  rangeFile.close();
  
  cout << "Arquivos gerados: " << knnPath << " e " << rangePath << endl;
}

// --- Funções Utilitárias ---

// Calcula Distância Euclidiana (L2) entre vetor e array
double getL2(const vector<double>& p1, const double* p2, int dim) {
  double sum = 0;
  for (int i = 0; i < dim; ++i) {
    double diff = p1[i] - p2[i];
    sum += diff * diff;
  }
  return sqrt(sum);
}

// --- Visitante para Validação ---
// Coleta os IDs retornados pela busca na R-Tree para comparar com a busca linear
class ValidationVisitor : public IVisitor {
public:
  vector<uint64_t> neighborIds; // IDs dos vizinhos encontrados

  void visitNode(const INode& n) override {}
  
  // Armazena o ID de cada dado encontrado
  void visitData(const IData& d) override { neighborIds.push_back(d.getIdentifier()); }
  
  void visitData(vector<const IData*>& v) override {}
};

int main(int argc, char** argv) {
  if (argc < 3) {
    cerr << "Uso: " << argv[0] << " <caminho_dataset> <dimensao>" << endl;
    cerr << "Exemplo: " << argv[0] << " ../datasets/data.txt 128" << endl;
    return 1;
  }

  // --- PARÂMETROS VIA CLI ---
  string datasetPath = argv[1];
  unsigned int dimension = stoi(argv[2]);

  // Extrai nome do dataset
  string datasetName = fs::path(datasetPath).stem().string();

  // --- OUTROS PARÂMETROS ---
  int kNeighbors = 5;
  string baseName = "rtree_index_" + datasetName; // Carrega o índice específico do dataset
  
  if (!fs::exists("results")) {
    fs::create_directory("results");
  }
  string resultsFile = "results/validacao_" + datasetName + ".csv";
  // --------------------------------

  // --- CARREGAMENTO DO DATASET NA MEMÓRIA (GROUND TRUTH) ---
  // Necessário para fazer a Busca Linear (Força Bruta) e comparar resultados
  cout << "Carregando Ground Truth (Dataset Completo) na RAM..." << endl;
  vector<PointEntry> fullData;
  ifstream infile(datasetPath);
  string line; uint64_t currentId = 0;
  
  while (getline(infile, line)) {
    stringstream ss(line); string val; vector<double> c;
    while (getline(ss, val, ',')) c.push_back(stod(val));
    if (c.size() == dimension) fullData.push_back({currentId++, c, 0.0});
  }

  // --- CARREGAMENTO DA R-TREE ---
  IStorageManager* storage = StorageManager::loadDiskStorageManager(baseName);
  ISpatialIndex* tree = RTree::loadRTree(*storage, 1);

  // --- CARREGAMENTO DAS QUERIES ---
  generateQueryFiles(datasetPath, datasetName, dimension);
  string knnPath = "queries/" + datasetName + "_knn.csv";
  
  ifstream qfile(knnPath);
  vector<vector<double>> queries;
  while(getline(qfile, line)) {
    stringstream ss(line); string val; vector<double> c;
    while(getline(ss, val, ',')) c.push_back(stod(val));
    if (c.size() == dimension) queries.push_back(c);
  }

  // --- EXECUÇÃO E VALIDAÇÃO ---
  ofstream report(resultsFile);
  report << "Query_ID,Tipo,Tempo_ms,Paginas_Lidas,Recall_vs_LinearScan\n";

  // Executa queries k-NN
  for (int i = 0; i < queries.size(); ++i) {
    // 1. BUSCA LINEAR (GROUND TRUTH / FORÇA BRUTA)
    // Calcula a distância de TODOS os pontos para a query
    for (auto& entry : fullData) {
      entry.dist = getL2(entry.coords, queries[i].data(), dimension);
    }
    
    // Ordena por distância (menor para maior)
    sort(fullData.begin(), fullData.end(), [](const PointEntry& a, const PointEntry& b) {
      return a.dist < b.dist;
    });

    // Pega os IDs dos K vizinhos mais próximos reais
    vector<uint64_t> groundTruthIds;
    for (int j = 0; j < kNeighbors; ++j) groundTruthIds.push_back(fullData[j].id);

    // 2. BUSCA NA R*-TREE (APROXIMADA OU EXATA DEPENDE DA ESTRUTURA)
    ValidationVisitor visitor;
    IStatistics* statsPre; tree->getStatistics(&statsPre);
    uint64_t readsPre = statsPre->getReads();
    
    auto start = chrono::high_resolution_clock::now();
    Point queryPoint(queries[i].data(), dimension);
    
    // Executa k-NN na árvore
    tree->nearestNeighborQuery(kNeighbors, queryPoint, visitor);
    
    auto end = chrono::high_resolution_clock::now();
    
    IStatistics* statsPost; tree->getStatistics(&statsPost);
    
    // 3. CÁLCULO DE RECALL (PRECISÃO)
    // Verifica quantos IDs retornados pela árvore estão no Ground Truth
    int matches = 0;
    for (auto id : visitor.neighborIds) {
      if (find(groundTruthIds.begin(), groundTruthIds.end(), id) != groundTruthIds.end()) matches++;
    }
    
    // Recall = (Relevantes Encontrados) / (Total Relevantes)
    double recall = (double)matches / kNeighbors;

    // Salva relatório
    report << i << ",kNN," << chrono::duration<double, milli>(end-start).count() 
      << "," << (statsPost->getReads() - readsPre) << "," << recall << "\n";
    
    cout << "Query " << i << " Validada. Recall: " << recall << endl;
  }

  delete tree; delete storage;
  report.close();
  return 0;
}
