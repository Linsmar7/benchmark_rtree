#include <spatialindex/SpatialIndex.h>
#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <filesystem>
#include <numeric>
#include <cmath>
#include <unistd.h>
#include <iomanip>
#include <algorithm>
#include <random>

namespace fs = std::filesystem;
using namespace SpatialIndex;
using namespace std;

// --- Funções Utilitárias ---

// Obtém o uso de RAM atual em MB
double getRAMUsageMB() {
  long rss = 0L;
  ifstream stat_stream("/proc/self/statm", ios_base::in);
  if (stat_stream >> rss) return (rss * sysconf(_SC_PAGESIZE)) / (1024.0 * 1024.0);
  return 0.0;
}

// Calcula a Distância Euclidiana (L2) entre dois pontos
double calculateL2(const double* p1, const double* p2, uint32_t dim) {
  double sum = 0;
  for (uint32_t i = 0; i < dim; ++i) {
    double diff = p1[i] - p2[i];
    sum += diff * diff;
  }
  return std::sqrt(sum);
}

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

// --- Visitante para Consultas ---
// Esta classe é chamada para cada nó ou dado encontrado durante a busca na árvore.
class BenchmarkVisitor : public IVisitor {
  public:
    uint32_t resultCount = 0;
    const double* queryPoint; 
    double queryRadius; 
    uint32_t dimension; 
    bool isRangeQuery;
    double minDistanceFound = 999999.0; // Distância mínima encontrada (útil para debug ou k-NN)

    BenchmarkVisitor(const double* q, double r, uint32_t d, bool range) 
      : queryPoint(q), queryRadius(r), dimension(d), isRangeQuery(range) {}

    // VisitNode é chamado para nós internos da árvore (não folhas).
    void visitNode(const INode& n) override {}
    
    // VisitData é chamado quando um objeto de dados (folha) é encontrado.
    void visitData(const IData& d) override {
      IShape* shape;
      d.getShape(&shape);
      Region mbr;
      shape->getMBR(mbr);
      
      // Calcula a distância do ponto de consulta até o MBR do dado encontrado
      // Como nossos dados são pontos, o MBR low é igual ao próprio ponto.
      double currentDistance = calculateL2(queryPoint, mbr.m_pLow, dimension);
      
      // Atualiza a menor distância encontrada (apenas para estatísticas).
      if (currentDistance < minDistanceFound) minDistanceFound = currentDistance;

      if (isRangeQuery) {
        // Para Range Query, verificamos se a distância está dentro do raio.
        if (currentDistance <= queryRadius) resultCount++;
      } else {
        // Para k-NN, o R-Tree já filtra os vizinhos mais próximos.
        resultCount++;
      }
      delete shape;
    }

    // Método para visitar múltiplos dados de uma vez.
    void visitData(std::vector<const IData*>& v) override {}
};

int main(int argc, char** argv) {
  // --- VERIFICAÇÃO DE ARGUMENTOS ---
  if (argc < 3) {
    cerr << "Uso: " << argv[0] << " <caminho_dataset> <dimensao>" << endl;
    cerr << "Exemplo: " << argv[0] << " ../datasets/data.txt 128" << endl;
    return 1;
  }

  // --- PARÂMETROS VIA CLI ---
  string datasetPath = argv[1];
  unsigned int dimension = stoi(argv[2]);
  
  // Extrai o nome do dataset para usar nos arquivos de saída e índice
  string datasetName = fs::path(datasetPath).stem().string();

  // --- PARÂMETROS CONFIGURÁVEIS ---
  int kNeighbors = 5;                   // K para consulta k-NN
  double rangeRadius = 0.1;               // Raio para Range Query
  
  // Nomes de arquivos dinâmicos baseados no dataset
  string baseName = "rtree_index_" + datasetName;      
  
  // Cria diretório de resultados se não existir
  if (!fs::exists("results")) {
    fs::create_directory("results");
  }
  string resultsFile = "results/benchmark_" + datasetName + ".csv";
  // --------------------------------

  IStorageManager* storage = nullptr;
  ISpatialIndex* tree = nullptr;
  id_type indexIdentifier = 1;
  double buildTime = 0;

  // --- CARREGAMENTO / CONSTRUÇÃO DO ÍNDICE ---
  if (!fs::exists(baseName + ".idx")) {
    cout << "Índice não encontrado. Construindo nova R*-Tree..." << endl;
    auto startBuild = chrono::high_resolution_clock::now();
    
    // Cria gerenciador de armazenamento em disco
    storage = StorageManager::createNewDiskStorageManager(baseName, 4096);
    // Cria a R-Tree (RSTAR variant)
    tree = RTree::createNewRTree(*storage, 0.7, 100, 10, dimension, RTree::RV_RSTAR, indexIdentifier);
    
    ifstream infile(datasetPath);
    string line; id_type id = 0;
    
    // Lê o dataset e insere ponto a ponto
    while (getline(infile, line)) {
      stringstream ss(line); string val; vector<double> coords;
      while (getline(ss, val, ',')) coords.push_back(stod(val));
      
      if (coords.size() == dimension) {
        Point p(coords.data(), dimension);
        // Insere dados: (payload size, payload ptr, shape, object ID)
        tree->insertData(0, nullptr, p, id++);
      }
    }
    
    auto endBuild = chrono::high_resolution_clock::now();
    buildTime = chrono::duration<double>(endBuild - startBuild).count();
    cout << "Construção finalizada." << endl;
  } else {
    cout << "Carregando R*-Tree existente do disco..." << endl;
    storage = StorageManager::loadDiskStorageManager(baseName);
    tree = RTree::loadRTree(*storage, indexIdentifier);
  }

  // --- PREPARAÇÃO DAS CONSULTAS ---
  generateQueryFiles(datasetPath, datasetName, dimension);
  
  string knnPath = "queries/" + datasetName + "_knn.csv";
  string rangePath = "queries/" + datasetName + "_range.csv";

  // Carrega queries k-NN
  vector<vector<double>> knnQueries;
  ifstream knnFile(knnPath);
  string qline;
  while(getline(knnFile, qline)){
    stringstream ss(qline); string val; vector<double> coords;
    while(getline(ss, val, ',')) coords.push_back(stod(val));
    if (coords.size() == dimension) knnQueries.push_back(coords);
  }

  // Carrega queries Range
  vector<vector<double>> rangeQueries;
  ifstream rangeFile(rangePath);
  while(getline(rangeFile, qline)){
    stringstream ss(qline); string val; vector<double> coords;
    while(getline(ss, val, ',')) coords.push_back(stod(val));
    if (coords.size() == dimension) rangeQueries.push_back(coords);
  }

  // --- EXECUÇÃO DAS CONSULTAS ---
  ofstream log(resultsFile);
  log << "Query_ID,Tipo,K_ou_Raio,Tempo_ms,Paginas_Lidas,RAM_MB,Resultados_Encontrados\n";

  cout << "Executando " << knnQueries.size() << " k-NN queries..." << endl;

  int queryId = 0;

  // 1. Executa k-NN
  for (const auto& qCoords : knnQueries) {
    BenchmarkVisitor visitor(qCoords.data(), rangeRadius, dimension, false); // isRange = false
    
    IStatistics* statsPre; tree->getStatistics(&statsPre);
    uint64_t readsPre = statsPre->getReads();

    auto startQuery = chrono::high_resolution_clock::now();
    Point queryPoint(qCoords.data(), dimension);
    tree->nearestNeighborQuery(kNeighbors, queryPoint, visitor);
    auto endQuery = chrono::high_resolution_clock::now();
    
    IStatistics* statsPost; tree->getStatistics(&statsPost);

    log << queryId++ << ",kNN," << kNeighbors << ","
      << chrono::duration<double, milli>(endQuery - startQuery).count() << ","
      << (statsPost->getReads() - readsPre) << ","
      << getRAMUsageMB() << ","
      << visitor.resultCount << "\n";
  }

  cout << "Executando " << rangeQueries.size() << " Range queries..." << endl;

  // 2. Executa Range
  for (const auto& qCoords : rangeQueries) {
    BenchmarkVisitor visitor(qCoords.data(), rangeRadius, dimension, true); // isRange = true
    
    IStatistics* statsPre; tree->getStatistics(&statsPre);
    uint64_t readsPre = statsPre->getReads();

    auto startQuery = chrono::high_resolution_clock::now();
    
    double low[32], high[32]; // Assumindo max dim 32, mas ideal seria vetor dinâmico se dim for maior
    // Para segurança, vamos usar vector se dim > 32 ou alocação dinâmica. 
    // Mas o código original usava array fixo 32. Vamos manter vetor dinâmico para segurança.
    vector<double> lowV(dimension), highV(dimension);
    for(uint32_t d=0; d<dimension; d++) { 
      lowV[d] = qCoords[d] - rangeRadius; 
      highV[d] = qCoords[d] + rangeRadius; 
    }
    Region queryRegion(lowV.data(), highV.data(), dimension);
    tree->intersectsWithQuery(queryRegion, visitor);
    
    auto endQuery = chrono::high_resolution_clock::now();
    
    IStatistics* statsPost; tree->getStatistics(&statsPost);

    log << queryId++ << ",Range," << rangeRadius << ","
      << chrono::duration<double, milli>(endQuery - startQuery).count() << ","
      << (statsPost->getReads() - readsPre) << ","
      << getRAMUsageMB() << ","
      << visitor.resultCount << "\n";
  }

  // --- RELATÓRIO FINAL ---
  cout << "\n--- RESUMO DE CONSTRUCAO ---" << endl;
  if (buildTime > 0) cout << "Tempo de Construção: " << buildTime << " s" << endl;
  
  uintmax_t diskSize = fs::file_size(baseName + ".idx") + fs::file_size(baseName + ".dat");
  cout << "Tamanho da Árvore em Disco: " << diskSize / (1024.0 * 1024.0) << " MB" << endl;
  cout << "Resultados salvos em " << resultsFile << endl;

  delete tree; delete storage;
  log.close();
  return 0;
}
