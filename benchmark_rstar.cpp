#include <spatialindex/SpatialIndex.h>
#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <filesystem>
#include <set>

namespace fs = std::filesystem;

using namespace SpatialIndex;
using namespace std;

// // Classe Visitor completa para evitar o erro de "abstract type"
// class MyVisitor : public IVisitor {
// public:
//   void visitNode(const INode& n) override {}
  
//   // Sobrecarga 1: Para resultados individuais
//   void visitData(const IData& d) override {
//     // id_type id = d.getIdentifier();
//   }

//   // Sobrecarga 2: Necessária para a interface ser considerada completa
//   void visitData(std::vector<const IData*>& v) override {}
// };

// Visitor que salva os IDs encontrados na R*-Tree
class RecallVisitor : public IVisitor {
public:
    std::set<id_type> found_ids; // Armazena os IDs dos vizinhos

    void visitNode(const INode& n) override {}
    
    void visitData(const IData& d) override {
        found_ids.insert(d.getIdentifier()); // Salva o ID do objeto
    }

    void visitData(std::vector<const IData*>& v) override {
        for (auto const* data : v) {
            found_ids.insert(data->getIdentifier());
        }
    }
};

// Função para calcular o Recall comparando com o LIMS
double calculateRecall(const std::set<id_type>& groundTruth, const std::vector<id_type>& limsResults) {
    if (groundTruth.empty()) return 0.0;
    
    int matches = 0;
    for (auto id : limsResults) {
        if (groundTruth.find(id) != groundTruth.end()) {
            matches++;
        }
    }
    // Recall = (Itens corretos encontrados) / (Total de itens que deveriam ser encontrados)
    return static_cast<double>(matches) / groundTruth.size();
}

int main() {
  unsigned int dimension = 32;
  
  std::string baseName = "rtree_benchmark";
  IStorageManager* storage = StorageManager::createNewDiskStorageManager(baseName, 4096);
  
  id_type indexIdentifier;
  ISpatialIndex* tree = RTree::createNewRTree(*storage, 0.7, 100, 10, dimension, RTree::RV_RSTAR, indexIdentifier);

  ifstream infile("../datasets_processed/Imagenet32_train/color_32.txt");
  string line;
  id_type id = 0;

  cout << "Iniciando inserção na R*-Tree..." << endl;
  auto start_build = chrono::high_resolution_clock::now();

  while (getline(infile, line)) {
    stringstream ss(line);
    string val;
    vector<double> coords;
    
    // Parsing de CSV (vírgulas)
    while (getline(ss, val, ',')) {
      coords.push_back(stod(val));
    }

    if (coords.size() == dimension) {
      Point p(coords.data(), dimension);
      tree->insertData(0, nullptr, p, id++);
    }
  }

  auto end_build = chrono::high_resolution_clock::now();
  cout << "Tempo de Construção: " << chrono::duration<double>(end_build - start_build).count() << "s" << endl;

  double queryCoords[32] = {0.5};
  Point queryPoint(queryCoords, dimension);
  RecallVisitor rstar_visitor;

  IStatistics* statsBefore;
  tree->getStatistics(&statsBefore);
  uint64_t readsBefore = statsBefore->getReads();

  auto start_query = chrono::high_resolution_clock::now();
  tree->nearestNeighborQuery(5, queryPoint, rstar_visitor);
  auto end_query = chrono::high_resolution_clock::now();

  IStatistics* statsAfter;
  tree->getStatistics(&statsAfter);
  uint64_t readsOnlyQuery = statsAfter->getReads() - readsBefore;

  std::set<id_type> groundTruth = rstar_visitor.found_ids;
  
  cout << "Latência k-NN: " << chrono::duration<double, milli>(end_query - start_query).count() << "ms" << endl;
  cout << "Acessos exclusivos da consulta k-NN: " << readsOnlyQuery << endl;

  std::vector<id_type> lims_returned_ids = {10, 25, 30, 42, 55};
  // double recall = calculateRecall(groundTruth, lims_returned_ids);

  cout << "--- Resultados de Acurácia ---" << endl;
  cout << "IDs esperados (R*-Tree): ";
  for(auto id : groundTruth) cout << id << " ";
  // cout << "\nRecall do LIMS: " << (recall * 100.0) << "%" << endl;

  delete tree;
  delete storage;

  uintmax_t size = fs::file_size("./rtree_benchmark.idx") + fs::file_size("./rtree_benchmark.dat");
  cout << "Ocupação total em disco: " << size / (1024.0 * 1024.0) << " MB" << endl;
  return 0;
}
