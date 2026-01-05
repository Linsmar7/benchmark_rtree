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
#include <iomanip>
#include <unistd.h>

using namespace SpatialIndex;
using namespace std;
namespace fs = std::filesystem;

// --- Helper Functions ---

double getRAMUsageMB() {
  long rss = 0L;
  ifstream stat_stream("/proc/self/statm", ios_base::in);
  if (stat_stream >> rss) return (rss * sysconf(_SC_PAGESIZE)) / (1024.0 * 1024.0);
  return 0.0;
}

double getL2(const vector<double>& p1, const double* p2, int dim) {
  double sum = 0;
  for (int i = 0; i < dim; ++i) {
    double diff = p1[i] - p2[i];
    sum += diff * diff;
  }
  return sqrt(sum);
}

// --- Structures ---

struct PointEntry {
  uint64_t id;
  vector<double> coords;
  double dist;
};

// --- Validation Visitor ---
// Captures IDs of results.
// For Range Queries, explicitly checks L2 distance to filter false positives from MBR search.
class ValidationVisitor : public IVisitor {
public:
  vector<uint64_t> neighborIds;
  const double* queryPoint;
  double queryRadius;
  uint32_t dimension;
  bool isRangeQuery;

  ValidationVisitor() : queryPoint(nullptr), queryRadius(0), dimension(0), isRangeQuery(false) {}
  
  void setQuery(const double* q, double r, uint32_t d, bool isRange) {
      queryPoint = q;
      queryRadius = r;
      dimension = d;
      isRangeQuery = isRange;
      neighborIds.clear();
  }

  void visitNode(const INode& n) override {}
  
  void visitData(const IData& d) override { 
      if (isRangeQuery) {
          IShape* shape;
          d.getShape(&shape);
          Region mbr;
          shape->getMBR(mbr);
          
          // Assuming point data, low == high == point coords
          double dist = getL2(vector<double>(mbr.m_pLow, mbr.m_pLow + dimension), queryPoint, dimension);
          if (dist <= queryRadius) {
               neighborIds.push_back(d.getIdentifier());
          }
          delete shape;
      } else {
          // k-NN handles its own filtering
          neighborIds.push_back(d.getIdentifier()); 
      }
  }
  
  void visitData(vector<const IData*>& v) override {}
};

int main(int argc, char** argv) {
  if (argc < 3) {
    cerr << "Uso: " << argv[0] << " <caminho_dataset> <dimensao>" << endl;
    return 1;
  }

  string dataset_arg = argv[1];
  fs::path argPath(dataset_arg);
  string datasetName = argPath.stem().string();
  unsigned int dimension = stoi(argv[2]);

  string baseName = "rtree_index_" + datasetName;

  // --- 1. Load Ground Truth ---
  cout << "Loading Ground Truth Dataset..." << endl;
  string datasetPath = dataset_arg;
   if (!fs::exists(datasetPath)) {
        vector<string> searchPaths = {
            "./datasets/" + dataset_arg + ".txt",
            "../datasets/" + dataset_arg + ".txt",
            "data/" + dataset_arg + ".txt",
            "datasets_processed/Imagenet32_train/" + dataset_arg + ".txt",
            dataset_arg
        };
        for (const auto& path : searchPaths) {
            if (fs::exists(path)) {
                datasetPath = path;
                break;
            }
        }
   }
   
  if (!fs::exists(datasetPath)) {
      cerr << "Dataset not found: " << datasetPath << endl;
      return 1;
  }

  vector<PointEntry> fullData;
  ifstream infile(datasetPath);
  string line; uint64_t currentId = 0;
  while (getline(infile, line)) {
    stringstream ss(line); string val; vector<double> c;
    while (getline(ss, val, ',')) c.push_back(stod(val));
    if (c.size() == dimension) fullData.push_back({currentId++, c, 0.0});
  }
  cout << "Loaded " << fullData.size() << " points." << endl;

  // --- 2. Load R-Tree ---
  // Ensure we look in r_tree folder if baseName doesn't have it, but we added it above.
  cout << "Loading R-Tree: " << baseName << " (Index ID: 1)" << endl;
  try {
      IStorageManager* storage = StorageManager::loadDiskStorageManager(baseName);
      ISpatialIndex* tree = nullptr;
      vector<id_type> idsToTry = {1, 2, 0};
      bool loaded = false;
      for (id_type id : idsToTry) {
        try {
            cout << "Trying to load Index ID " << id << "..." << endl;
            tree = RTree::loadRTree(*storage, id);
            
            // Quick Dimension Check via Probe Query
            // We cannot check dimension directly, so we run a dummy query.
            // If dimension is mismatch, it throws IllegalArgumentException.
            try {
                ValidationVisitor dummyVisitor;
                dummyVisitor.setQuery(new double[dimension]{}, 0, dimension, false); // dummy zero point
                Point dummyPoint(dummyVisitor.queryPoint, dimension);
                tree->nearestNeighborQuery(1, dummyPoint, dummyVisitor);
                delete[] dummyVisitor.queryPoint; 
            } catch (Tools::IllegalArgumentException&) {
                cout << "Index ID " << id << " has faulty dimension (probe failed). Skipping." << endl;
                delete tree;
                tree = nullptr;
                continue;
            }
            
            cout << "Successfully loaded Index ID " << id << " with correct dimension." << endl;
            loaded = true;
            break;
        } catch (...) {
            cout << "Failed to load Index ID " << id << "." << endl;
        }
      }
      
      if (!loaded || !tree) {
         cerr << "Could not load any valid R-Tree index with dimension " << dimension << "." << endl;
         delete storage;
         return 1;
      }

      // --- 3. Run Validation ---
      string resultsFile = "./results/validacao_rtree_" + datasetName + ".csv";
      if (!fs::exists("./results")) fs::create_directory("./results");
      
      ofstream report(resultsFile);
      report << "Query_ID,Tipo,K_ou_Raio,Tempo_ms,Paginas_Lidas,Recall,Resultados_Encontrados\n";

      // Load Queries
      string knnPath = "./queries/" + datasetName + "_knn.csv";
      ifstream qfile(knnPath);
      vector<vector<double>> knnQueries;
      while(getline(qfile, line)) {
        stringstream ss(line); string val; vector<double> c;
        while(getline(ss, val, ',')) c.push_back(stod(val));
        if (c.size() == dimension) knnQueries.push_back(c);
      }
      
      string rangePath = "./queries/" + datasetName + "_range.csv";
      ifstream rfile(rangePath);
      vector<vector<double>> rangeQueries;
      while(getline(rfile, line)) {
        stringstream ss(line); string val; vector<double> c;
        while(getline(ss, val, ',')) c.push_back(stod(val));
        if (c.size() == dimension) rangeQueries.push_back(c);
      }

      int K = 5;
      cout << "\nRunning " << knnQueries.size() << " k-NN queries (k=" << K << ")..." << endl;
      
      int qId = 0;
      for (auto& q : knnQueries) {
          // GT
          for (auto& entry : fullData) {
              entry.dist = getL2(q, entry.coords.data(), dimension);
          }
          partial_sort(fullData.begin(), fullData.begin() + K, fullData.end(), [](const PointEntry& a, const PointEntry& b) {
              return a.dist < b.dist;
          });
          vector<uint64_t> gtIds;
          for(int k=0; k<K; ++k) gtIds.push_back(fullData[k].id);

          // R-Tree
          ValidationVisitor visitor;
          visitor.setQuery(q.data(), 0, dimension, false);
          
          IStatistics* statsPre; tree->getStatistics(&statsPre);
          uint64_t readsPre = statsPre->getReads();
          
          auto start = chrono::high_resolution_clock::now();
          Point queryPoint(q.data(), dimension);
          
          try {
            tree->nearestNeighborQuery(K, queryPoint, visitor);
          } catch (Tools::IllegalArgumentException& e) {
             cerr << "Error running k-NN: " << e.what() << endl;
             cerr << "Hint: The R-Tree might have been built with a different dimension than " << dimension << "." << endl;
             return 1;
          }
          
          auto end = chrono::high_resolution_clock::now();
          
          IStatistics* statsPost; tree->getStatistics(&statsPost);

          // Recall
          int matches = 0;
          for (auto id : visitor.neighborIds) {
              if (find(gtIds.begin(), gtIds.end(), id) != gtIds.end()) matches++;
          }
          double recall = (double)matches / K;
          
          double time_ms = chrono::duration<double, milli>(end - start).count();
          report << qId++ << ",kNN," << K << "," << time_ms << "," << (statsPost->getReads() - readsPre) << "," << recall << "," << visitor.neighborIds.size() << "\n";
          cout << "kNN " << qId-1 << ": Recall=" << recall << " Time=" << time_ms << "ms" << endl;
      }

      // Range
      double radius = 0.1;
      cout << "\nRunning " << rangeQueries.size() << " Range queries (r=" << radius << ")..." << endl;
      
      qId = 0;
      for (auto& q : rangeQueries) {
          // GT
          vector<uint64_t> gtIds;
          for (auto& entry : fullData) {
              // Recalc dist
              double dist = getL2(q, entry.coords.data(), dimension);
              if (dist <= radius) gtIds.push_back(entry.id);
          }
          sort(gtIds.begin(), gtIds.end());

          // R-Tree
          ValidationVisitor visitor;
          visitor.setQuery(q.data(), radius, dimension, true);

          IStatistics* statsPre; tree->getStatistics(&statsPre);
          uint64_t readsPre = statsPre->getReads();

          auto start = chrono::high_resolution_clock::now();
          vector<double> lowV(dimension), highV(dimension);
          for(uint32_t d=0; d<dimension; d++) { 
             lowV[d] = q[d] - radius; 
             highV[d] = q[d] + radius; 
          }
          Region queryRegion(lowV.data(), highV.data(), dimension);
          
          try {
            tree->intersectsWithQuery(queryRegion, visitor);
          } catch (Tools::IllegalArgumentException& e) {
             cerr << "Error running Range Query: " << e.what() << endl;
             return 1;
          }
           
          auto end = chrono::high_resolution_clock::now();

          IStatistics* statsPost; tree->getStatistics(&statsPost);

          // Recall
          sort(visitor.neighborIds.begin(), visitor.neighborIds.end());
          vector<uint64_t> intersection;
          set_intersection(visitor.neighborIds.begin(), visitor.neighborIds.end(),
                           gtIds.begin(), gtIds.end(),
                           back_inserter(intersection));
          
          double recall = 0.0;
          if (!gtIds.empty()) recall = (double)intersection.size() / gtIds.size();
          else recall = 1.0;

          double time_ms = chrono::duration<double, milli>(end - start).count();
          report << qId++ << ",Range," << radius << "," << time_ms << "," << (statsPost->getReads() - readsPre) << "," << recall << "," << visitor.neighborIds.size() << "\n";
          cout << "Range " << qId-1 << ": Recall=" << recall << " Time=" << time_ms << "ms (Found " << visitor.neighborIds.size() << "/" << gtIds.size() << ")" << endl;
      }

      delete tree; 
      delete storage;
      report.close();
      cout << "Saved validation to " << resultsFile << endl;

  } catch (Tools::Exception& e) {
      cerr << "SpatialIndex Error: " << e.what() << endl;
      return 1;
  } catch (std::exception& e) {
      cerr << "Std Error: " << e.what() << endl;
      return 1;
  }

  return 0;
}
