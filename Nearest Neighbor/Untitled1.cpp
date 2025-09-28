#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <thread>
#include <future>

using namespace std;

struct DataPoint {
    vector<double> features;
    int label;
};

vector<DataPoint> readCSV(const string& filename) {
    vector<DataPoint> data;
    ifstream file(filename);
    string line, word;

    if (!file.is_open()) {
        cerr << "Failed to open file: " << filename << endl;
        return data;
    }

    while (getline(file, line)) {
        stringstream s(line);
        DataPoint point;
        vector<double> features;

        while (getline(s, word, ',')) {
            try {
                features.push_back(stod(word));
            } catch (const invalid_argument& e) {
                cerr << "Invalid data: " << word << endl;
                continue;
            }
        }

        if (features.size() < 2) {
            cerr << "Invalid line (not enough data): " << line << endl;
            continue;
        }

        point.label = static_cast<int>(features.back());
        features.pop_back();
        point.features = features;

        data.push_back(point);
    }

    cout << "Read " << data.size() << " data points from " << filename << endl;
    return data;
}

double calculateDistance(const DataPoint& a, const DataPoint& b) {
    double distance = 0.0;
    for (size_t i = 0; i < a.features.size(); ++i) {
        distance += pow(a.features[i] - b.features[i], 2);
    }
    return sqrt(distance);
}

int customPartition(vector<pair<double, int>>& distances, int left, int right, int pivotIndex) {
    double pivotValue = distances[pivotIndex].first;
    swap(distances[pivotIndex], distances[right]);
    int storeIndex = left;
    for (int i = left; i < right; i++) {
        if (distances[i].first < pivotValue) {
            swap(distances[i], distances[storeIndex]);
            storeIndex++;
        }
    }
    swap(distances[storeIndex], distances[right]);
    return storeIndex;
}

void quickSelect(vector<pair<double, int>>& distances, int left, int right, int k) {
    while (left < right) {
        int pivotIndex = (left + right) / 2;
        pivotIndex = customPartition(distances, left, right, pivotIndex);
        if (k == pivotIndex) {
            return;
        } else if (k < pivotIndex) {
            right = pivotIndex - 1;
        } else {
            left = pivotIndex + 1;
        }
    }
}

int knnClassify(const vector<DataPoint>& trainData, const DataPoint& testPoint, int k) {
    vector<pair<double, int>> distances;
    for (const auto& trainPoint : trainData) {
        double dist = calculateDistance(trainPoint, testPoint);
        distances.push_back(make_pair(dist, trainPoint.label));
    }

    quickSelect(distances, 0, distances.size() - 1, k);

    unordered_map<int, int> labelCount;
    for (int i = 0; i < k; ++i) {
        int label = distances[i].second;
        labelCount[label]++;
    }

    int bestLabel = -1, maxCount = 0;
    for (const auto& lc : labelCount) {
        if (lc.second > maxCount) {
            bestLabel = lc.first;
            maxCount = lc.second;
        }
    }
    return bestLabel;
}

void processPart(const vector<DataPoint>& trainData, const vector<DataPoint>& partData, vector<int>& results, int k, int offset) {
    for (size_t i = 0; i < partData.size(); ++i) {
        results[offset + i] = knnClassify(trainData, partData[i], k);
    }
}

double calculatePrecision(const vector<int>& predictedLabels, const vector<int>& trueLabels, int classLabel) {
    int truePositive = 0, falsePositive = 0;
    for (size_t i = 0; i < predictedLabels.size(); ++i) {
        if (predictedLabels[i] == classLabel && predictedLabels[i] == trueLabels[i]) {
            truePositive++;
        }
        if (predictedLabels[i] == classLabel && predictedLabels[i] != trueLabels[i]) {
            falsePositive++;
        }
    }
    return static_cast<double>(truePositive) / (truePositive + falsePositive);
}

double calculateRecall(const vector<int>& predictedLabels, const vector<int>& trueLabels, int classLabel) {
    int truePositive = 0, falseNegative = 0;
    for (size_t i = 0; i < predictedLabels.size(); ++i) {
        if (predictedLabels[i] == classLabel && predictedLabels[i] == trueLabels[i]) {
            truePositive++;
        }
        if (trueLabels[i] == classLabel && predictedLabels[i] != trueLabels[i
]) {
            falseNegative++;
        }
    }
    return static_cast<double>(truePositive) / (truePositive + falseNegative);
}

double calculateF1Score(double precision, double recall) {
    if (precision + recall == 0) {
        return 0.0;
    }
    return 2.0 * (precision * recall) / (precision + recall);
}

void processKNN(const string& trainFile, const string& testFile, const string& resultTrainFile, const string& resultTestFile, int k) {
    cout << "processing..." << endl;
    vector<DataPoint> trainData = readCSV(trainFile);
    vector<DataPoint> testData = readCSV(testFile);

    if (trainData.empty() || testData.empty()) {
        cerr << "No data loaded, exiting." << endl;
        return;
    }

    int numThreads = thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 2;

    vector<future<void>> futures;
    vector<int> trainResults(trainData.size());
    vector<int> testResults(testData.size());

    for (int t = 0; t < numThreads; ++t) {
        int partSize = trainData.size() / numThreads;
        int start = t * partSize;
        int end = (t == numThreads - 1) ? trainData.size() : start + partSize;
        futures.push_back(async(launch::async, processPart, ref(trainData), vector<DataPoint>(trainData.begin() + start, trainData.begin() + end), ref(trainResults), k, start));
    }

    for (auto& f : futures) {
        f.get();
    }

    futures.clear();
    for (int t = 0; t < numThreads; ++t) {
        int partSize = testData.size() / numThreads;
        int start = t * partSize;
        int end = (t == numThreads - 1) ? testData.size() : start + partSize;
        futures.push_back(async(launch::async, processPart, ref(trainData), vector<DataPoint>(testData.begin() + start, testData.begin
() + end), ref(testResults), k, start));
    }

    for (auto& f : futures) {
        f.get();
    }

    ofstream resultTrain(resultTrainFile);
    if (!resultTrain.is_open()) {
        cerr << "Failed to open result file: " << resultTrainFile << endl;
        return;
    }

    vector<int> trueTrainLabels;
    for (const auto& point : trainData) {
        trueTrainLabels.push_back(point.label);
    }

    double precision_train = calculatePrecision(trainResults, trueTrainLabels, 1);
    double recall_train = calculateRecall(trainResults, trueTrainLabels, 1);
    double f1_train = calculateF1Score(precision_train, recall_train);

    cout << "Train F1 Score: " << f1_train << endl;

    for (const auto& res : trainResults) {
        resultTrain << res << endl;
    }
    resultTrain.close();

    ofstream resultTest(resultTestFile);
    if (!resultTest.is_open()) {
        cerr << "Failed to open result file: " << resultTestFile << endl;
        return;
    }

    vector<int> trueTestLabels;
    for (const auto& point : testData) {
        trueTestLabels.push_back(point.label);
    }

    double precision_test = calculatePrecision(testResults, trueTestLabels, 1);
    double recall_test = calculateRecall(testResults, trueTestLabels, 1);
    double f1_test = calculateF1Score(precision_test, recall_test);

    cout << "Test F1 Score: " << f1_test << endl;

    for (const auto& res : testResults) {
        resultTest << res << endl;
    }
    resultTest.close();

    cout << "KNN processing complete." << endl;
}

int main() {
    int k = 3;
    string trainFile = "train.csv";
    string testFile = "test.csv";
    processKNN(trainFile, testFile, "result_train.csv", "result_test.csv", k);
    return 0;
}
