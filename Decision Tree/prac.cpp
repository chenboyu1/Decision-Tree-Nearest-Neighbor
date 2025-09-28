#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <map>
#include <sstream>
#include <cstring> 
using namespace std;

int predict_vs_actual[100][100];

struct Node {

    Node() {
        isLeaf = false;
    }
    vector<int> children;
    int standindex, treeIndex;
    double xvalue, label;
	bool isLeaf;
};

struct Rota {

    void getxvalue() {
        xvalue.resize(xname.size());
        for (int j = 0; j < xname.size(); j++) {
            map<double, int> value;
            for (int i = 0; i < data.size(); i++) {
                value[data[i][j]] = 1;
            }

            for (auto iter = value.begin(); iter != value.end(); iter++) {
                xvalue[j].push_back(iter->first);
            }
        }
    }
    vector<string> xname;
    vector<vector<double>> data;

    vector<vector<double>> xvalue;
};

struct Tree {

    Rota beginRota;
    vector<Node> tree;

    Tree(Rota rota) {
        beginRota = rota;
        beginRota.getxvalue();

        Node root;
        root.treeIndex = 0;
        tree.push_back(root);
        build(beginRota, 0);

    }
    
    int getchoose(Rota rota) {
        int maxAttrIndex = -1;
        double maxAttrValue = 0.0;

        for (int i = 0; i < beginRota.xname.size() - 1; i++) {
            if (maxAttrValue < getGainRatio(rota, i)) {
                maxAttrValue = getGainRatio(rota, i);
                maxAttrIndex = i;
            }
        }

        return maxAttrIndex;
    }

    void build(Rota rota, int nodeIndex) {
        if (isLeafNode(rota) == true) {
            tree[nodeIndex].isLeaf = true;
            tree[nodeIndex].label = rota.data.back().back();
            return;
        }

        int choosex = getchoose(rota);

        map<double, vector<int>> attrValueMap;
        for (int i = 0; i < rota.data.size(); i++) {
            attrValueMap[rota.data[i][choosex]].push_back(i);
        }

        tree[nodeIndex].standindex = choosex;

        pair<double, int> majority = getmainLabel(rota);
        if ((double)majority.second / rota.data.size() > 0.8) {
            tree[nodeIndex].isLeaf = true;
            tree[nodeIndex].label = majority.first;
            return;
        }

        for (int i = 0; i < beginRota.xvalue[choosex].size(); i++) {
            double attrValue = beginRota.xvalue[choosex][i];

            Rota nextRota;
            vector<int> candi = attrValueMap[attrValue];
            for (int i = 0; i < candi.size(); i++) {
                nextRota.data.push_back(rota.data[candi[i]]);
            }

            Node nextNode;
            nextNode.xvalue = attrValue;
            nextNode.treeIndex = (int)tree.size();
            tree[nodeIndex].children.push_back(nextNode.treeIndex);
            tree.push_back(nextNode);

            if (nextRota.data.size() == 0) {
                nextNode.isLeaf = true;
                nextNode.label = getmainLabel(rota).first;
                tree[nextNode.treeIndex] = nextNode;
            } else {
                build(nextRota, nextNode.treeIndex);
            }
        }
    }

    int dfs(vector<double>& v, int here) {
        if (tree[here].isLeaf) {
            return here;
        }

        int standindex = tree[here].standindex;

        for (int i = 0; i < tree[here].children.size(); i++) {
            int next = tree[here].children[i];

            if (v[standindex] == tree[next].xvalue) {
                return dfs(v, next);
            }
        }
        return -1;
    }

    pair<double, int> getmainLabel(Rota rota) {
        double mainLabel = 0;
        int mainCount = 0;

        map<double, int> labelCount;
        for (int i = 0; i < rota.data.size(); i++) {
            labelCount[rota.data[i].back()]++;

            if (labelCount[rota.data[i].back()] > mainCount) {
                mainCount = labelCount[rota.data[i].back()];
                mainLabel = rota.data[i].back();
            }
        }

        return {mainLabel, mainCount};
    }

    bool isLeafNode(Rota rota) {
        for (int i = 1; i < rota.data.size(); i++) {
            if (rota.data[0].back() != rota.data[i].back()) {
                return false;
            }
        }
        return true;
    }


    double getGainRatio(Rota rota, int xIndex) {
        return getGain(rota, xIndex) / getSplitInfoAttrD(rota, xIndex);
    }

    double getInfoD(Rota rota) {
        double ret = 0.0;

        int itemCount = (int)rota.data.size();
        map<double, int> labelCount;

        for (int i = 0; i < rota.data.size(); i++) {
            labelCount[rota.data[i].back()]++;
        }

        for (auto iter = labelCount.begin(); iter != labelCount.end(); iter++) {
            double p = (double)iter->second / itemCount;

            ret += -1.0 * p * log(p) / log(2);
        }

        return ret;
    }

    double getInfoAttrD(Rota rota, int xIndex) {
        double ret = 0.0;
        int itemCount = (int)rota.data.size();

        map<double, vector<int>> xValue;
        for (int i = 0; i < rota.data.size(); i++) {
            xValue[rota.data[i][xIndex]].push_back(i);
        }

        for (auto iter = xValue.begin(); iter != xValue.end(); iter++) {
            Rota nextrota;
            for (int i = 0; i < iter->second.size(); i++) {
                nextrota.data.push_back(rota.data[iter->second[i]]);
            }
            int nextItemCount = (int)nextrota.data.size();

            ret += (double)nextItemCount / itemCount * getInfoD(nextrota);
        }

        return ret;
    }

    double getGain(Rota rota, int xIndex) {
        return getInfoD(rota) - getInfoAttrD(rota, xIndex);
    }

    double getSplitInfoAttrD(Rota rota, int xIndex) {
        double ret = 0.0;

        int itemCount = (int)rota.data.size();

        map<double, vector<int>> xValue;
        for (int i = 0; i < rota.data.size(); i++) {
            xValue[rota.data[i][xIndex]].push_back(i);
        }

        for (auto iter = xValue.begin(); iter != xValue.end(); iter++) {
            Rota nextrota;
            for (int i = 0; i < iter->second.size(); i++) {
                nextrota.data.push_back(rota.data[iter->second[i]]);
            }
            int nextItemCount = (int)nextrota.data.size();

            double d = (double)nextItemCount / itemCount;
            ret += -1.0 * d * log(d) / log(2);
        }

        return ret;
    }
	
	double guess(vector<double> v) {
        double label = 0;
        int leafNode = dfs(v, 0);
        if (leafNode == -1) {
            return -1;
        }
        label = tree[leafNode].label;
        return label;
    }
};

struct Input{

    ifstream fin;
    Rota rota;

    Input(string filename) {
        fin.open(filename);
        seperate();
    }
    void seperate() {
        string str;
        for(int i = 1; i <= 17; i++){
        	rota.xname.push_back(to_string(i));
		}
        
        while (getline(fin, str)) {
            vector<double> row;
            stringstream ss(str);
            string token;

            while (getline(ss, token, ',')) {
                row.push_back(stod(token));
            }
            rota.data.push_back(row);

        }
    }
    Rota getTable() {
        return rota;
    }
};


int main(int argc, const char * argv[]) {
	
    string trainFileName = "train.csv";
    Input trainInputReader(trainFileName);
    Tree tree(trainInputReader.getTable());
	
	// 用train裡面的資料 
    Input traindata(trainFileName);
    Rota train = traindata.getTable();

    string result1FileName = "result_train.csv";
    ofstream fout;
	fout.open(result1FileName);
    for (int i = 0; i < train.data.size(); i++) {
        string s = to_string(tree.guess(train.data[i]));
        fout << s << endl;
    }

    Input answer1("train.csv");
    Rota table1 = answer1.getTable();
    int totalCount = (int)table1.data.size();
	
	int numclassification = 0;
	memset(predict_vs_actual, 0, sizeof(predict_vs_actual));
    for (int i = 0; i < train.data.size(); i++) {
        int row = table1.data[i].back();
    	int col = tree.guess(train.data[i]);
        predict_vs_actual[row][col]++;
        numclassification = max(numclassification, row);
    }
    double TP, FP, FN, P, R, F1, sumF1 = 0;
    for(int i = 0; i < numclassification; i++){
    	FP = 0, FN = 0;
    	TP = predict_vs_actual[i][i];
    	for(int j = 0; j < numclassification; j++){
    		FP += predict_vs_actual[j][i];
		}
		P = TP / FP;
    	for(int j = 0; j < numclassification; j++){
    		FN += predict_vs_actual[i][j];
		}
		R = TP / FN;
		F1 = (2*P*R) / (P+R);
		sumF1 += F1;
	}
	sumF1 /= numclassification;
	printf("Use train.csv MacroF1 = %f\n", sumF1);

    // 用test裡面的資料 
    string testFileName = "test.csv";
    Input testdata(testFileName);
    Rota test = testdata.getTable();

    string result2FileName = "result_test.csv";
    ofstream f2out;
	f2out.open(result2FileName);
    for (int i = 0; i < test.data.size(); i++) {
        string s = to_string(tree.guess(test.data[i]));
        f2out << s << endl;
    }

    Input answer2("test.csv");
    Rota table2 = answer2.getTable();
    totalCount = (int)table2.data.size();

    numclassification = 0;
    memset(predict_vs_actual, 0, sizeof(predict_vs_actual));
    for (int i = 0; i < test.data.size(); i++) {
    	int row = table2.data[i].back();
    	int col = tree.guess(test.data[i]);
        predict_vs_actual[row][col]++;
        numclassification = max(numclassification, row);
    }
    
    sumF1 = 0;
    for(int i = 0; i < numclassification; i++){
    	FP = 0, FN = 0;
    	TP = predict_vs_actual[i][i];
    	for(int j = 0; j < numclassification; j++){
    		FP += predict_vs_actual[j][i];
		}
		P = TP / FP;
    	for(int j = 0; j < numclassification; j++){
    		FN += predict_vs_actual[i][j];
		}
		R = TP / FN;
		F1 = (2*P*R) / (P+R);
		sumF1 += F1;
	}
	sumF1 /= numclassification;
	printf("Use test.csv MacroF1 = %f\n", sumF1);

    return 0;
}

