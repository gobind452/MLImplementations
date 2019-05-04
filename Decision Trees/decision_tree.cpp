#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
#include<vector>
#include<iterator>
#include<cmath>
#include<algorithm>
using namespace std;


// Variables are referred -1. 2 corresponds to X3
// Pass median arrays in decideVariable
// PopulateMatrices doesnt have a flag in them

int train_examples = 18000;
int test_examples = 6000;
int val_examples = 6000;
int dimensions = 23;

int default_variable = -5;
float default_median = -3;

int** train_data = new int*[train_examples];
int** test_data = new int*[test_examples];
int** validate_data = new int*[val_examples];

int variable_info[] = {2,2,7,4,2,12,12,12,12,12,12,2,2,2,2,2,2,2,2,2,2,2,2}; // Variable ranges
int variable_offset[] = {0,1,0,0,0,-2,-2,-2,-2,-2,-2,0,0,0,0,0,0,0,0,0,0,0,0}; // Variable starts
bool remaining[] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
bool continuous[] = {1,0,0,0,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1};

void storeData(string line,int** data,int count){
	data[count] = new int[dimensions+1];
	int* location = data[count];
	stringstream sep(line);
	string value;
	while(getline(sep,value,' ')){
		*(location++) = stoi(value);
	}
}

typedef struct Node{
	Node(){
		this->variable = default_variable;
		this->median = default_median;
		this->pruned = 0;
		this->validationCorrect = 0;
	}
	int variable; 
	vector<Node*> branches;
	float median; // For median splits
	bool maxPrediction; // Dominating Prediction (Useful for pruning)
	bool pruned;
	vector<int> validationRange;
	int validationCorrect;
}Node;

float calculateEntropy(int a,int b){
	if(a == 0 or b == 0){
		return 0;
	}
	float probab = (float(a)/(a+b));
	return -1*(probab*log2(probab)+(1-probab)*log2(1-probab));
}

class infoMatrix{
public:
	int** matrix;
	int variable;
	infoMatrix(int variable){
		this->variable = variable;
		matrix = new int*[variable_info[variable]]; 
		for(int count=0;count<variable_info[variable];count++){
			matrix[count] = new int[2];
			matrix[count][0] = 0;
			matrix[count][1] = 0;
		}
	}
	float information(int total){
		float entropy = 0;
		for(int count=0;count<variable_info[this->variable];count++){
			entropy = entropy + (matrix[count][0]+matrix[count][1])*calculateEntropy(matrix[count][0],matrix[count][1]);
		}
		return entropy/total;
	}
	void cleanup(){
		for(int i=0;i<variable_info[this->variable];i++){
			delete [] matrix[i];
		}
		delete [] matrix;
	}
};

ostream& operator <<(ostream& os,infoMatrix& info){
	for(int i=0;i<variable_info[info.variable];i++){
		os << info.matrix[i][0] << " " << info.matrix[i][1] << "\n";
	}
	return os;
}

bool populateMatrices(vector<infoMatrix>& matrices,vector<int>& range,float* medians){
	int count = 0;
	for(int i=0;i<range.size();i++){
		int* data = train_data[range[i]]; // Select the corresponding data
		for(int j=0;j<dimensions;j++){
			if(remaining[j] == 0){ // Dont do anything for those variables
				continue;
			}
			if(continuous[j] == 1){
				matrices[j].matrix[int(data[j]>=medians[j])][data[dimensions]]++; // Increment the counter by deciding on median
			}
			else{
				matrices[j].matrix[data[j]-variable_offset[j]][data[dimensions]]++; // 
			}
		}
		if(data[dimensions] == 1){
			count++;
		}
	}
	if(count==range.size() or count==0){ // Pure Node
		return 0;
	}
	return 1;
}

int decideVariable(vector<int>& range,vector<bool>& predictions,float* medians){ 
	if(range.size()==0){ // Nothing to decide
		return default_variable;
	}
	vector<infoMatrix> matrices;
	for(int i=0;i<dimensions;i++){
		matrices.push_back(infoMatrix(i)); // Create information matrices for all variables
	}
	bool check = populateMatrices(matrices,range,medians); // Would pass in median arrays here (Remember)
	if(check == 0){
		for(int i=0;i<dimensions;i++){
			matrices[i].cleanup();
		}
		return default_variable;
	}
	float min_entropy = 1.2;
	int splitVariable = default_variable;
	for(int count=0;count<dimensions;count++){
		if(remaining[count] == 0){ // If the variable has already been split
			continue;
		}
		float curr_entropy = matrices[count].information(range.size());
		if(curr_entropy<min_entropy){
			min_entropy = curr_entropy;
			splitVariable = count;
		}
	}
	if(splitVariable == default_variable){
		for(int i=0;i<dimensions;i++){
			matrices[i].cleanup();
		}
		return splitVariable;// All variables split on
	}
	for(int count=0;count<variable_info[splitVariable];count++){ // Domination predictions for each split (Useful in pruning)
		predictions.push_back(0);
		predictions[count] = (matrices[splitVariable].matrix[count][0]>matrices[splitVariable].matrix[count][1])?0:1;
	}
	for(int i=0;i<dimensions;i++){
		matrices[i].cleanup();
	}
	return splitVariable;
}

Node* decisionTree = new Node();

void split(vector<int>& range,vector<int>* splitRanges,int variable,int median,bool continuous,bool flag=0){ // Flag for multiple splits
	if(flag == 0 or continuous == 0){
		for(int i=0;i<range.size();i++){
			splitRanges[train_data[range[i]][variable]-variable_offset[variable]].push_back(range[i]);
		}
	}
	else{
		for(int i=0;i<range.size();i++){
			splitRanges[int(train_data[range[i]][variable]>=median)].push_back(range[i]);
		}
	}
}

void calculateMedian(vector<int>& range,float* median){
	vector<int>* values = new vector<int>[dimensions];
	for(int i=0;i<range.size();i++){
		for(int j=0;j<dimensions;j++){
			values[j].push_back(train_data[range[i]][j]);
		}
	}
	for(int j=0;j<dimensions;j++){
		sort(values[j].begin(), values[j].end());
	}
	if(range.size()%2==0){
		for(int j=0;j<dimensions;j++){
			median[j] = float(values[j][int(values[j].size()/2)-1]+values[j][int((values[j].size()+2)/2)-1])/2;
			values[j].clear();
		}
	}
	else{
		for(int j=0;j<dimensions;j++){
			median[j] = float(values[j][int(values[j].size()/2)-1]);
			values[j].clear();
		}
	}
	delete [] values;
	return;	
}

int depth = 0;
void growTree(Node* root,vector<int>& range,int height=0){ // Node to split, range of data coming here
	if(range.size() == 0){
		if(height>depth){
			depth = height;
		}		
		return;
	}
	float medians[] = {0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5};
	vector<bool> predictions; // Max predictions for each branch (Useful for pruning)
	root->variable = decideVariable(range,predictions,medians); // Decide variable
	if (root->variable == default_variable){ // No further splits
		if(height>depth){
			depth = height;
		}
		return;
	}
	vector<int>* splitRanges = new vector<int>[variable_info[root->variable]];
	split(range,splitRanges,root->variable,medians[root->variable],continuous[root->variable],0); // Partitions the data, stores maximum predictions for the given variables
	remaining[root->variable] = 0; 
	root->median = (continuous[root->variable] == 1)?medians[root->variable]:default_median; // If continuous variable 
	for(int count=0;count<variable_info[root->variable];count++){ // Branch off
		Node* branch = new Node();
		branch->maxPrediction = (splitRanges[count].size() == 0)?root->maxPrediction:predictions[count];
		root->branches.push_back(branch);
		growTree(branch,splitRanges[count],height+1);
	}
	remaining[root->variable] = 1; // Backtrack
	for(int i=0;i<variable_info[root->variable];i++){
		splitRanges[i].clear();
	}
	delete [] splitRanges;
	predictions.clear();
}

void growTreeMedian(Node* root,vector<int>& range,int height=0){
	if(range.size() == 0){
		if(height>depth){
			depth = height;
		}		
		return;
	}
	float medians[] = {0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5};
	calculateMedian(range,medians);
	vector<bool> predictions; // Max predictions for each branch (Useful for pruning)
	root->variable = decideVariable(range,predictions,medians); // Decide variable
	if (root->variable == default_variable){ // No further splits
		if(height>depth){
			depth = height;
		}
		return;
	}
	vector<int>* splitRanges = new vector<int>[variable_info[root->variable]];
	split(range,splitRanges,root->variable,medians[root->variable],continuous[root->variable],1); // Partitions the data, stores maximum predictions for the given variables
	for(int j=0;j<variable_info[root->variable];j++){
		if(splitRanges[j].size() == range.size()){
			remaining[root->variable] = 0;
			break;
		}
	}
	remaining[root->variable] = (continuous[root->variable] == 0)?0:remaining[root->variable];
	root->median = (continuous[root->variable] == 1)?medians[root->variable]:default_median; // If continuous variable 
	for(int count=0;count<variable_info[root->variable];count++){ // Branch off
		Node* branch = new Node();
		branch->maxPrediction = (splitRanges[count].size() == 0)?root->maxPrediction:predictions[count];
		root->branches.push_back(branch);
		growTreeMedian(branch,splitRanges[count],height+1);
	}
	remaining[root->variable] = 1; // Backtrack
	for(int i=0;i<variable_info[root->variable];i++){
		splitRanges[i].clear();
	}
	delete [] splitRanges;
	predictions.clear();
}

int makePrediction(int* example,Node* node,bool validate=0,int count=0){
	int prediction;
	if(node->variable == default_variable or node->pruned == 1){
		prediction = node->maxPrediction;
	}
	else if(continuous[node->variable] == 0){
		prediction = makePrediction(example,node->branches[example[node->variable]-variable_offset[node->variable]],validate,count);
	}
	else{
		if(example[node->variable]<node->median){
			prediction = makePrediction(example,node->branches[0],validate,count);
		}
		else{
			prediction = makePrediction(example,node->branches[1],validate,count);
		}
	}
	if(validate == 1){
		node->validationRange.push_back(count);
		if(prediction == example[dimensions]){
			node->validationCorrect++;
		}
	}
	return prediction;
}

float testData(int** data,int examples,bool validate);

int traverseAndPrune(Node* node,int height,int curr_height){ // Height at which node pruning is to be considered
	if(node->pruned == 1 or node->variable == default_variable){
		return 0;
	}
	if(height == curr_height){
		int correct = 0;
		for(int i=0;i<node->validationRange.size();i++){
			if(node->maxPrediction == validate_data[node->validationRange[i]][dimensions]){
				correct++;
			}
		}
		if(correct > node->validationCorrect){
			node->pruned = 1;
			return correct-node->validationCorrect;
		}
		return 0;
	}
	else{
		int correct = 0;
		for(int branch=0;branch<node->branches.size();branch++){
			correct = correct + traverseAndPrune(node->branches[branch],height,curr_height+1);
		}
		node->validationCorrect = node->validationCorrect + correct;
		return correct;
	}
}
int countNodes(Node* node);

void pruneTree(Node* root){
	cout << testData(validate_data,val_examples,1);
	int leaf = depth;
	for(int i=leaf;i>=0;i--){
		traverseAndPrune(root,i,0);
		cout << "Number of nodes " << countNodes(root) << " ";
		cout << "Validation Accuracy " << float(root->validationCorrect)/val_examples <<  " " << testData(test_data,test_examples,0) << " " << testData(train_data,train_examples,0) << endl;
	}
}

int countNodes(Node* node){
	if(node->variable == default_variable or node->pruned == 1){
		return 0;
	}
	else{
		int count = 0;
		for(int branch=0;branch<node->branches.size();branch++){
			if(node->branches[branch]->variable!= default_variable){
				count = count + countNodes(node->branches[branch]);
			}
		}
		return count+1;
	}
}

int splitted[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

int getSplitInformation(Node* node,int variable){
	if(node->pruned == 1 or node->variable == default_variable){
		return 0;
	}
	else{
		int max = 0;
		for(int branch=0;branch<node->branches.size();branch++){
			int answer = getSplitInformation(node->branches[branch],variable);
			if(answer>max){
				max = answer;
			}
		}
		if(node->variable == variable){
			return max+1;
		}
		return max;
	}
}

void getInformationVariables(){
	for(int i=0;i<dimensions;i++){
		splitted[i] = splitted[i]+getSplitInformation(decisionTree,i);
	}
}
float testData(int** data,int examples,bool validate=0){
	int predicted = 0;
	int correct = 0;
	for(int count=0;count<examples;count++){
		int* example = data[count];
		predicted = makePrediction(example,decisionTree,validate,count);
		if(predicted == data[count][dimensions]){
			correct++;
		}
	}
	float accuracy = float(correct)/examples;
	return accuracy;
}

int main(int argc,char* argv[]){
	vector<int> range;
	for(int i=0;i<train_examples;i++){
		range.push_back(i);
	}
	ifstream train_file;
	train_file.open(argv[2]);
	string line;
	getline(train_file,line);
	int count = 0;
	while(getline(train_file,line)){
		storeData(line,train_data,count);
		count++;
	}
	train_file.close();
	ifstream validate_file;
	validate_file.open(argv[4]);
	getline(validate_file,line);
	count = 0;
	while(getline(validate_file,line)){
		storeData(line,validate_data ,count);
		count++;
	}
	validate_file.close();
	validate_file.open(argv[3]);
	getline(validate_file,line);
	count = 0;
	while(getline(validate_file,line)){
		storeData(line,test_data ,count);
		count++;
	}
	validate_file.close();
	int part = atoi(argv[1]);
	switch(part){
		case 1:
		growTree(decisionTree,range);
		cout << "Training Accuracy " << testData(train_data,train_examples) << endl;
		cout << "Validation Accuracy " << testData(validate_data,val_examples) << endl;
		cout << "Test Accuracy " << testData(test_data,test_examples) << endl;
		break;
		case 2 :
		growTree(decisionTree,range);
		pruneTree(decisionTree);
		break;
		case 3:
		growTreeMedian(decisionTree,range);
		cout << "Training Accuracy " << testData(train_data,train_examples)<< endl;
		cout << "Validation Accuracy " << testData(validate_data,val_examples) << endl;
		cout << "Test Accuracy " << testData(test_data,test_examples) << endl;
		break;
	}
	return 0;
}