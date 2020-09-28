#include <iostream>
#include <fstream>
#include <cmath>
#include <Eigen/Dense>
#include <vector>

using namespace Eigen;
using namespace std;


/**
 * Computes the dot product of regression coefficients and input data while correctly
 * handling zero order coefficients.
 */
template <typename DerivedA, typename DerivedB>
ArrayXd interceptDot(const ArrayBase<DerivedA>& x,const ArrayBase<DerivedB>& coef)
{
	return (x.matrix() * coef.head(coef.size() - 1).matrix()).array() + coef(coef.size() - 1);
}

/**
 * Computes the sigmoid function i.e. (1 + e^(-z))^(-1) elementwise.
 */
template <typename Derived>
ArrayXd sigmoid(const ArrayBase<Derived>& z)
{
	return 1. / (1. + exp(-z));
}

/**
 * Return the log-likelihood for the logistic regression model.
 */
template <typename DerivedA, typename DerivedB, typename DerivedC>
typename DerivedA::Scalar logLikelihood(const ArrayBase<DerivedA>& x, const ArrayBase<DerivedB>& y, const ArrayBase<DerivedC>& coef)
{
	ArrayXd z = interceptDot(x, coef);
	return (y * log(sigmoid(z)) + (1 - y) * log(sigmoid(-z))).sum();
}

/**
 * Return the gradient of the log-likelihood for the logistic regression model.
 */
template <typename DerivedA, typename DerivedB, typename DerivedC>
ArrayXd logLikelihoodGrad(const ArrayBase<DerivedA>& x, const ArrayBase<DerivedB>& y, const ArrayBase<DerivedC>& coef)
{
	ArrayXXd xHom(x.rows(), x.cols() + 1);
	xHom << x, ArrayXd::Constant(x.rows(), 1, 1);

	ArrayXd z = (xHom.matrix() * coef.matrix()).array();
	ArrayXXd weighted = xHom.colwise() * (y - sigmoid(z));

	return weighted.colwise().sum().transpose() / x.rows();
}

/**
 * Parse a csv file into an eigen array or matrix object.
 * (https://stackoverflow.com/a/39146048)
 */
template<typename M>
M loadCSV(const std::string & path) 
{
    ifstream indata;
    indata.open(path);
    string line;
    vector<double> values;
    uint rows = 0;
    while (std::getline(indata, line)) {
        stringstream lineStream(line);
        string cell;
        while (getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, RowMajor> >(values.data(), rows, values.size()/rows);
}


int main(int argc, char *argv[])
{	
	if (argc < 2) {
		cout << "No input data found!" << endl;
		return 1;
	}
	string file = argv[1];
		
	// load data
	ArrayXXd data = loadCSV<ArrayXXd>(file);
	ArrayXXd x = data.leftCols(data.cols() - 1);
	ArrayXd y = data.col(data.cols() - 1);

	int n_features = x.cols();
	int n_data = x.rows();
	double lr = 0.5;
	double steps = 10000;

	// initiate regression coefficients 
	ArrayXd theta = ArrayXd::Constant(n_features + 1, 1, 1);

	vector<double> lls;
	for (int i=0; i<steps; i++) {
		// eval current log loss
		double ll = logLikelihood(x, y, theta);
		lls.push_back(ll);

		// maximize by gradient step
		ArrayXd llGrad = logLikelihoodGrad(x, y, theta);
		theta += lr * llGrad;
	}
	
	cout << "after " << lls.size() << " steps log-likelihood = " << lls[lls.size() - 1] << endl;
	cout << "learned coefficients = " << theta.transpose() << endl;

	ArrayXi yPred = (sigmoid(interceptDot(x, theta)) > 0.5).cast<int>();
	cout << "accurary = " << (double)((yPred - (y.cast<int>())) == 0).count() / (double)x.rows() << endl;

	return 0;
}
