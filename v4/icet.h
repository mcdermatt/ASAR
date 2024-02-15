// icet.h
#ifndef ICET_H
#define ICET_H

// Include necessary headers
#include <iostream>
#include <Eigen/Dense>
#include <vector>

using namespace Eigen;
using namespace std;

Eigen::MatrixXf generateEigenNormal(int rows, int cols, float mean, float stddev);

Eigen::MatrixXf generateEigenCovariance(int rows, const Eigen::Vector3f& mean, const Eigen::Matrix3f& covariance);

MatrixXf cartesianToSpherical(const MatrixXf& cartesianPoints);

Eigen::MatrixXf sphericalToCartesian(const Eigen::MatrixXf& sphericalPoints);

pair<float, float> findCluster(const MatrixXf& sphericalCoords, int n, float thresh, float buff);

vector<vector<vector<int>>> sortSphericalCoordinates(const MatrixXf& sphericalCoords, int numBinsTheta, int numBinsPhi);

MatrixXf filterPointsInsideCluster(const MatrixXf& selectedPoints, const MatrixXd& clusterBounds);

MatrixXi testSigmaPoints(const MatrixXf& selectedPoints, const MatrixXd& clusterBounds);

MatrixXf loadPointCloudCSV(string filename, string datasetType = "csv");

MatrixXf get_H(Eigen::Vector3f mu, Eigen::Vector3f angs);

Matrix3f R(float phi, float theta, float psi);

tuple<MatrixXf, MatrixXf, MatrixXf> checkCondition(MatrixXf HTWH);

Eigen::VectorXf icet(Eigen::MatrixXf points, Eigen::MatrixXf points2, Eigen::VectorXf X0, 
                        int numBinsPhi, int numBinsTheta, int n, float thresh, float buff, int runlen, bool draw);

#endif // ICET_H