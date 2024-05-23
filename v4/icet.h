#ifndef ICET_H
#define ICET_H

#include <GL/glew.h>
#include <GL/glut.h>
#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <glm/glm.hpp>
#include "csv-parser/single_include/csv.hpp"
#include <fstream>
#include <cmath>
#include <limits>
#include <algorithm> 
#include <map>
#include <execution>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <future>
#include <queue>
#include "ThreadPool.h"
#include "utils.h"

using CovarianceMatrix = Eigen::Matrix<float, 3, 3>;
using CovarianceMap = std::map<int, std::map<int, CovarianceMatrix>>;
using MeanMap = std::map<int, std::map<int, Eigen::Vector3f>>;

class ICET{
public:
    ICET(Eigen::MatrixXf scan1, Eigen::MatrixXf scan2, int runlen);
    ~ICET();

    //avoid using static methods so we can run multiple ICETs at once?
    void step();

    void fitScan1();
    void prepScan2();
    void fitScan2();

    std::vector<std::vector<std::vector<int>>> sortSphericalCoordinates(Eigen::MatrixXf sphericalCoords);

    std::pair<float, float> findCluster(const Eigen::MatrixXf& sphericalCoords, int n, float thresh, float buff);

    Eigen::MatrixXf filterPointsInsideCluster(const Eigen::MatrixXf& selectedPoints, const Eigen::MatrixXf& clusterBounds);

    Eigen::MatrixXi testSigmaPoints(const Eigen::MatrixXf& selectedPoints, const Eigen::MatrixXf& lims);

    void fitCells1(const std::vector<int>& indices, int theta, int phi);

    Eigen::MatrixXf get_H(Eigen::Vector3f mu, Eigen::Vector3f angs);

    //algorithm params
    int rl;
    int numBinsPhi;  
    int numBinsTheta;
    int n; 
    float thresh;
    float buff; 

    Eigen::MatrixXf points1;
    Eigen::MatrixXf points1Spherical;
    Eigen::MatrixXf points2;
    Eigen::MatrixXf points2_OG;
    Eigen::MatrixXf points2Spherical;
    Eigen::MatrixXf clusterBounds;
    Eigen::MatrixXf testPoints;

    CovarianceMap sigma1;
    CovarianceMap sigma2;
    MeanMap mu1;
    MeanMap mu2;
    CovarianceMap L;
    CovarianceMap U;
    std::vector<std::vector<std::vector<int>>> pointIndices1;
    std::vector<std::vector<std::vector<int>>> pointIndices2;

    int occupiedCount; //debug

    // state vector X--- [x, y, z, phi, theta, psi]
    Eigen::VectorXf X; //global solution vector
    Eigen::VectorXf X0; //initial estimate
    Eigen::VectorXf dx; //linear perterbation to X solved for during each iteration

    //for viz
    std::vector<Eigen::Vector3f> ellipsoid1Means;
    std::vector<Eigen::Matrix3f> ellipsoid1Covariances;
    std::vector<float> ellipsoid1Alphas;
    std::vector<Eigen::Vector3f> ellipsoid2Means;
    std::vector<Eigen::Matrix3f> ellipsoid2Covariances;
    std::vector<float> ellipsoid2Alphas;

private:

    std::string status; 

    ThreadPool pool;
    std::vector<std::future<void>> futures;

};

#endif