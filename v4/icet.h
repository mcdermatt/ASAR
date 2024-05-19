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

    std::vector<std::vector<std::vector<int>>> sortSphericalCoordinates(Eigen::MatrixXf sphericalCoords);

    std::pair<float, float> findCluster(const Eigen::MatrixXf& sphericalCoords, int n, float thresh, float buff);

    Eigen::MatrixXf filterPointsInsideCluster(const Eigen::MatrixXf& selectedPoints, const Eigen::MatrixXf& clusterBounds);

    Eigen::MatrixXi testSigmaPoints(const Eigen::MatrixXf& selectedPoints, const Eigen::MatrixXf& clusterBounds);

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
    Eigen::MatrixXf points2Spherical;
    Eigen::MatrixXf clusterBounds;
    Eigen::MatrixXf testPoints;

    CovarianceMap sigma1;
    CovarianceMap sigma2;
    MeanMap mu1;
    MeanMap mu2;
    CovarianceMap L;
    CovarianceMap U;

private:

    std::string status; 

};

#endif