#include <GL/glew.h>
#include <GL/glut.h>
#include <Eigen/Dense>
#include <random>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>
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
#include "icet.h"

using namespace Eigen;
using namespace std;

// Constructor implementation
ICET::ICET(MatrixXf scan1, MatrixXf scan2, int runlen) : points1(scan1), points2(scan2), rl(runlen), pool(8) {

    // init hyperparameters
    numBinsPhi = 24;  // Adjust the number of bins as needed
    numBinsTheta = 75; // Adjust the number of bins as needed
    n = 10; //50; // min size of the cluster
    thresh = 0.3; // 0.1 indoor, 0.3 outdoor; // Jump threshold for beginning and ending radial clusters
    buff = 0.5; // 0.1 indoor, outdoor 0.5; //buffer to add to inner and outer cluster range (helps attract nearby distributions)

    clusterBounds.resize(numBinsPhi*numBinsTheta,6);
    testPoints.resize(numBinsPhi*numBinsTheta*6,3);

    auto before = std::chrono::system_clock::now();
    auto beforeMs = std::chrono::time_point_cast<std::chrono::milliseconds>(before);

    fitScan1();

    auto after1 = std::chrono::system_clock::now();
    auto after1Ms = std::chrono::time_point_cast<std::chrono::milliseconds>(after1);
    auto elapsedTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(after1Ms - beforeMs).count();
    std::cout << "Fit spherical voxels and guassians for scan 1 in: " << elapsedTimeMs << " ms" << std::endl;

}

ICET::~ICET() {
    // Destructor implementation
}

void ICET::fitScan1(){

    // auto beforesort = std::chrono::system_clock::now();
    // auto beforesortms = std::chrono::time_point_cast<std::chrono::milliseconds>(beforesort);

    points1Spherical = utils::cartesianToSpherical(points1);

    // auto aftersort = std::chrono::system_clock::now();
    // auto aftersortms = std::chrono::time_point_cast<std::chrono::milliseconds>(aftersort);
    // auto ets = std::chrono::duration_cast<std::chrono::milliseconds>(aftersortms - beforesortms).count();
    // std::cout << "c2s took: " << ets << " ms" << std::endl;

    // Sort sphericalCoords based on radial distance
    vector<int> index(points1Spherical.rows());
    iota(index.begin(), index.end(), 0);
    // sort(index.begin(), index.end(), [&](int a, int b) {
    sort(std::execution::par, index.begin(), index.end(), [&](int a, int b) {
        return points1Spherical(a, 0) < points1Spherical(b, 0); // Sort by radial distance
    });
    for (int i = 0; i < points1Spherical.rows(); i++) {
        if (index[i] != i) {
            points1Spherical.row(i).swap(points1Spherical.row(index[i]));
            std::swap(index[i], index[index[i]]); // Update the index
        }
    }

    //get spherical coordiantes and fit gaussians to points from first scan 
    vector<vector<vector<int>>> pointIndices1 = sortSphericalCoordinates(points1Spherical);

    // Define a lambda function to wrap the member function fitCells1
    auto task = [this](const std::vector<int>& indices, int theta, int phi) {
        this->fitCells1(indices, theta, phi);
    };


    int count = 0;
    for (int phi = 0; phi < numBinsPhi; phi++){
        for (int theta = 0; theta< numBinsTheta; theta++){
            // Retrieve the point indices inside angular bin
            const vector<int>& indices = pointIndices1[theta][phi];
            futures.push_back(pool.enqueue(task, indices, theta, phi));
        }
    }
    // Wait for all tasks to complete
    for (auto &fut : futures) {
        fut.get();
    }
}

void ICET::fitCells1(const vector<int>& indices, int theta, int phi){
    float innerDistance;
    float outerDistance;

    cout << "theta: " << theta << "  phi: " << phi << endl;
    if (phi * numBinsTheta + theta >= numBinsPhi*numBinsTheta){
        cout << " problem " <<endl;
        return;
    }


    // only calculate inner/outer bounds if there are a sufficient number of points in the spike 
    if (indices.size() > n) {
        // Use the indices to access the corresponding rows in sortedPointsSpherical
        MatrixXf selectedPoints = MatrixXf::Zero(indices.size(), points1Spherical.cols());
        for (int i = 0; i < indices.size(); ++i) {
            selectedPoints.row(i) = points1Spherical.row(indices[i]);
        }

        // find inner and outer bounds for each theta/phi bin
        pair<float, float> clusterDistances = findCluster(selectedPoints, n, thresh, buff);
        innerDistance = clusterDistances.first;
        outerDistance = clusterDistances.second;

        //convert [desiredPhi][desiredTheta] to azimMin, azimMax, elevMin, elevMax
        float azimMin_i =  (static_cast<float>(theta) / numBinsTheta) * (2 * M_PI) ;
        float azimMax_i =  (static_cast<float>(theta+1) / numBinsTheta) * (2 * M_PI) ;
        float elevMin_i =  (static_cast<float>(phi) / numBinsPhi) * (M_PI) ;
        float elevMax_i =  (static_cast<float>(phi+1) / numBinsPhi) * (M_PI) ;
        //hold on to these values
        clusterBounds.row(numBinsTheta*phi + theta) << azimMin_i, azimMax_i, elevMin_i, elevMax_i, innerDistance, outerDistance;

        // find points from first scan inside voxel bounds and fit gaussians to each cluster
        MatrixXf filteredPoints = filterPointsInsideCluster(selectedPoints, clusterBounds.row(numBinsTheta*phi + theta));
        if (outerDistance > 0.1){
            MatrixXf filteredPointsCart = utils::sphericalToCartesian(filteredPoints);
            Eigen::VectorXf mean = filteredPointsCart.colwise().mean();
            Eigen::MatrixXf centered = filteredPointsCart.rowwise() - mean.transpose();
            Eigen::MatrixXf covariance = (centered.adjoint() * centered) / static_cast<float>(filteredPointsCart.rows() - 1);

            //hold on to means and covariances of clusters from scan1
            sigma1[theta][phi] = covariance;
            mu1[theta][phi] = mean;

            // get U and L ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigensolver(covariance);
            Eigen::Vector3f eigenvalues = eigensolver.eigenvalues().real();
            Eigen::Matrix3f eigenvectors = eigensolver.eigenvectors().real();
            // U[theta][phi] = eigenvectors; // was this --> eigen and TF have different convenions for outputting eigenvectors?
            U[theta][phi] = eigenvectors.transpose(); // test

            // create 6 2-sigma test points for each cluster and test to see if they fit inside the voxel
            MatrixXf axislen(3,3);
            axislen << eigenvalues[0], 0, 0,
                        0, eigenvalues[1], 0,
                        0, 0, eigenvalues[2];
            axislen = 2.0 * axislen.array().sqrt(); //theoretically should be *2 not *3 but this seems to work better

            MatrixXf rotated = axislen * U[theta][phi].transpose(); //was this
            // MatrixXf rotated = axislen * U[theta][phi]; //test

            Eigen::MatrixXf sigmaPoints(6,3);
            //converges faster on Ouster dataset, but won't work in simulated tunnel
            sigmaPoints.row(0) = mu1[theta][phi] + rotated.row(0).transpose(); //most compact axis
            sigmaPoints.row(1) = mu1[theta][phi] - rotated.row(0).transpose();
            sigmaPoints.row(2) = mu1[theta][phi] + rotated.row(1).transpose(); //middle
            sigmaPoints.row(3) = mu1[theta][phi] - rotated.row(1).transpose();
            sigmaPoints.row(4) = mu1[theta][phi] + rotated.row(2).transpose(); //largest axis
            sigmaPoints.row(5) = mu1[theta][phi] - rotated.row(2).transpose();

            // find out which test points fall inside the voxel bounds
            Eigen::MatrixXf sigmaPointsSpherical = utils::cartesianToSpherical(sigmaPoints);
            MatrixXi sigmaPointsInside = testSigmaPoints(sigmaPointsSpherical, clusterBounds.row(numBinsTheta*phi + theta));
            
            //see if each axis contains at least one test point within voxel
            if ((sigmaPointsInside.array() == 0).any() || (sigmaPointsInside.array() == 1).any()){
                L[theta][phi].row(0) << 1, 0, 0; 
            } 
            else{
                L[theta][phi].row(0) << 0, 0, 0;
                testPoints.row(6*(numBinsTheta*phi + theta)) = sigmaPoints.row(0).transpose();
                testPoints.row(6*(numBinsTheta*phi + theta)+1) = sigmaPoints.row(1).transpose();
            }
            if ((sigmaPointsInside.array() == 2).any() || (sigmaPointsInside.array() == 3).any()){
                L[theta][phi].row(1) << 0, 1, 0; 
            } 
            else{
                L[theta][phi].row(1) << 0, 0, 0;
                testPoints.row(6*(numBinsTheta*phi + theta)+2) = sigmaPoints.row(2).transpose();
                testPoints.row(6*(numBinsTheta*phi + theta)+3) = sigmaPoints.row(3).transpose();
            }
            if ((sigmaPointsInside.array() == 4).any() || (sigmaPointsInside.array() == 5).any()){
                L[theta][phi].row(2) << 0, 0, 1; 
            } 
            else{
                L[theta][phi].row(2) << 0, 0, 0;
                testPoints.row(6*(numBinsTheta*phi + theta)+4) = sigmaPoints.row(4).transpose();
                testPoints.row(6*(numBinsTheta*phi + theta)+5) = sigmaPoints.row(5).transpose();
            }
            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            // //update for drawing
            // float alpha1 = 0.3f;
            // ellipsoid1Means.push_back(mean);
            // ellipsoid1Covariances.push_back(covariance);
            // ellipsoid1Alphas.push_back(alpha1);
        }
    }
    // use 0 value as a flag for unoccupied voxels
    else{
        innerDistance = 0;
        outerDistance = 0;
        float azimMin_i =  (static_cast<float>(theta) / numBinsTheta) * (2 * M_PI) ;
        float azimMax_i =  (static_cast<float>(theta+1) / numBinsTheta) * (2 * M_PI) ;
        float elevMin_i =  (static_cast<float>(phi) / numBinsPhi) * (M_PI) ;
        float elevMax_i =  (static_cast<float>(phi+1) / numBinsPhi) * (M_PI) ;      
        clusterBounds.row(numBinsTheta*phi + theta) << azimMin_i, azimMax_i, elevMin_i, elevMax_i, innerDistance, outerDistance;
    }
}

void ICET::step(){
    cout << "rl: " << rl << endl;
    rl--;
}

vector<vector<vector<int>>> ICET::sortSphericalCoordinates(Eigen::MatrixXf sphericalCoords) {
    // Create a 2D vector of vectors to store point indices in each bin
    vector<vector<vector<int>>> pointIndices(numBinsTheta, vector<vector<int>>(numBinsPhi));

    // Iterate through each spherical coordinate
    for (int i = 0; i < sphericalCoords.rows(); ++i) {
        // Extract phi and theta values
        float theta = sphericalCoords(i, 1);
        float phi = sphericalCoords(i, 2);

        // Calculate bin indices
        int binTheta = static_cast<int>((theta / (2 * M_PI)) * numBinsTheta) % numBinsTheta;
        int binPhi = static_cast<int>((phi / M_PI) * numBinsPhi) % numBinsPhi;

        // Store the point index in the corresponding bin
        pointIndices[binTheta][binPhi].push_back(i);
    }

    // Return the vector of point indices
    return pointIndices;
}

pair<float, float> ICET::findCluster(const MatrixXf& sphericalCoords, int n, float thresh, float buff) {
    int numPoints = sphericalCoords.rows();

    float innerDistance = 0.0;
    float outerDistance = 0.0;
    vector<Vector3f> localPoints;

    for (int i = 0; i < numPoints; i++) {
        Vector3f point = sphericalCoords.row(i);

        // Check if the point is within the threshold of the last point
        if (!localPoints.empty() && std::abs(localPoints.back()(0) - point(0)) <= thresh) {
            // Add the point to the current cluster
            localPoints.push_back(point);
        } else {
            // Check if the cluster is sufficiently large
            if (localPoints.size() >= n) {
                // Found a sufficiently large cluster
                innerDistance = localPoints.front()(0) - buff;
                outerDistance = localPoints.back()(0) + buff;
                // cout << "Found cluster - Inner Distance: " << innerDistance << ", Outer Distance: " << outerDistance << endl;
                return {innerDistance, outerDistance};
            } else {
                // Reset the cluster if it's not large enough
                localPoints.clear();
                // Add the current point to start a new cluster
                localPoints.push_back(point);
            }
        }
    }
    // Check for the last cluster at the end of the loop
    if (localPoints.size() >= n) {
        // innerDistance = localPoints.front()(0);
        // outerDistance = localPoints.back()(0);
        if (localPoints.front()(0) !=0){
            innerDistance = localPoints.front()(0) - buff;
            outerDistance = localPoints.back()(0) + buff;
            // cout << "Found cluster - Inner Distance: " << innerDistance << ", Outer Distance: " << outerDistance << endl;
            return {innerDistance, outerDistance};
        }
        else{
            return {0.0, 0.0};
        }
    }

    return {innerDistance, outerDistance};
}

MatrixXf ICET::filterPointsInsideCluster(const MatrixXf& selectedPoints, const MatrixXf& clusterBounds) {
    int numPoints = selectedPoints.rows();
    int numClusters = clusterBounds.rows();

    MatrixXf filteredPoints(numPoints, 3);
    int filteredRowCount = 0;

    for (int i = 0; i < numClusters; i++) {
        float azimMin = clusterBounds(i, 0);
        float azimMax = clusterBounds(i, 1);
        float elevMin = clusterBounds(i, 2);
        float elevMax = clusterBounds(i, 3);
        float innerDistance = clusterBounds(i, 4);
        float outerDistance = clusterBounds(i, 5);

        for (int j = 0; j < numPoints; j++) {
            float azim = selectedPoints(j, 1);
            float elev = selectedPoints(j, 2);
            float r = selectedPoints(j, 0);

            // Check if the point is within the cluster bounds
            if (azim >= azimMin && azim <= azimMax &&
                elev >= elevMin && elev <= elevMax &&
                r >= innerDistance && r <= outerDistance) {
                // Add the point to the filteredPoints matrix
                filteredPoints.row(filteredRowCount++) = selectedPoints.row(j);
            }

            // If the current point is beyond the outer distance, break the inner loop
            if (r > outerDistance) {
                break;
            }
        }
    }

    // Resize the matrix to remove unused rows
    filteredPoints.conservativeResize(filteredRowCount, 3);

    return filteredPoints;
}

MatrixXi ICET::testSigmaPoints(const MatrixXf& selectedPoints, const MatrixXf& clusterBounds) {
    int numPoints = selectedPoints.rows();
    int numClusters = clusterBounds.rows();

    // Vector to store indices of filtered points
    vector<int> filteredIndices;

    for (int i = 0; i < numClusters; i++) {
        float azimMin = clusterBounds(i, 0);
        float azimMax = clusterBounds(i, 1);
        float elevMin = clusterBounds(i, 2);
        float elevMax = clusterBounds(i, 3);
        float innerDistance = clusterBounds(i, 4);
        float outerDistance = clusterBounds(i, 5);

        for (int j = 0; j < numPoints; j++) {
            float azim = selectedPoints(j, 1);
            float elev = selectedPoints(j, 2);
            float r = selectedPoints(j, 0);

            // Check if the point is within the cluster bounds
            if (azim >= azimMin && azim <= azimMax &&
                elev >= elevMin && elev <= elevMax &&
                r >= innerDistance && r <= outerDistance) {
                // Add the index to the filteredIndices vector
                filteredIndices.push_back(j);
            }

            // If the current point is beyond the outer distance, break the inner loop
            if (r > outerDistance) {
                break;
            }
        }
    }

    // Create a matrix from the indices
    MatrixXi filteredIndicesMatrix(filteredIndices.size(), 1);
    for (size_t i = 0; i < filteredIndices.size(); i++) {
        filteredIndicesMatrix(i, 0) = filteredIndices[i];
    }

    return filteredIndicesMatrix;
}