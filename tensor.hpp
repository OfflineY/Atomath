#pragma once
#include <iostream>
#include <vector>

struct Tensor {
    std::vector<int> shape;
    std::vector<float> data;
    std::vector<float> gradients;

    // Constructor to initialize a tensor with a given shape
    // and fill it with zeros.
    Tensor(std::vector<int> s) : shape(std::move(s)) {
        int n = 1;
        for (int d : shape) n *= d;
        data.resize(n, 0.0f);
    }

    // Function to print the tensor shape and data
    void print() const {
        std::cout << "[Tensor]\nShape: [";
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i];
            if (i < shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]\nData: [";
        for (size_t i = 0; i < data.size(); ++i) {
            std::cout << data[i];
            if (i < data.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    }
};