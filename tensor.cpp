#include <iostream>
#include <stdexcept>
#include "atomath.hpp"

static void checkSameShape(const Tensor& a, const Tensor& b) {
    if (a.shape != b.shape) {
        throw std::runtime_error("Shape mismatch: Tensors must have the same shape.");
    }
}

Tensor add(const Tensor& a, const Tensor& b) {
    checkSameShape(a, b);
    Tensor out(a.shape);
    for (size_t i = 0; i < a.shape.size(); i++) {
        out.data[i] = a.data[i] + b.data[i];
    }
    return out;
}

int main() {
    Tensor A({2, 3});
    Tensor B({2, 3});
    
    // Print the tensor's shape and data
    A.print();

    return 0;
}