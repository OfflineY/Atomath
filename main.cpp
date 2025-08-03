#include <iostream>
#include "tensor.cpp"

int main() {
    Tensor A({2,2,2});
    Tensor B({2,2,2});

    A.array({4,5,6,1,4,5,6,1});
    B.array({7,8,9,2,4,5,6,1});
    
    add(A, B).print("added");
    mul(A, B).print("muled");
    dot(A, B).print("doted");

    A.at(1,1,1) = 99;
    std::cout << A.at(1,1,1) << std::endl;
    A.print("A after set");
    return 0;
}