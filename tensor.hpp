#pragma once
#include <iostream>
#include <vector>

struct Tensor {
    std::vector<int> shape;
    std::vector<int> strides;
    std::vector<float> data;
    int size = 1;

    // initialize a tensor with a given shape
    // fill it with zeros.
    Tensor(std::vector<int> s) : shape(std::move(s)), strides(shape.size()) {
        // row-major, last-dimension fastest
        for (int i = (int)shape.size() - 1; i >= 0; --i) {
            strides[i] = size;
            size *= shape[i];
        }
        data.resize(size, 0.0f);
    }

    Tensor& array(std::initializer_list<float> values) {
        if ((int)values.size() != size)
            throw std::runtime_error("values/size mismatch: values size error");
        std::copy(values.begin(), values.end(), data.begin());
        return *this;
    }

    float& at(const std::vector<int>& p) {
        int index = idx(p);
        return data[index];
    }
    const float& at(const std::vector<int>& p) const {
        int index = idx(p);
        return data[index];
    }

    template<typename... Args>
    float& at(Args... args) {
        std::vector<int> indices{args...};
        return at(indices);
    }

    int idx(const std::vector<int>& indices) const {
        if (indices.size() != shape.size())
            throw std::out_of_range("rank mismatch");
        int offset = 0;
        for (int i = (int)shape.size() - 1; i >= 0; --i) {
            int d = indices[i];
            if (d < 0 || d >= shape[i])
                throw std::out_of_range("index out of range");
            offset += d * strides[i];
        }
        return offset;
    }

    // print the tensor shape and data
    void print(const std::string& name = "") const {
        std::string k = (name != "") ? ": " + name : "";
        std::cout << "[Tensor"<< k;
        std::cout <<"]\nShape: [";
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i];
            if (i < shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]\nData:";
        print_data();
        std::cout << std::endl;
    }

private:
    void print_data(size_t dim = 0, int offset = 0) const {
        if (dim == shape.size() - 1) {
            std::cout << "[";
            for (int i = 0; i < shape[dim]; ++i) {
                std::cout << data[offset + i];
                if (i < shape[dim] - 1) std::cout << ", ";
            }
            std::cout << "]";
        } else {
            std::cout << "[";
            int step = 1;
            for (size_t d = dim + 1; d < shape.size(); ++d) step *= shape[d];
            for (int i = 0; i < shape[dim]; ++i) {
                print_data(dim + 1, offset + i * step);
                if (i < shape[dim] - 1) std::cout << ", ";
            }
            std::cout << "]";
        }
    }
};

Tensor add(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor dot(const Tensor& a_raw, const Tensor& b_raw);