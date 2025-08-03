#include <iostream>
#include <stdexcept>
#include "tensor.hpp"

static void checkSameShape(const Tensor& a, const Tensor& b) {
    if (a.shape != b.shape) {
        throw std::runtime_error("Shape mismatch: Tensors must have the same shape.");
    }
}

static std::vector<int> broadcastShape(const std::vector<int>& a, const std::vector<int>& b) {
    int na = a.size(), nb = b.size();
    int n  = std::max(na, nb);
    std::vector<int> res(n);
    for (int i = 1; i <= n; ++i) {
        int da = (i <= na) ? a[na - i] : 1;
        int db = (i <= nb) ? b[nb - i] : 1;
        if (da == 1) da = db;
        if (db == 1) db = da;
        if (da != db)
            throw std::runtime_error("incompatible broadcast");
        res[n - i] = da;
    }
    return res;
}

static Tensor auto2d(const Tensor& t) {
    if (t.shape.size() == 1) {
        Tensor v({1, t.shape[0]});
        v.data = t.data;
        return v;
    }
    return t;
}

Tensor add(const Tensor& a, const Tensor& b) {
    checkSameShape(a, b);
    Tensor out(a.shape);
    for (size_t i = 0; i < a.data.size(); i++) {
        out.data[i] = a.data[i] + b.data[i];
    }
    return out;
}

Tensor mul(const Tensor& a, const Tensor& b) {
    checkSameShape(a, b);
    Tensor out(a.shape);
    for (size_t i = 0; i < a.data.size(); i++) {
        out.data[i] = a.data[i] * b.data[i];
    }
    return out;
}

// test version
Tensor dot(const Tensor& a_raw, const Tensor& b_raw)
{
    Tensor a = auto2d(a_raw);
    Tensor b = auto2d(b_raw);

    int ra = a.shape.size(), rb = b.shape.size();
    if (ra < 2 || rb < 2)
        throw std::runtime_error("after auto-2d need >=2-D");

    std::vector<int> batch_a(a.shape.begin(), a.shape.end() - 2);
    std::vector<int> batch_b(b.shape.begin(), b.shape.end() - 2);
    std::vector<int> batch_shape = broadcastShape(batch_a, batch_b);

    int M = a.shape[ra - 2], K = a.shape[ra - 1];
    int N = b.shape[rb - 1];
    if (K != b.shape[rb - 2])
        throw std::runtime_error("contracted dim mismatch");

    std::vector<int> out_shape = batch_shape;
    out_shape.push_back(M);
    out_shape.push_back(N);
    Tensor out(out_shape);

    int a_batch_vol = 1;
    for (int d : batch_a) a_batch_vol *= d;
    int b_batch_vol = 1;
    for (int d : batch_b) b_batch_vol *= d;

    int a_step = M * K;
    int b_step = K * N;
    int o_step = M * N;

    int batch_vol = 1;
    for (int d : batch_shape) batch_vol *= d;

    for (int flat = 0; flat < batch_vol; ++flat) {
        int tmp = flat;
        int a_idx = 0, b_idx = 0, stride = 1;
        for (int i = (int)batch_shape.size() - 1; i >= 0; --i) {
            int pos = tmp % batch_shape[i]; tmp /= batch_shape[i];
            int pa_idx = (i < (int)batch_a.size()) ? pos : 0;
            int pb_idx = (i < (int)batch_b.size()) ? pos : 0;
            a_idx += pa_idx * stride;
            b_idx += pb_idx * stride;
            stride *= batch_shape[i];
        }

        const float* pa = a.data.data() + a_idx * a_step;
        const float* pb = b.data.data() + b_idx * b_step;
        float* po = out.data.data() + flat * o_step;

        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j) {
                float s = 0.0f;
                for (int k = 0; k < K; ++k)
                    s += pa[i * K + k] * pb[k * N + j];
                po[i * N + j] = s;
            }
    }
    return out;
}