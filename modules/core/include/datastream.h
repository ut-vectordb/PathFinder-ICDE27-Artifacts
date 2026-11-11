#ifndef DATASTREAM_H_INCLUDED

#include <cstddef>
#include <vector>

// A tiny streaming interface + adapter for your existing in-memory data.
// If your data source is a file/reader, implement IDataStream for it.

struct IDataStream {
    virtual ~IDataStream() = default;
    virtual size_t num_cols() const = 0;
    virtual void reset() = 0;
    // Returns false when exhausted. row points to a contiguous array of length num_cols().
    virtual bool next(const float*& row) = 0;
};

// simple adapter for tabular data stored as vector<vector<T>>

struct Vector2DStream : IDataStream {
    const std::vector<std::vector<float>> &data;
    size_t i = 0, na = 0;
    Vector2DStream(const std::vector<std::vector<float>> &d) : data(d) {
        na = data.empty() ? 0 : data[0].size();
    }
    size_t num_cols() const override { return na; }
    void reset() override { i = 0; }
    bool next(const float*& row) override {
        if (i >= data.size()) return false;
        row = data[i].data();
        ++i;
        return true;
    }
};

struct Vector1DStream : IDataStream {
    const std::vector<float> &data;
    size_t i = 0, na = 0;
    Vector1DStream(const std::vector<float> &d, int na) : data(d), na(na) {}
    size_t num_cols() const override { return na; }
    void reset() override { i = 0; }
    bool next(const float*& row) override {
        if (i * na >= data.size()) return false;
        row = data.data() + i * na;
        ++i;
        return true;
    }
};

#define DATASTREAM_H_INCLUDED
#endif