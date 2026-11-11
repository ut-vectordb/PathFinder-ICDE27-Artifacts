#ifndef QUERYTYPE_H_INCLUDED

#include <cmath>
#include <limits>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <variant>
#include <algorithm>
#include <iostream>

#include <boost/assert.hpp>

namespace ANNS {

using namespace std;

// A small, self-contained implementation for conjunctive predicates over na attributes.
// C++17+ (uses std::variant). Fast, clear, and easy to extend.

struct RangeCriterion {
    float ql = 0.0f, qr = 0.0f; // inclusive [ql, qr)
    bool contains(float x) const noexcept {
        return (x >= ql) && (x < qr);
    }
};

struct MemberCriterion {
    vector<float> values; // sorted, unique
    // Exact match (fast). If you need tolerance, see contains_eps below.
    bool contains(float x) const noexcept {
        return std::binary_search(values.begin(), values.end(), x);
    }
    bool contains_eps(float x, float eps) const noexcept {
        if (eps <= 0) return contains(x);
        auto it = std::lower_bound(values.begin(), values.end(), x - eps);
        for (; it != values.end() && *it <= x + eps; ++it) {
            if (std::fabs(*it - x) <= eps) return true;
        }
        return false;
    }
    // Ensure sorted/unique for binary_search
    void normalize() {
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end());
    }
};

class QueryType {
public:
    enum class PredType { RANGE, MEMBER };

    QueryType(int na = 0) : na_(na), preds_(na) {}

    int na() const noexcept { return na_; }

    // Set per-attribute predicates
    void set_range(int i, float ql, float qr) {
        check_index(i);
        preds_[i] = RangeCriterion{ql, qr};
    }

    void set_member(int i, vector<float> vals) {
        check_index(i);
        std::sort(vals.begin(), vals.end());
        MemberCriterion m{std::move(vals)};
        m.normalize();
        preds_[i] = std::move(m);
    }

    // If you need a global epsilon for float matches (optional).
    void set_member_epsilon(float eps) { member_eps_ = eps; }

    // Main check: returns true iff ALL attribute-level predicates pass (conjunction).
    // attr points to an array of length >= na(), with attr[i] the i-th attribute.
    bool check(const float* attr) const noexcept {
        for (int i = 0; i < na_; ++i) {
            const auto& v = preds_[i];
            if (std::holds_alternative<RangeCriterion>(v)) {
                const auto& rc = std::get<RangeCriterion>(v);
                if (!rc.contains(attr[i])) return false;
            } else {
                const auto& mc = std::get<MemberCriterion>(v);
                if (member_eps_ > 0) {
                    if (!mc.contains_eps(attr[i], member_eps_)) return false;
                } else {
                    if (!mc.contains(attr[i])) return false;
                }
            }
        }
        return true;
    }

    bool check_one(int i, const float* attr) const noexcept {
        const auto& v = preds_[i];
        if (std::holds_alternative<RangeCriterion>(v)) {
            const auto& rc = std::get<RangeCriterion>(v);
            if (!rc.contains(attr[i])) return false;
        } else {
            const auto& mc = std::get<MemberCriterion>(v);
            if (member_eps_ > 0) {
                if (!mc.contains_eps(attr[i], member_eps_)) return false;
            } else {
                if (!mc.contains(attr[i])) return false;
            }
        }
        return true;
    }

    std::string boolean_string(const float* attr) const {
        std::string out;
        for (int i = 0; i < na_; ++i) {
            const auto& v = preds_[i];
            bool res = false;
            if (std::holds_alternative<RangeCriterion>(v)) {
                const auto& rc = std::get<RangeCriterion>(v);
                res = rc.contains(attr[i]);
            } else {
                const auto& mc = std::get<MemberCriterion>(v);
                if (member_eps_ > 0) {
                    res = mc.contains_eps(attr[i], member_eps_);
                } else {
                    res = mc.contains(attr[i]);
                }
            }
            out += (res ? '1' : '0');
        }
        return out;
    }

    bool get_range_bounds(int i, float &L, float &R) const {
        check_index(i);
        if (const auto* rc = std::get_if<RangeCriterion>(&preds_[i])) {
            L = rc->ql; R = rc->qr;
            return true;
        }
        return false;
    }

    bool get_member_set(int i, vector<float> &set) const {
        check_index(i);
        if (const auto* rc = std::get_if<MemberCriterion>(&preds_[i])) {
            set = rc->values;
            return true;
        }
        return false;
    }

    operator std::vector<std::pair<float,float>>() const {
        std::vector<std::pair<float,float>> out;
        out.reserve(na_);

        for (int i = 0; i < na_; ++i) {
            if (auto rc = std::get_if<RangeCriterion>(&preds_[i])) {
                out.emplace_back(rc->ql, rc->qr);
            } else {
                // MemberCriterion -> dummy values
                out.emplace_back(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
            }
        }
        return out;
    }

    friend std::ostream& operator<<(std::ostream& out, const QueryType& q);

// private:
    void check_index(int i) const {
        if (i < 0 || i >= na_) throw std::out_of_range("QueryType: attribute index out of range");
    }

    int na_ = 0;
    float member_eps_ = 0.0f; // 0 => exact match
    std::vector<std::variant<RangeCriterion, MemberCriterion>> preds_;
};

struct DisjunctiveQueryType {
    std::vector<QueryType> clauses;
    DisjunctiveQueryType() {}

    DisjunctiveQueryType(QueryType &one_liner, int na) {
        // this function splits a querytype into .na() / na objects
        BOOST_VERIFY(na > 0);
        BOOST_VERIFY(one_liner.na() % na == 0);
        int many = one_liner.na() / na;
        clauses.resize(many, na);
        for (int i = 0; i < many; i++) {
            for (int j = 0; j < na; j++) {
                clauses[i].preds_[j] = one_liner.preds_[i * na + j];
            }
        }
    }

    bool check(const float* attr) const noexcept {
        for (auto &q : clauses) {
            if (q.check(attr)) return true;
        }
        return false;
    }

    bool check_one(int i, const float* attr) const noexcept {
        for (auto &q : clauses) {
            if (q.check_one(i, attr)) return true;
        }
        return false;
    }
};

struct QueryCollection {
// private:
public:
    // raw pointers to avoid ownership
    const std::vector<QueryType>* qt_ = nullptr;
    const std::vector<DisjunctiveQueryType>* dq_ = nullptr;

public:
    // implicit constructors
    QueryCollection(const std::vector<QueryType>* p) : qt_(p) {}
    QueryCollection(const std::vector<DisjunctiveQueryType>* p) : dq_(p) {}

    // implicit constructors from references
    QueryCollection(const std::vector<QueryType>& ref) : qt_(&ref) {}
    QueryCollection(const std::vector<DisjunctiveQueryType>& ref) : dq_(&ref) {}

    // check(i, attrs)
    bool check(size_t i, const float* attrs) const {
        if (qt_) return (*qt_)[i].check(attrs);
        if (dq_) return (*dq_)[i].check(attrs);
        throw std::runtime_error("QueryCollection: uninitialized");
    }

    bool check_one(size_t i, int j, const float* attrs) const {
        if (qt_) return (*qt_)[i].check_one(j, attrs);
        if (dq_) return (*dq_)[i].check_one(j, attrs);
        throw std::runtime_error("QueryCollection: uninitialized");
    }

    vector<QueryType> get_clauses(size_t i) const {
        if (qt_) return { (*qt_)[i] };
        if (dq_) return (*dq_)[i].clauses;
        return {};
    }

    // size()
    size_t size() const {
        if (qt_) return qt_->size();
        if (dq_) return dq_->size();
        return 0;
    }

    // convenience
    bool empty() const { return size() == 0; }
};

// Small struct holding parsed header metadata
struct ParsedHeader {
    std::vector<std::string> names;               // full header tokens
    std::vector<QueryType::PredType> types;       // type per column

    void describe() const {
        cout << "Parsed 3 query columns:" << endl;
        for (int i = 0; i < names.size(); i++) {
            string type = (types[i] == QueryType::PredType::RANGE ? "RANGE" : "HASH");
            cout << " #" << i << " name = \""
                 << names[i] << "\", type = "
                 << type << endl;
        }
    }
};

// Parse the header line from a TSV and identify column types
ParsedHeader parse_header_line(const std::string& header_line);

// Parse one TSV row into a QueryType, using parsed header metadata
QueryType parse_row_to_query(const std::string& line, const ParsedHeader& H);

// Read the whole TSV file into a vector<QueryType>
std::pair<std::vector<QueryType>, ParsedHeader> read_queries_from_tsv(const std::string& path);

std::pair<std::vector<DisjunctiveQueryType>, ParsedHeader> read_disjunctive_queries_from_tsv(const std::string& path, int na);

}

#define QUERYTYPE_H_INCLUDED
#endif