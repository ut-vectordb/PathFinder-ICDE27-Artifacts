// Reads a tab-separated file and converts each row to a QueryType.
// Header format example:
//   Attr_Name_1_Range<TAB>Attr_Name_2_Categorical<TAB>...
//
// Cell formats:
//   *_Range:        "L,R"              (floats; inclusive [L, R))
//   *_Categorical:  "v1,v2,v3,..."     (floats; exact match)
//
// Assumptions:
// - Categorical values are numeric (e.g., 1,2,6). If you truly have strings,
//   change MemberCriterion to store strings and adapt QueryType accordingly.
// - All columns must have a recognized suffix: "Range" or "Categorical".
// - Blank cells are treated as “no constraint”; we set a pass-all range [-inf, +inf).

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cctype>
#include <limits>
#include <stdexcept>
#include <algorithm>

#include "querytype.h"

namespace ANNS {

namespace detail {

inline void trim_inplace(std::string& s) {
    auto not_space = [](unsigned char ch){ return !std::isspace(ch); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
}

inline std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> out;
    std::string cur;
    std::istringstream iss(s);
    while (std::getline(iss, cur, delim)) {
        trim_inplace(cur);
        out.push_back(cur);
    }
    // handle trailing empty token (e.g., "a,b,")
    if (!s.empty() && s.back() == delim) out.emplace_back();
    return out;
}

inline bool ends_with(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() &&
           std::equal(s.end()-suf.size(), s.end(), suf.begin(), suf.end());
}

inline float parse_float_or_throw(const std::string& tok, const std::string& context) {
    char* end = nullptr;
    const char* cstr = tok.c_str();
    errno = 0;
    float v = std::strtof(cstr, &end);
    if (errno != 0 || end == cstr) {
        throw std::runtime_error("Failed to parse float '" + tok + "' in " + context);
    }
    return v;
}

} // namespace detail

inline ParsedHeader parse_header_line(const std::string& header_line) {
    using PT = QueryType::PredType;
    auto cols = detail::split(header_line, '\t');
    if (cols.empty()) throw std::runtime_error("Empty header line");

    ParsedHeader H;
    H.names = cols;
    H.types.reserve(cols.size());

    for (const auto& name : cols) {
        if (detail::ends_with(name, "Range")) {
            H.types.push_back(PT::RANGE);
        } else if (detail::ends_with(name, "Categorical")) {
            H.types.push_back(PT::MEMBER);
        } else {
            throw std::runtime_error("Header column must end with 'Range' or 'Categorical': '" + name + "'");
        }
    }
    return H;
}

// Parse one TSV row into a QueryType instance
inline QueryType parse_row_to_query(const std::string& line, const ParsedHeader& H) {
    auto cells = detail::split(line, '\t');
    if (cells.size() != H.types.size()) {
        throw std::runtime_error("Row has " + std::to_string(cells.size()) +
                                 " columns, but header has " + std::to_string(H.types.size()));
    }

    const int na = static_cast<int>(H.types.size());
    QueryType q(na);

    for (int i = 0; i < na; ++i) {
        const auto& cell = cells[i];
        const auto type = H.types[i];

        // Blank cell => no constraint: use pass-all range [-inf, +inf).
        if (cell.empty()) {
            if (type == QueryType::PredType::RANGE) {
                q.set_range(i, -std::numeric_limits<float>::infinity(),
                               std::numeric_limits<float>::infinity());
            } else {
                // For categorical, blank => also pass-all: emulate with pass-all range
                q.set_range(i, -std::numeric_limits<float>::infinity(),
                               std::numeric_limits<float>::infinity());
            }
            continue;
        }

        if (type == QueryType::PredType::RANGE) {
            // Expect "L,R"
            auto toks = detail::split(cell, ',');
            if (toks.size() != 2) {
                throw std::runtime_error("Range cell must be 'L,R' but got '" + cell + "'");
            }
            float L = detail::parse_float_or_throw(toks[0], "range L");
            float R = detail::parse_float_or_throw(toks[1], "range R");
            q.set_range(i, L, R);
        } else { // Categorical
            // Expect "v1,v2,..."
            auto toks = detail::split(cell, ',');
            std::vector<float> vals;
            vals.reserve(toks.size());
            for (const auto& t : toks) {
                if (!t.empty()) vals.push_back(detail::parse_float_or_throw(t, "categorical value"));
            }
            q.set_member(i, std::move(vals));
        }
    }

    return q;
}

// Main API: read a TSV file and return vector<QueryType>
std::pair<std::vector<QueryType>, ParsedHeader> read_queries_from_tsv(const std::string& path) {
    std::ifstream fin(path);
    if (!fin) throw std::runtime_error("Failed to open file: " + path);

    std::string line;

    // Read header
    if (!std::getline(fin, line)) {
        throw std::runtime_error("Empty file (no header): " + path);
    }
    auto header = parse_header_line(line);

    // Parse rows
    std::vector<QueryType> out;
    while (std::getline(fin, line)) {
        if (line.empty()) continue; // skip blank lines
        out.emplace_back(parse_row_to_query(line, header));
    }
    return make_pair(out, header);
}

std::pair<std::vector<DisjunctiveQueryType>, ParsedHeader> read_disjunctive_queries_from_tsv(const std::string& path, int na) {
    std::ifstream fin(path);
    if (!fin) throw std::runtime_error("Failed to open file: " + path);

    std::string line;

    // Read header
    if (!std::getline(fin, line)) {
        throw std::runtime_error("Empty file (no header): " + path);
    }
    auto header = parse_header_line(line);
    if (header.names.size() == na) {
        header.names.insert(header.names.end(), header.names.begin(), header.names.end());
        header.types.insert(header.types.end(), header.types.begin(), header.types.end());
    }

    // Parse rows
    std::vector<DisjunctiveQueryType> out;
    while (std::getline(fin, line)) {
        if (line.empty()) continue; // skip blank lines
        QueryType q = parse_row_to_query(line, header);
        // split the line to form a disjunctiveQuery object
        out.emplace_back(q, na);
    }

    // Modify the header so that we only keep the starting na() columns
    header.names.resize(na);
    header.types.resize(na);
    return make_pair(out, header);
}

std::ostream& operator<<(std::ostream& out, const QueryType& q) {
    for (int i = 0; i < q.na_; ++i) {
        out << "  Dim " << i << ": ";
        if (const auto* rc = std::get_if<RangeCriterion>(&q.preds_[i])) {
            out << "[" << rc->ql << ", " << rc->qr << ")";
        } else if (const auto* mc = std::get_if<MemberCriterion>(&q.preds_[i])) {
            if (mc->values.size() < 10) {
                out << "[{";
                for (size_t j = 0; j < mc->values.size(); ++j) {
                    out << mc->values[j];
                    if (j + 1 < mc->values.size()) out << ", ";
                }
                out << "}]";
            } else {
                out << "[Member of set sized " << mc->values.size() << "]";
            }
        } else {
            out << "[<unset>]";
        }
        if (i + 1 < q.na_) out << '\n';
    }
    return out;
}

} // namespace ANNS

/* ------------------- Example -------------------
auto queries = read_queries_from_tsv("predicates.tsv");

// Example header:
//   Age_Range\tScore_Range\tTier_Categorical
// Example row:
//   18,65\t300,850\t1,2,3
//------------------------------------------------ */