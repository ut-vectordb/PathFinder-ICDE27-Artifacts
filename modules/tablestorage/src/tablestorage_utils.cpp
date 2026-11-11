#include "tablestorage_utils.h"

std::vector<float> fvecs_read(const char* fname, size_t& d_out, size_t& n_out) {
    FILE* f = fopen(fname, "rb");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    my_assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    my_assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    d_out = d;
    n_out = n;
    std::vector<float> x(n * d);
    std::vector<float> buf(n * (d + 1));
    size_t nr = fread(buf.data(), sizeof(float), n * (d + 1), f);
    my_assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++)
        std::memmove(x.data() + i * d, buf.data() + 1 + i * (d + 1), d * sizeof(float));

    buf.resize(n * d * sizeof(float));

    fclose(f);
    return x;
}

size_t get_file_size(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary | std::ios::ate);  // open at end
    if (!in) {
        std::cerr << "Error: Failed to open file '" << filename << "'\n";
        std::perror("Reason");
        throw std::runtime_error("Failed to open file for get_file_size. Filename: " + filename);
    }
    return static_cast<size_t>(in.tellg());  // position = size in bytes
}

std::vector<float> read_binary_float32(const std::string &filename) {
    std::vector<float> data;

    size_t bytes = get_file_size(filename);
    data.resize(bytes / 4);

    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Error: Failed to open file '" << filename << "'\n";
        std::perror("Reason");
        throw std::runtime_error("Failed to open file for read_binary_float32. Filename: " + filename);
    }

    in.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
    in.close();

    return data;
}

std::vector<float> csv_read(const char* fname, size_t& d_out, size_t& n_out, char sep, bool has_header) {
    std::vector<float> x;
    std::ifstream file(fname);
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << fname << std::endl;
        return x;
    }

    std::string line;
    size_t d = 0;
    size_t n = 0;

    if (has_header) {
        // Read the first line as (n, d)
        if (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string value;
            if (std::getline(ss, value, sep)) {
                try {
                    n = static_cast<size_t>(std::stoul(value));
                } catch (...) {
                    std::cerr << "Invalid n in CSV header: " << value << std::endl;
                    return x;
                }
            } else {
                std::cerr << "Missing n in CSV header." << std::endl;
                return x;
            }
            if (std::getline(ss, value, sep)) {
                try {
                    d = static_cast<size_t>(std::stoul(value));
                } catch (...) {
                    std::cerr << "Invalid d in CSV header: " << value << std::endl;
                    return x;
                }
            } else {
                std::cerr << "Missing d in CSV header." << std::endl;
                return x;
            }
            d_out = d;
            n_out = n;
        } else {
            std::cerr << "CSV file is empty." << std::endl;
            return x;
        }
    }

    size_t row_count = 0;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<float> row;

        while (std::getline(ss, value, sep)) {
            try {
                row.push_back(std::stof(value));
            }
            catch (const std::invalid_argument& e) {
                std::cerr << "Invalid value in CSV file: " << value << std::endl;
                return std::vector<float>();
            }
        }

        if (!has_header && row_count == 0) {
            d = row.size();
            d_out = d;
        } else if (row.size() != d) {
            std::cerr << "Inconsistent number of columns in CSV file." << std::endl;
            return std::vector<float>();
        }

        x.insert(x.end(), row.begin(), row.end());
        row_count++;
    }

    if (!has_header) {
        n = row_count;
        n_out = n;
    } else {
        // If header is present, check if the actual number of rows matches n
        if (row_count != n) {
            std::cerr << "Row count in CSV (" << row_count << ") does not match header n (" << n << ")." << std::endl;
            return std::vector<float>();
        }
    }

    return x;
}

bool needs_regeneration(const std::string& input_file, const std::string& output_file) {
    // Return true if output_file is missing or older than input_file
    if (!fs::exists(output_file))
        return true;  // Must regenerate because output does not exist

    if (!fs::exists(input_file))
        throw std::runtime_error("Input file does not exist: " + input_file);

    auto input_time = fs::last_write_time(input_file);
    auto output_time = fs::last_write_time(output_file);

    return input_time > output_time;  // input newer => needs regeneration
}