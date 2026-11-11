#include "tablestorage.h"

using namespace std;

static bool ends_with(const string &str, const string &suffix) {
    if (str.length() < suffix.length()) return false;
    return str.compare(str.length() - suffix.length(), suffix.length(), suffix) == 0;
}

#ifndef BYPASS_B_TREE
char *TableStorage::alloc_new_row() const {
    my_assert(aligned_record_size != 0);
    return static_cast<char*>(std::aligned_alloc(32, aligned_record_size));
}
#endif

void TableStorage::load_table_storage_helper(
        const std::shared_ptr<ANNS::IStorage> base_storage,
        const std::vector<float> &attr_data,
        TableStorage &tbstorage) {

#ifndef BYPASS_B_TREE
    size_t n_dim, n_points, n_attrs;
    n_dim = base_storage->get_dim();
    n_points = base_storage->get_num_points();

    my_assert(n_dim > 0 && n_points > 0);
    n_attrs = attr_data.size() / n_points;
    my_assert(n_attrs > 0);
    my_assert(n_attrs * n_points == attr_data.size());

    tbstorage.nb = n_points;
    tbstorage.na = n_attrs;
    tbstorage.n_dim = n_dim;
    tbstorage.table_schema.resize(n_attrs, ColumnType::NUMERICAL);

    cout << "Number of points: " << n_points << ", Dimension: " << n_dim << ", Attributes: " << n_attrs << endl;
    
    // Insert data into TableStorage
    // The type conversion is for accessing "vecs" to check "vecs != nullptr"
    if (auto storetype = dynamic_pointer_cast<ANNS::Storage<float>>(base_storage); storetype) {
        tbstorage.dim_byte_count = 4;
        tbstorage.vector_bytes = (size_t)tbstorage.dim_byte_count * n_dim;
        tbstorage.attr_offset = roundUp(tbstorage.vector_bytes, 4);
        tbstorage.aligned_record_size = roundUp(tbstorage.attr_offset + n_attrs * sizeof(float), 32);

        if (storetype->vecs != nullptr)
            for (size_t i = 0; i < n_points; ++i) {
                auto vec_ptr = reinterpret_cast<float *>(base_storage->get_vector(i));
                auto attr_ptr = attr_data.data() + i * n_attrs;
                if (!tbstorage.insert(i, vec_ptr, attr_ptr)) {
                    cerr << "Failed to insert data for point " << i << endl;
                    abort();
                }
            }
        else
            cout << "Warning: using metadata-only storage as a hack" << endl;
    } else if (auto storetype = dynamic_pointer_cast<ANNS::Storage<uint8_t>>(base_storage); storetype) {
        tbstorage.dim_byte_count = 1;
        tbstorage.vector_bytes = (size_t)tbstorage.dim_byte_count * n_dim;
        tbstorage.attr_offset = roundUp(tbstorage.vector_bytes, 4);
        tbstorage.aligned_record_size = roundUp(tbstorage.attr_offset + n_attrs * sizeof(float), 32);

        if (storetype->vecs != nullptr)
            for (size_t i = 0; i < n_points; ++i) {
                auto vec_ptr = reinterpret_cast<float *>(base_storage->get_vector(i));
                auto attr_ptr = attr_data.data() + i * n_attrs;
                if (!tbstorage.insert(i, vec_ptr, attr_ptr)) {
                    cerr << "Failed to insert data for point " << i << endl;
                    abort();
                }
            }
        else
            cout << "Warning: using metadata-only storage as a hack" << endl;
    } else {
        cout << "Unknown datatype" << endl;
    }
#else
    size_t n_dim, n_points, n_attrs;
    n_dim = base_storage->get_dim();
    n_points = base_storage->get_num_points();

    my_assert(n_dim > 0 && n_points > 0);
    n_attrs = attr_data.size() / n_points;
    my_assert(n_attrs > 0);
    my_assert(n_attrs * n_points == attr_data.size());

    tbstorage.nb = n_points;
    tbstorage.na = n_attrs;
    tbstorage.n_dim = n_dim;
    tbstorage.table_schema.resize(n_attrs, ColumnType::NUMERICAL);

    cout << "Number of points: " << n_points << ", Dimension: " << n_dim << ", Attributes: " << n_attrs << endl;

    tbstorage.base_storage = base_storage;
    tbstorage.attr_data = attr_data;
    tbstorage.dim_byte_count = std::dynamic_pointer_cast<ANNS::Storage<uint8_t>>(base_storage) ? 1 : 4;
    tbstorage.vector_bytes = n_dim * tbstorage.dim_byte_count;
    tbstorage.live_keys.resize(n_points);
    for (size_t i = 0; i < n_points; ++i) tbstorage.live_keys[i] = i;
#endif
}

void TableStorage::describe() const {
    // TableStorage: nb rows of (%d dimensional vector, %d attributes)
    std::cout << "TableStorage: " << nb << " rows of (" << n_dim << " dimensional vector, " << na << " attributes)" << std::endl;
#ifdef BYPASS_B_TREE
    std::cout << "Storage mode: direct arrays (BYPASS_B_TREE; read-only)" << std::endl;
#else
    std::cout << "B+ Tree stats: " << std::endl;
    std::cout << "  Inner Nodes: " << this->tree.getNumInnerNodes() << std::endl;
    std::cout << "  Leaf Nodes:  " << this->tree.getNumLeafNodes() << std::endl;
    std::cout << "  Total Size:  " << this->tree.size() << " items" << std::endl;
#endif
}

bool TableStorage::insert(KeyType key, const float *vecdata, const float *attrdata) {
#ifndef BYPASS_B_TREE
    char *row = alloc_new_row();
    if (row == nullptr) {
        std::cerr << "Failed to allocate memory for new row." << std::endl;
        return false;
    }
    memcpy(get_attr(row), attrdata, na * sizeof(float));
    memcpy(get_vector(row), vecdata, this->vector_bytes);
    ValueType value(row);
    bool ok = this->tree.insert(key, value);
    if (ok) {
        std::lock_guard<std::mutex> lock(live_key_mutex);
        live_key_set.insert(key);
    } else {
        std::free(row);
    }
    return ok;
#else
    throw std::runtime_error("BYPASS_B_TREE supports read-only experiments");
#endif
}

void *TableStorage::lookup(KeyType key) {
    ValueType res;

#ifndef BYPASS_B_TREE
    bool ret = this->tree.lookup(key, res);

    if (ret == false) {
        std::cerr << "Key " << key << " not found in TableStorage." << std::endl;
        return nullptr;
    } else {
        if (res.getValue() == nullptr) {
            std::cerr << "Value for key " << key << " is null." << std::endl;
            return nullptr;
        }
        return res.getValue();
    }

#else
    // bool ret = key_to_row_map.count(key) > 0;
    // if (ret) res = key_to_row_map[key];
    return key < nb ? (void *)(size_t(key) + 1) : nullptr;
#endif
}

void *TableStorage::find(KeyType key) {
    ValueType res;

#ifndef BYPASS_B_TREE
    bool ret = this->tree.lookup(key, res);
    if (!ret || res.getValue() == nullptr) {
        return nullptr;
    }
    return res.getValue();
#else
    return lookup(key);
#endif
}

std::vector<TableStorage::KeyType> TableStorage::keys() {
    std::vector<KeyType> out;
#ifndef BYPASS_B_TREE
    std::lock_guard<std::mutex> lock(live_key_mutex);
    out.reserve(live_key_set.size());
    for (KeyType key : live_key_set) out.push_back(key);
    std::sort(out.begin(), out.end());
#else
    out = live_keys;
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
#endif
    return out;
}

bool TableStorage::remove(KeyType key) {
#ifndef BYPASS_B_TREE
    // NOTE: This doesn't release the row storage; We can't just add a dtor for the ValueType struct, because the dtor will be called every time the B+ Tree adjusts the leaf storage (it will be called in conjunction with Placement New), leading to double free. Maybe we need to work it into the EBR logic, but not for now.
    bool ok = this->tree.remove(key);
    if (ok) {
        std::lock_guard<std::mutex> lock(live_key_mutex);
        live_key_set.erase(key);
    }
    return ok;
#else
    throw std::runtime_error("BYPASS_B_TREE supports read-only experiments");
#endif
}
