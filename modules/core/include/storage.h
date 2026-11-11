#ifndef ANNS_STORAGE_H
#define ANNS_STORAGE_H

#include <limits>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <string>
#include <memory>
#include <xmmintrin.h>
#include <immintrin.h>
#include "config.h"
#include "distance.h"
#include <stdexcept>

using namespace std::literals;

namespace ANNS {

    // interface for storage
    class IStorage {
        public:
            virtual ~IStorage() = default;

            // I/O
            virtual void load_from_file(const std::string& bin_file, const std::string& label_file, 
                                        IdxType max_num_points = std::numeric_limits<IdxType>::max()) = 0;
            virtual void write_to_file(const std::string& bin_file, const std::string& label_file) = 0;

            // reorder the vector data
            virtual void reorder_data(const std::vector<IdxType>& new_to_old_ids) = 0;

            // discard all labels -- used to build a monolithic Label Nav Graph
            virtual void discard_labels() = 0;

            // get statistics
            virtual DataType get_data_type() const = 0;
            virtual IdxType get_num_points() const = 0;
            virtual void truncate(IdxType new_n) {};
            virtual IdxType get_dim() const = 0;

            // get data
            virtual std::vector<LabelType>* get_offseted_label_sets(IdxType idx) = 0;
            virtual char* get_vector(IdxType idx) = 0;
            virtual std::vector<LabelType>& get_label_set(IdxType idx) = 0;
            virtual inline void prefetch_vec_by_id(IdxType idx) const = 0;

            // obtain a point cloest to the center
            virtual IdxType choose_medoid(uint32_t num_threads, std::shared_ptr<DistanceHandler> distance_handler) = 0;

            // clean
            virtual void clean() = 0;
    };
    

    // obtain corresponding storage class
    std::shared_ptr<IStorage> create_storage(const std::string& data_type, bool verbose = true);
    std::shared_ptr<IStorage> create_storage(std::shared_ptr<IStorage> storage, IdxType start, IdxType end);
    std::shared_ptr<IStorage> create_storage_masked(std::shared_ptr<IStorage> storage, std::unordered_map<int, int> &old_to_new);


    // storage class
    template<typename T>
    class Storage : public IStorage {

        public:
            Storage(DataType data_type, bool verbose);
            Storage(std::shared_ptr<IStorage> storage); // copy
            Storage(std::shared_ptr<IStorage> storage, const std::vector<IdxType>& new_to_old_ids); // select rows
            Storage(std::shared_ptr<IStorage> storage, IdxType start, IdxType end); // zero-copy
            ~Storage() = default;       // TODO: Sometimes the Storage object re-uses other people's vector. in this case we should not call clean().

            // I/O
            void load_from_file(const std::string& bin_file, const std::string& label_file, IdxType max_num_points);
            void write_to_file(const std::string& bin_file, const std::string& label_file);

            // reorder the vector data
            void reorder_data(const std::vector<IdxType>& new_to_old_ids);

            // discard all labels -- used to build a monolithic Label Nav Graph
            void discard_labels() { dismiss_labels = true; }

            // get statistics
            DataType get_data_type() const { return data_type; };
            IdxType get_num_points() const { return num_points; };
            void truncate(IdxType new_n) { if (new_n <= num_points) num_points = new_n; else throw std::runtime_error("truncation invalid"); }
            IdxType get_dim() const { return dim; };

            // get data
            std::vector<LabelType>* get_offseted_label_sets(IdxType idx) { return label_sets + idx; }
            char* get_vector(IdxType idx) { return reinterpret_cast<char *>(vecs + (size_t)idx * dim); }
            std::vector<LabelType>& get_label_set(IdxType idx) {
                static std::vector<LabelType> empty = {1};
                if (dismiss_labels) { return empty; }
                return label_sets[idx];
            }
            inline void prefetch_vec_by_id(IdxType idx) const {
                // throw std::invalid_argument("Explosion!");
                for (size_t d = 0; d < prefetch_byte_num; d += 64) _mm_prefetch((const char *)(vecs + (size_t)idx * dim) + d, _MM_HINT_T0);
            }

            // obtain a point cloest to the center
            IdxType choose_medoid(uint32_t num_threads, std::shared_ptr<DistanceHandler> distance_handler);

            // clean
            void clean() {
                if (vecs) {
                    // delete[] vecs;
                    free(vecs);
                    vecs = nullptr;
                }
                if (label_sets) {
                    delete[] label_sets;
                    label_sets = nullptr;
                }
            }

            // friend class MappedStorage<T>;

        // private:
            DataType data_type;
            IdxType num_points, dim;
            T* vecs = nullptr;
            size_t prefetch_byte_num;
            bool dismiss_labels = false;
            std::vector<LabelType>* label_sets = nullptr;

            // for logs
            bool verbose;
    };

    // mapped storage class
    // we cannot inherit from Storage<T>, because Storage<T> didn't have a proper constructor to call to
    template<typename T>
    class MappedStorage : public IStorage {

        public:
            MappedStorage(std::shared_ptr<IStorage> storage, std::unordered_map<int, int> &old_to_new);

            ~MappedStorage() { clean(); }

            // I/O
            void load_from_file(const std::string& bin_file, const std::string& label_file, IdxType max_num_points) {
                throw "shouldn't be called"s;
            }
            void write_to_file(const std::string& bin_file, const std::string& label_file) {
                throw "shouldn't be called"s;
            }

            // reorder the vector data
            void reorder_data(const std::vector<IdxType>& new_to_old_ids) {}

            // discard all labels -- used to build a monolithic Label Nav Graph
            void discard_labels() {}

            // get statistics
            DataType get_data_type() const { return data_type; };
            IdxType get_num_points() const { return num_points; };
            IdxType get_dim() const { return dim; };

            // get data
            std::vector<LabelType>* get_offseted_label_sets(IdxType idx) { return nullptr; }
            char* get_vector(IdxType idx) { return reinterpret_cast<char *>(vecs + (new_to_orig[idx]) * this->dim); }
            std::vector<LabelType>& get_label_set(IdxType idx) { return label_sets[idx]; }
            inline void prefetch_vec_by_id(IdxType idx) const {
                // throw std::invalid_argument("Explosion!");
                for (size_t d = 0; d < prefetch_byte_num; d += 64) _mm_prefetch((const char *)(vecs + idx * dim) + d, _MM_HINT_T0);
            }

            // obtain a point cloest to the center
            IdxType choose_medoid(uint32_t num_threads, std::shared_ptr<DistanceHandler> distance_handler) {return 0;}

            // clean
            void clean() {
                // if (vecs)
                //     delete[] vecs;
                // if (label_sets)
                //     delete[] label_sets;
            }

            int map_old_to_new(int k) { return orig_to_new[k]; }
            // int map_new_to_old(int k) { return new_to_orig.at(k); }
            int map_new_to_old(int k) { return new_to_orig[k]; }

        private:
            T* vecs = nullptr;
            // std::unordered_map<int, int> orig_to_new;
            std::vector<int> orig_to_new;   // just allocate 1000000 ints -- what could happen?
            std::vector<int> new_to_orig;

            DataType data_type;
            IdxType num_points, dim;
            size_t prefetch_byte_num;
            std::vector<LabelType>* label_sets = nullptr;

            // for logs
            bool verbose;
    };
}



#endif // ANNS_STORAGE_H