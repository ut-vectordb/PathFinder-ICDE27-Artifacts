#ifndef VISITED_SET_H
#define VISITED_SET_H

#include <immintrin.h>
#include <x86intrin.h>
#include <cstring>
#include "config.h"


// #define ANNS_VISITEDSET_USE_UNORDERED_SET
#define ANNS_VISITEDSET_USE_ROBIN_SET

#ifdef ANNS_VISITEDSET_USE_UNORDERED_SET
#include <unordered_set>
#endif

#ifdef ANNS_VISITEDSET_USE_ROBIN_SET
#include "tsl/robin_set.h"
#endif

namespace ANNS {
#ifdef ANNS_VISITEDSET_USE_UNORDERED_SET
    class VisitedSet {
    public:
        VisitedSet() = default;

        void init(IdxType num_elements) {
            _marks.clear();
        }

        void clear() {
            _marks.clear();
        }

        inline void prefetch(IdxType) const {
            // no-op for unordered_map
        }

        inline void set(IdxType idx) {
            _marks.insert(idx);
        }

        inline bool check(IdxType idx) const {
            return (_marks.count(idx) > 0);
        }

    private:
        std::unordered_set<IdxType> _marks;
    };
#elif defined(ANNS_VISITEDSET_USE_ROBIN_SET)
    class VisitedSet {
    public:
        VisitedSet() = default;

        void init(IdxType num_elements) {
            _marks.clear();
        }

        void clear() {
            _marks.clear();
        }

        inline void prefetch(IdxType) const {
            // no-op for unordered_map
        }

        inline void set(IdxType idx) {
            _marks.insert(idx);
        }

        inline bool check(IdxType idx) const {
            return (_marks.count(idx) > 0);
        }

    private:
        tsl::robin_set<IdxType> _marks;
    };
#else
    class VisitedSet {
        public:
            VisitedSet() = default;

            void init(IdxType num_elements) {
                _curValue = -1;
                _num_elements = num_elements;
                if (_marks != nullptr)
                    delete[] _marks;
                _marks = new MarkType[num_elements];
            }

            void clear() {
                _curValue++;
                if (_curValue == 0) {
                    memset(_marks, 0, sizeof(MarkType) * _num_elements);
                    _curValue++;
                }
            }

            inline void prefetch(IdxType idx) const {
                _mm_prefetch((char *)_marks + idx, _MM_HINT_T0);
                // do nothing.
            }

            inline void set(IdxType idx) { 
                _marks[idx] = _curValue; 
            }

            inline bool check(IdxType idx) const { 
                return _marks[idx] == _curValue; 
            }

            ~VisitedSet() { 
                delete[] _marks; 
            }

        private:
            MarkType _curValue;
            MarkType* _marks = nullptr;
            IdxType _num_elements;
    };
#endif
}

#endif // VISITED_SET_H