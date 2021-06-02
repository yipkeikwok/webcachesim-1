#include <random>
#include <utility>
#include <cmath>
#include <ctime> // std::asctime(), std::localtime()
#include <string>
#include <sstream> // std::stringstream
#include <unordered_map> // latest_decision
/**
// NDEBUG is defined somewhere in this application. I do not know where.
#include <cassert>
*/
#include <cstdlib>
/** exporting calculateOPT() decisions::beginning */ 
#include <fstream>
/** exporting calculateOPT() decisions::end */ 
#include <LightGBM/application.h>
#include <LightGBM/c_api.h>
#include "lfo.h"
#include "random_helper.h"
#include "param_helper.h"

#ifndef ISSUE20210206b
#endif

// golden section search helpers
#define SHFT2(a,b,c) (a)=(b);(b)=(c);
#define SHFT3(a,b,c,d) (a)=(b);(b)=(c);(c)=(d);

// math model below can be directly copiedx
// static inline double oP1(double T, double l, double p) {
static inline double oP1(double T, double l, double p) {
    return (l * p * T * (840.0 + 60.0 * l * T + 20.0 * l*l * T*T + l*l*l * T*T*T));
}

static inline double oP2(double T, double l, double p) {
    return (840.0 + 120.0 * l * (-3.0 + 7.0 * p) * T + 60.0 * l*l * (1.0 + p) * T*T + 4.0 * l*l*l * (-1.0 + 5.0 * p) * T*T*T + l*l*l*l * p * T*T*T*T);
}

#ifndef DERIVEFEATURES_TESTING
#define DERIVEFEATURES_TESTING
#endif

#ifndef MODEL_REFITTING 
/** #define MODEL_REFITTING */
#endif

#define NR_NON_TIMEGAP_ELMNT 3

/*
  LFO: Learning from OPT
*/

LFOCache::LFOCache()
    : Cache(), 
      _objective(0)
{
    std::time_t result = std::time(nullptr);
    std::asctime(std::localtime(&result));
    std::stringstream stringstream0;
    stringstream0<<result;
    stringstream0>>LFO::timestamp;
}

bool LFOCache::lookup(SimpleRequest& req)
{


#ifdef EVICTION_LOGGING
    {
        auto &_req = dynamic_cast<AnnotatedRequest &>(req);
        current_t = req._t;

        if (_cacheMap.find(req._id) != _cacheMap.end()) {
            //hit
            auto it = last_timestamps.find(req._id);
            unsigned int hit =
                    static_cast<double>(current_t - it->second) / (_cacheSize * 1e6 / byte_million_req);
            hit = min((unsigned int) 255, hit);
            hits.emplace_back(hit);
            hit_timestamps.emplace_back(current_t / 65536);
        }

        auto it = last_timestamps.find(req._id);
        if (it == last_timestamps.end()) {
            last_timestamps.insert({_req._id, current_t});
        } else {
            it->second = current_t;
        }


        it = future_timestamps.find(req._id);
        if (it == future_timestamps.end()) {
            future_timestamps.insert({_req._id, _req._next_seq});
        } else {
            it->second = _req._next_seq;
        }
    }
#endif

    if(!(LFO::train_seq%LFO::windowSize) && LFO::train_seq!=0) {
        std::cerr<<"Concluding Window "<<(LFO::train_seq-1)/LFO::windowSize
            <<std::endl;
        LFO::conclude_window(_objective, getSize()); 
        LFO::init = false;
    } // if(!(LFO::train_seq%LFO::windowSize)) 

    bool in_cache=false;
    LFO::train_seq++;
    //202012231608::beginning
    if(_objective == LFO::OHR) 
        LFO::annotate(LFO::train_seq, req._id, req._size, 1.0); 
    else if(_objective == LFO::BHR)
        LFO::annotate(LFO::train_seq, req._id, req._size, req._size); 
    else {
        std::cerr<<"Invalid LFOCache::_objective " << _objective <<std::endl; 
        std::exit(EXIT_FAILURE);
    }
    //202012231608::end

    if(LFO::init) {
        #if 0
        if(!(LFO::train_seq%LFO::windowSize)) {
            std::cerr<<"Concluding Window "
            <<(LFO::train_seq-1)/LFO::windowSize <<std::endl;
            LFO::conclude_window(_objective, getSize()); 
            LFO::init = false;
        } 
        #endif
        uint64_t & obj = req._id;
        auto it = _lruCacheMap.find(obj);
        if (it != _lruCacheMap.end()) {
            // log hit
            auto & size = _size_map[obj];
            if(size != req._size) {
                std::cerr<<"LFOCache::lookup(), LRU object size not match"
                    <<std::endl;
                std::exit(1);
            }
            LOG("h", 0, obj.id, obj.size);
            _lruCacheList.splice(_lruCacheList.begin(), _lruCacheList, 
                it->second);
            in_cache = true;
        } else {
            in_cache = false;
        }
        #if 0
        std::cerr
            <<"lookup(): Object "<<obj<<" in_cache="<<in_cache
            <<", "<<_lruCacheMap.size()<<" items in cache"<<std::endl;
        if(obj==(uint64_t)4163405339) {
            std::cerr<<"lookup(): _lruCacheMap: ";
            for(auto it1: _lruCacheMap) 
                std::cerr<<it1.first<<" ";
            std::cerr<<std::endl;
        }
        #endif
    } else {
        if(LFO::init) { 
            std::cerr<<"LFO::init should == false at this point"<<std::endl;
            std::exit(EXIT_FAILURE);
        }

        uint64_t & obj = req._id;
        auto it = _cacheMap.left.find(
            std::make_pair((std::uint64_t)req._id, (double)req._size)
            );
        if (it != _cacheMap.left.end()) {
            // log hit
            auto & size = _size_map[obj];
            /** TESTING_CODE::verifying _size_map::beginning */
            if(size != req._size) {
                std::cerr<<"size != req._size"<<size<<" "<<req._size<<std::endl;
                std::exit(EXIT_FAILURE);
            }
            /** TESTING_CODE::verifying _size_map::end */
            LOG("h", 0, req._id, req._size);
            in_cache = true;

            // evict hit object if rehit_probability <.5
            double rehit_probability = LFO::calculate_rehit_probability(
                req, getSize()-getCurrentSize(), _objective
                );
            _cacheMap.left.replace_data(it, rehit_probability);
            /** 
            // hit() is from LRU implementation
            hit(it, rehit_probability);
            */
            if(rehit_probability<(double).5) {
                LFO::windowTrace.end()->decision_prediction=false;
                #ifdef VERIFY_ACCURACY_CALCULATION
                LFO::decision_array[(LFO::train_seq-1)%LFO::windowSize]=false;
                #endif
                // evict hit object
                KeyT evicted_req_id = evict();

                /** TESTING_CODE::beginning */
                // Object ID verification: Object ID of accessed and evicted object 
                //  should be same 
                if(evicted_req_id != req._id) {
                    std::cerr<<"LFOCache::lookup(): evicted_req_id != req._id, "
                        << "evicted_req_id= " << evicted_req_id  << ", "
                        << "req._id= " << req._id
                        << std::endl;
                    std::exit(EXIT_FAILURE);
                }
                /** TESTING_CODE::end */
            } else {
                /** TESTING_CODE::beginning */
                if(rehit_probability<(double).5) {
                    std::cerr<<"rehit_probability should >= .5"<<std::endl;
                    std::exit(EXIT_FAILURE);
                }
                /** TESTING_CODE::end */
                LFO::windowTrace.end()->decision_prediction=true;
                #ifdef VERIFY_ACCURACY_CALCULATION
                LFO::decision_array[(LFO::train_seq-1)%LFO::windowSize]=true;
                #endif
            }
        } else {
            // log miss
            in_cache = false;
        }
    }

    /**
    std::cerr
        <<"lookup(): "<<LFO::init<<" "<<LFO::train_seq<<" "<<req._id<<" "
        <<req._size<<" "<<in_cache<<std::endl;
    */
    return in_cache;
}

void LFO::conclude_window(int objective, uint64_t cache_size) 
{
    if((LFO::train_seq%LFO::windowSize)) {
        std::cerr<<"LFO::train_seq%LFO::windowSize should be 0"<<std::endl; 
        std::exit(1);
    }
    /**
    end-of-sliding window routine
    - deduce OPT decisions
    - derive features
    - train model
    */
    if(LFO::init) 
        std::cerr<<"LFO::init == true"<<std::endl;
    else
        std::cerr<<"LFO::init == false"<<std::endl;

    /**
    For LFO::calculateOPT(), available cache size should be the cache 
    capacity (i.e., total cache space). RSN: LFO's version of OPT makes
    caching decisions based on the cache capacity, not remaining cache size
    */
    LFO::calculateOPT(cache_size, objective);

    /** exporting calculateOPT() decisions::beginning */
    #if 0
    #endif
    std::stringstream stringstream0;
    stringstream0<<"calculateOPT.decision."<<LFO::timestamp<<".txt";
    std::string decision_filename;
    stringstream0>>decision_filename; 
    std::ofstream decision_ofstream;
    decision_ofstream.open(decision_filename, std::ios_base::app);
    for(auto it: LFO::windowTrace) {
        decision_ofstream
            <<it.seq<<" "
            <<it.id<<" "<<it.size<<" "<<it.toCache<<" "<<it.lastSeenIndex<<" "
            <<it.lastSeenCount; 
        decision_ofstream<<std::endl;
    }
    decision_ofstream.close();
    /** exporting calculateOPT() decisions::end */

    /** 
    */
    // TESTING CODE
    // purpose: make sure that access timestamps during the latest 
    //  window of each object are stored in statistics[object ID]
    std::cerr<<"statistics.size()= "<<LFO::statistics.size()<<", ";
    const auto it = LFO::statistics.cbegin();
    std::list<uint64_t> list0 = it->second;

    bool flag0=true; 
    uint32_t count0;
    count0=(uint32_t)0;
    for(const auto& it : LFO::statistics) {
        std::list listAccessTimestamps = it.second;
        if(listAccessTimestamps.size() > 1) {
            flag0=false;
            count0++;
        } else if(listAccessTimestamps.size() == 1) {
        } else {
            std::cerr
                <<"Object "<<it.first<<" has "<<listAccessTimestamps.size() 
                <<" elements"<<std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
    if(flag0) {
        std::cerr<<"only 1 element on each list"<<std::endl;
    } else {
        std::cerr<<count0<<" lists have >=1 elements"<<std::endl;
    }

    std::vector<float> labels;
    std::vector<int32_t> indptr;
    std::vector<int32_t> indices;
    std::vector<double> data;

    #if 0
    /** 202102281620::beginning */
    int window_id=(LFO::train_seq-1)/windowSize;
    int window_index=window_id*100000;
    uint64_t object_id=LFO::windowTrace[window_index].id;
    std::list<uint64_t> statistics_list = LFO::statistics[object_id];
    std::cerr<<"202102281620:"; 
    for(uint64_t timestamp0 : statistics_list) 
        std::cerr<<'\t'<<timestamp0;
    std::cerr<<'\t'<<statistics_list.size();
    std::cerr<<'\t'<<object_id;
    std::cerr<<std::endl;
    /** 202102281620::end */
    #endif

    LFO::deriveFeatures(labels, indptr, indices, data, LFO::sampling, 
        cache_size
        );

    /** TESTING_CODE::beginning */
    /** 
    // assert("labels.size()==LFO::windowTrace.size()");
    if(labels.size()!=LFO::windowTrace.size()) {
        std::cerr<<"labels.size()!=LFO::windowTrace.size()"<<std::endl;
        std::exit(EXIT_FAILURE);
    }

    // assert("indptr.size()==LFO::windowTrace.size()+1");
    if(indptr.size()!=(1+LFO::windowTrace.size())) {
        std::cerr<<"indptr.size()!=1+LFO::windowTrace.size()"<<std::endl;
        std::exit(EXIT_FAILURE);
    }

    uint64_t expected_indices_size = (uint64_t)0;
    for(auto const it: LFO::windowTrace) {
        expected_indices_size += LFO::statistics[it.id].size();
        expected_indices_size += (uint64_t)NR_NON_TIMEGAP_ELMNT;
    }
    if(indices.size()!=expected_indices_size) {
        std::cerr
            <<"indices.size()!=expected_indices_size"<<": "
            <<"indices.size() = "<<indices.size()<<", "
            <<"expected_indices_size = "<< expected_indices_size << ", "
            <<"windowTrace.size() = "<< LFO::windowTrace.size()
            <<std::endl;
        std::exit(EXIT_FAILURE);
    }
    if(indices.size()!=data.size()) {
        std::cerr
            <<"indices.size()!=data.size()"<<": "
            <<"indices.size() = "<<indices.size()<<", "
            <<"data.size() = "<< data.size()
            <<std::endl;
    }
    */
    /** TESTING_CODE::end */

    #if 0
    /** 202101131442::beginning */
    int window_id=(LFO::train_seq-1)/windowSize;
    int window_index=window_id*100000;
    uint64_t object_id=LFO::windowTrace[window_index].id;
    std::list<uint64_t> statistics_list = LFO::statistics[object_id];
    std::cerr<<"202101131442: "<<labels[window_index];
    for(uint64_t timestamp0 : statistics_list) 
        std::cerr<<'\t'<<timestamp0;
    std::cerr<<'\t'<<statistics_list.size();
    std::cerr<<'\t'<<object_id;
    std::cerr<<std::endl;
    /** 202101131442::end */
    #endif

    /** 202101211508::beginning */
    int window_id1=(LFO::train_seq-1)/windowSize;
    int window_index1=window_id1*100000;
    uint64_t object_id1=LFO::windowTrace[window_index1].id;
    int32_t indices_begin = indptr[window_index1]; 
    if(!
        (
            (indices[indices_begin]==(int32_t) 0) 
            #ifdef ISSUE20210206b
            ||
            (indices[indices_begin]==(int32_t)50)
            #endif
        )) {
        std::cerr
            <<"indices["<<indices_begin<<"]=="<<indices[indices_begin]
            <<" should==(int32_t)0";
        std::exit(EXIT_FAILURE);
    }
    std::cerr<<"202101211508:";
    int32_t iter0;
    for(
        iter0=0;
        indices[indices_begin+iter0]<HISTFEATURES;
        iter0++) 
        std::cerr<<'\t'<<data[indices_begin+iter0];
    if(!(indices[indices_begin+iter0]==HISTFEATURES)) {
        std::cerr<<"indices[indices_begin+iter0]==HISTFEATURES";
        std::exit(EXIT_FAILURE);
    }
    std::cerr<<'\t'<<iter0;
    std::cerr<<'\t'<<object_id1;
    std::cerr<<std::endl;
    /** 202101211508::end */

    LFO::trainModel(labels, indptr, indices, data);
    labels.clear();
    indptr.clear();
    indices.clear();
    data.clear();

    /** EVALUATING MODEL ACCURACY::beginning */
    uint64_t nr_prediction = (uint64_t)0;
    uint64_t nr_correct = (uint64_t)0;
    for(const trEntry entry0 : windowTrace) {
        #ifdef VERIFY_ACCURACY_CALCULATION
        if(entry0.decision_prediction != LFO::decision_array[nr_prediction]) {
            std::cerr
                <<"entry0.decision_prediction != LFO::decision_array["
                <<nr_prediction<<"]";
            std::exit(EXIT_FAILURE);
        }
        #endif
        nr_prediction++;
        if(entry0.decision_prediction == entry0.toCache) 
            nr_correct++;
    }
    if(nr_prediction != windowTrace.size()) {
        std::cerr<<"conclude_window(): nr_prediction should == window.size()"
            <<std::endl;
        std::exit(EXIT_FAILURE);
    }
    std::cerr<<"Window"<<LFO::train_seq/LFO::windowSize-1<<" accuracy= "
        << (float)nr_correct/nr_prediction << std::endl;

    /** EVALUATING MODEL ACCURACY::end */

    LFO::windowByteSum=(uint64_t)0; 
    LFO::statistics.clear();
    LFO::windowLastSeen.clear();
    LFO::windowOpt.clear();
    LFO::windowTrace.clear();

    /** TESTING_CODE::cnt_quartile::beginning */
    /** 
    std::cerr
        <<"LFO::cnt_quartile= "
        <<LFO::cnt_quartile0<<" "
        <<LFO::cnt_quartile1<<" "
        <<LFO::cnt_quartile2<<" "
        <<LFO::cnt_quartile3<<" "
        <<std::endl;

    LFO::cnt_quartile0
        =LFO::cnt_quartile1
        =LFO::cnt_quartile2
        =LFO::cnt_quartile3
        =(uint64_t)0;
    */
    /** TESTING_CODE::cnt_quartile::end */
}

void LFOCache::admit(SimpleRequest& req)
{
    /** TESTING_CODE::make sure admitted object not already cached::beginning
    */
    auto it0 = _cacheMap.left.find(
        std::make_pair((std::uint64_t)req.get_id(), (double)req.get_size())
        );
    if(it0 != _cacheMap.left.end()) {
        std::cerr<<"LFOCache::admit(): object already cached, LFO::train_seq= "
            <<LFO::train_seq
            <<", req ID= "
            << req.get_id() << ", size= " << req.get_size() << std::endl; 
        std::cerr<<"_lruCacheMap: "<<_lruCacheMap.size()<<" items: ";
        for(auto& it1: _lruCacheMap) 
            std::cerr<<it1.first<<" ";
        std::cerr<<std::endl;

        auto it1 = _lruCacheMap.find(req.get_id());
        if(it1 != _lruCacheMap.end()) {
            std::cerr
                <<"admit(): Object "<<req.get_id()
                <<" found in _lruCacheMap"<<std::endl;
        } else {
            std::cerr
                <<"admit(): Object "<<req.get_id()
                <<" not found in _lruCacheMap"<<std::endl;
        }
        std::exit(1);
    }
    /** TESTING_CODE::make sure admitted object not already cached::end*/

    const uint64_t size = req.get_size();
    // object feasible to store?
    if (size > _cacheSize) {
        LOG("L", _cacheSize, req._id, size);
        return;
    }

    if(LFO::init) {
        while(_currentSize + size > _cacheSize) {
            evict();
        }
        // admit new object
        uint64_t & obj = req._id;
        _lruCacheList.push_front(obj);
        _lruCacheMap[obj] = _lruCacheList.begin();
        auto it1 = _lruCacheMap.find(obj); 
        if(it1 == _lruCacheMap.end()) {
            std::cerr<<"admit(): Object "<<obj<<" cannot be found on LRU Cache"
                <<std::endl;
            std::exit(EXIT_FAILURE);
        }

        // update LFO _cacheMap 
        //  use .5 as rehit_probability because 
        //      1. object with <.5 rehit_probability should not be cached
        //      2. LFO not kick in yet to calculate rehit_probability 
        _cacheMap.insert(
            {
                {req.get_id(), req.get_size()}, 
                .5 // rehit_probability
            }
            );

        _currentSize += size;
        _size_map[obj] = size;
        LOG("a", _currentSize, obj.id, obj.size);

        return;
    }

    // not admit object if rehit_probability <.5
    double rehit_probability = LFO::calculate_rehit_probability(
        req, getSize()-getCurrentSize(), _objective
        );
    if(rehit_probability<(double).5) {
        LFO::windowTrace.end()->decision_prediction=false;
        #ifdef VERIFY_ACCURACY_CALCULATION
        LFO::decision_array[(LFO::train_seq-1)%LFO::windowSize]=false;
        #endif
        return;
    }
    LFO::windowTrace.end()->decision_prediction=true;
    #ifdef VERIFY_ACCURACY_CALCULATION
    LFO::decision_array[(LFO::train_seq-1)%LFO::windowSize]=true;
    #endif

    // check eviction needed
    while (_currentSize + size > _cacheSize) {
        evict();
    }
    // admit new object
    /**
    uint64_t & obj = req._id;
    _cacheList.push_front(obj);
    _cacheMap[obj] = _cacheList.begin();
    _currentSize += size;
    _size_map[obj] = size;
    LOG("a", _currentSize, obj.id, obj.size);
    */
    _cacheMap.insert(
        {
            {req.get_id(), req.get_size()}, 
            rehit_probability
        }
        );
    /** TESTING_CODE::assert that object admitted successfully::beginning */
    // This test is COMMENTABLE
    auto it1 = _cacheMap.left.find(
        std::make_pair((std::uint64_t)req.get_id(), (double)req.get_size())
        );
    if(it1 == _cacheMap.left.end()) {
        std::cerr<<"LFOCache::admit(): failed to cache admitted object"
            <<std::endl;
        std::exit(1);
    }
    if(fabs(it1->second - rehit_probability) > 0.001) {
        std::cerr<<"LFOCache::admit(): rehit_probability not stored correctly" 
            <<std::endl;
        std::exit(1);
    }
    /** TESTING_CODE::assert that object admitted successfully::end */
    _currentSize += size;
    // make sure that admitted object not already in unordered_map _size_map
    if(_size_map.find(req._id)!=_size_map.end()) {
        if(_size_map[req._id] == size) {
            // the only reason to call admit() to admit an object that is 
            //  already cached is because the object size has changed
            std::cerr<<"admit(): admitting an object already cached, "
                <<"req._id= "<<req._id<<", "
                <<"req._size= "<<req.get_size()
                <<std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
    _size_map[req._id] = size;
    LOG("a", _currentSize, req._id, req._size);
}

#if 0
// implementation incomplete
void LFOCache::evict(SimpleRequest& req)
{
    /**
    // TODO: remove LRU code
    uint64_t & obj = req._id;
    auto it = _cacheMap.find(obj);
    if (it != _cacheMap.end()) {
        ListIteratorType lit = it->second;
        LOG("e", _currentSize, obj.id, obj.size);
        auto & size = _size_map[obj];
        _currentSize -= size;
        _size_map.erase(obj);
        _cacheMap.erase(obj);
        _cacheList.erase(lit);
    }
    */
    auto it = _cacheMap.left.find(
        std::make_pair((std::uint64_t)req._id, (double)req._size)
        );
    if(it==_cacheMap.left.end()) {
        std::cerr
            <<"LFOCache::evict(SimpleRequest& req): evicted object not found"
            <<std::endl;
        std::exit(EXIT_FAILURE);
    }
    LOG("e", _currentSize, req._id, req._size); 
    auto & size = _size_map[obj];
    _currentSize -= size;
    _size_map.erase(obj);
}
#endif

KeyT LFOCache::evict()
{
    if(LFO::init) {
        // evict least popular (i.e. last element)
        if (_lruCacheList.size() > 0) {
            ListIteratorType lit = _lruCacheList.end();
            lit--;
            uint64_t obj = *lit;


    #ifdef EVICTION_LOGGING
            {
                auto it = future_timestamps.find(obj);
                unsigned int decision_qulity =
                        static_cast<double>(it->second - current_t) / (_cacheSize * 1e6 / byte_million_req);
                decision_qulity = min((unsigned int) 255, decision_qulity);
                eviction_qualities.emplace_back(decision_qulity);
                eviction_logic_timestamps.emplace_back(current_t / 65536);
            }
    #endif

            LOG("e", _currentSize, obj.id, obj.size);
            auto & size = _size_map[obj];
            _currentSize -= size;
            _size_map.erase(obj);
            _lruCacheMap.erase(obj);
            _lruCacheList.erase(lit);

            // synchronize LFO _cacheMap
            uint64_t _cacheMap_size_old = _cacheMap.size();
            auto it = _cacheMap.left.find(
                std::make_pair((std::uint64_t)obj, (double)size)
                );
            if(it == _cacheMap.left.end()) {
                std::cerr<<"evict()::init==true::object cannot be found "
                    <<"on _cacheMap: object ID= "<<obj<<", size= "<<size
                    <<std::endl;
                std::exit(EXIT_FAILURE);
            }
            _cacheMap.left.erase(it);
            if(_cacheMap.size() != _cacheMap_size_old-1) {
                std::cerr<<"evict()::init==true::object cannot be erased "
                    <<"from _cacheMap"<<std::endl;
                std::exit(EXIT_FAILURE);
            }
            if(_cacheMap.size() != _cacheMap.left.size()) {
                std::cerr<<"evict()::init==true::inconsistent map sizes::"
                    <<"_cacheMap.size() != _cacheMap.left.size()"
                    <<std::endl;
                std::exit(EXIT_FAILURE);
            }
            if(_cacheMap.size() != _cacheMap.right.size()) {
                std::cerr<<"evict()::init==true::inconsistent map sizes::"
                    <<"_cacheMap.size() != _cacheMap.right.size()"
                    <<std::endl;
                std::exit(EXIT_FAILURE);
            }

            return obj;
        }
        std::cerr<<"should not reach here: evicting from empty cache"
            <<std::endl;
        std::exit(1);
    } // if(LFO::init) 

    // lfoCacheMapType::right_map::const_iterator right_iter; 
    // lfoCacheMapType::right_map::const_iterator right_iter; 
    auto right_iter = _cacheMap.right.begin();
    KeyT evicted_req_id = right_iter->second.first;
    _currentSize -= right_iter->second.second;
    _size_map.erase(evicted_req_id);
    _cacheMap.right.erase(right_iter);

    /** TESTING_CODE::beginning */
    // make sure that the cached object with smallest rehit_probability 
    //  is >=.5 
    right_iter = _cacheMap.right.begin(); 
    if(right_iter->first < (double)0.5) {
        std::cerr<<"LFOCache::evict(): rehit_probability must >= .5"
            <<std::endl; 
        std::exit(1);
    }

    return evicted_req_id;
}

/** 
void LFOCache::hit(lfoCacheMapType::left_map::const_iterator it, 
    double rehit_probability)
{
    #if 0
    // LRU code
    _cacheList.splice(_cacheList.begin(), _cacheList, it->second);
    #endif
}
*/

#if 0
// TODO: to remove this function
const SimpleRequest & LFOCache::evict_return()
{
    // evict least popular (i.e. last element)
    ListIteratorType lit = _cacheList.end();
    lit--;
    uint64_t obj = *lit;
    auto size = _size_map[obj];
    LOG("e", _currentSize, obj, size);
    SimpleRequest req(obj, size);
    _currentSize -= size;
    _size_map.erase(obj);
    _cacheMap.erase(obj);
    _cacheList.erase(lit);
    return req;
}
#endif

/** 
bool LFOCache::exist(const KeyT &key) {
    return _cacheMap.find(key) != _cacheMap.end();
}
*/

void LFO::annotate(uint64_t seq, uint64_t id, uint64_t size, double cost) {
    /** TESTING_CODE::end */
    /**
    if(!(seq % LFO::windowSize)) {
        std::cerr<<"End Window"<<(seq/LFO::windowSize-1)<<std::endl; 
    }
    */
    /** TESTING_CODE::end */

    const uint64_t idx= (seq-1) % LFO::windowSize; 
    // store access timestamps
    if(LFO::statistics.count(id)==1) {
        std::list<uint64_t>& list0=LFO::statistics[id];
        list0.push_front(idx);
        if(list0.size() > HISTFEATURES) 
            list0.pop_back();
    } else if(LFO::statistics.count(id)==0) {
        // first time this object is accessed in this sliding window 
        std::list<uint64_t> list0;
        list0.push_front(idx);
        LFO::statistics[id]=list0;
    } else {
        std::cerr
            <<"LFO::annotate(): invalid LFO::statistics.count(" 
            <<id<<")= "<<LFO::statistics.count(id)<<std::endl;
        std::exit(EXIT_FAILURE);
    }
    const auto idsize = std::make_pair(id, size); 
    bool hasLastSeen=false;
    uint64_t lastSeenIndex=std::numeric_limits<uint64_t>::max(); 
    int lastSeenCount=(int)LFO::windowLastSeen.count(idsize); 
    if(lastSeenCount!=0 && lastSeenCount!=1) {
        std::cerr<<"lastSeenCount= "<<lastSeenCount<<std::endl;
        std::exit(EXIT_FAILURE);
    }
    if(LFO::windowLastSeen.count(idsize)>0) {
        hasLastSeen=true;
        lastSeenIndex=(uint64_t)LFO::windowLastSeen[idsize];
        if(LFO::windowOpt[lastSeenIndex].id!=id) {
            std::cerr<<"id unmatch: "<<seq<<" "<<id<<" "<<size<<" "
                <<"lastSeen: "
                <<LFO::windowOpt[lastSeenIndex].seq<<" "
                <<LFO::windowOpt[lastSeenIndex].id<<" "
                <<LFO::windowOpt[lastSeenIndex].size;
            std::cerr<<"windowOpt size= "<<windowOpt.size()<<std::endl;
            for(auto &it: LFO::windowOpt) 
                std::cerr
                    <<it.seq<<" "<<it.idx<<" "<<it.id<<" "<<it.size
                    <<" "<<it.hasNext<<" "<<it.volume<<std::endl;
            std::cerr<<"windowTrace size= "<<windowTrace.size()<<std::endl;
            for(auto &it: LFO::windowTrace) 
                std::cerr
                    <<it.seq<<" "<<it.id<<" "<<it.size<<" "<<it.toCache<<" "
                    <<it.lastSeenIndex<<" "<<it.lastSeenCount<<std::endl;
            std::exit(EXIT_FAILURE);
        }
        if(LFO::windowOpt[lastSeenIndex].size!=size) {
            std::cerr<<"size unmatch: "<<seq<<" "<<id<<" "<<size<<" "
                <<"lastSeen: "
                <<LFO::windowOpt[lastSeenIndex].seq<<" "
                <<LFO::windowOpt[lastSeenIndex].id<<" "
                <<LFO::windowOpt[lastSeenIndex].size;
            std::exit(EXIT_FAILURE);
        }
        LFO::windowOpt[LFO::windowLastSeen[idsize]].hasNext = true;
        LFO::windowOpt[LFO::windowLastSeen[idsize]].volume = (idx - 
           LFO::windowLastSeen[idsize]) * size;
    }
    LFO::windowByteSum += size;
    LFO::windowLastSeen[idsize]=idx;
    LFO::windowOpt.emplace_back(idx, seq, id, size); 
    if(!hasLastSeen)
        LFO::windowTrace.emplace_back(seq, id, size, cost, lastSeenCount);
    else {
        if(LFO::windowOpt[lastSeenIndex].hasNext == false) {
            std::cerr<<"LFO::windowOpt[lastSeenIndex].hasNext == false";
            std::exit(EXIT_FAILURE);
        }
        LFO::windowTrace.emplace_back(seq, id, size, cost, lastSeenIndex, 
            lastSeenCount);
    }

    return; 
}

double LFO::calculate_rehit_probability(
        SimpleRequest& req, 
        uint64_t cacheAvailBytes, 
        int optimization_objective // 0 for OHR, 1 for BHR
        ) {
        //LFO
        /** predicting rehit probability::beginning */
        /** obtain shift time-invariant time gaps */
        std::vector<int32_t> indptr; 
        std::vector<int32_t> indices; 
        std::vector<double>  data; 
        std::vector<double>  result;
        int64_t              len_result;
        
        if(LFO::statistics.count(req._id) == 0) {
            std:cerr
                <<"LFO::calculate_rehit_probability()::train_seq= "
                <<LFO::train_seq<<": "
                <<"access timestamp of Object " << req._id 
                << " not exist in LFO::statisics" << std::endl;
            std::exit(EXIT_FAILURE);
        }
        // curQueue stores the access timestamps in descending order 
        //  i.e., latest first
        //  The timstamps are offset to the beginning of each sliding window 
        //  See deriveFeatures() 
        std::list<uint64_t>& curQueue = LFO::statistics[req._id];
        int32_t idx = 0; 
        uint64_t lastReqTime = (LFO::train_seq - 1)%LFO::windowSize; 
        /** TESTING_CODE::beginning */
        if(!(lastReqTime == *curQueue.begin())) {
            std::cerr<<"calculate_rehit_probability(): ";
            std::cerr<<"lastReqTime should== *curQueue.begin()";
            std::cerr<<std::endl;
            std::exit(EXIT_FAILURE);
        }
        /** TESTING_CODE::end */
        for(auto &lit: curQueue) {
            const uint64_t dist = lastReqTime - lit; 
            /**
            // 20210208::beginning
            */
            if(lastReqTime == lit) {
                if(!(dist == 0)) {
                    std::cerr<<"calculate_rehit_probability(): ";
                    std::cerr<<"dist should== 0";
                    std::exit(EXIT_FAILURE);
                }
                continue;
            }
            if(!(lastReqTime > lit)) {
                std::cerr<<"calculate_rehit_probability(): "; 
                std::cerr<<"lastReqTime should> lit";
                std::exit(EXIT_FAILURE);
            }
            /**
            // 20210208::end
            */
            indices.push_back(idx); 
            data.push_back(dist);
            idx++;
            lastReqTime = lit;
        }
        /** 202102141439::beginning */
        #if 0
        /** no longer needed since switching to standard LGBM */
        if(idx==0) {
            // the object is accessed for its 1st time in this window 
            indices.push_back((int32_t)idx);
            data.push_back((double)0);
            idx++;
        }
        #endif
        /** 202102141439::end */
        // object size
        indices.push_back(HISTFEATURES); 
        data.push_back(std::round(100*std::log2(req._size))); 
        // available cache space
        indices.push_back(HISTFEATURES+1); 
        double currentSize = cacheAvailBytes <= 0 ? 0 : 
            std::round(100*std::log2(cacheAvailBytes));
        //data.push_back(std::round(100*std::log2(currentSize))); 
        data.push_back(currentSize);
        // cost 
        indices.push_back(HISTFEATURES+2);
        /** TESTING_CODE::beginning */
        if((optimization_objective != LFO::OHR) && 
            (optimization_objective != LFO::BHR)) {
            std::cerr<<"Invalid LFOCache::_objective " 
                << optimization_objective << std::endl; 
            std::exit(EXIT_FAILURE);
        }
        /** 
        if((LFO::objective == LFO::OHR) && (req._cost!=1)) {
            std::cerr<<"Invalid req cost for OHR: " << req._cost << std::endl;
            std::exit(EXIT_FAILURE);
        }
        if((LFO::objective == LFO::BHR) && (req._cost!=req._size)) {
            std::cerr<<"Invalid req cost for BHR: " << req._cost << std::endl;
            std::exit(EXIT_FAILURE);
        }
        */
        /** TESTING_CODE::end */
        if(optimization_objective == LFO::OHR) 
            data.push_back((double)1); 
        else if(optimization_objective == LFO::BHR) 
            data.push_back((double)req._size); 
        else {
            std::cerr<<"Invalid LFOCache::_objective " 
                << optimization_objective << std::endl; 
            std::exit(EXIT_FAILURE);
        }

        indptr.push_back(0);
        indptr.push_back(indptr[indptr.size()-1] + idx + NR_NON_TIMEGAP_ELMNT);

        /** rehit prediction */
        /** TESTING_CODE::beginning */
        /**
        std::cerr<<"PREDICTION "
            << "id= " << req._id << " "
            << "nr_acesses= " << LFO::statistics[req._id].size() << " " 
            << indptr.size() << " " 
            << indices.size() << " " 
            << data.size() << " " 
            << std::endl;
        */
        /** TESTING_CODE::end */

        /** TESTING_CODE::beginning */
        /** 
        int64_t out_len; 
        char out_str[1024*1024]; 
        int return_BDM = LGBM_BoosterDumpModel(
            LFO::booster, 
            0, 
            -1, 
            C_API_FEATURE_IMPORTANCE_SPLIT
            (int64_t) 1024*1024,
            &out_len, 
            out_str
            );
        if(return_BDM==0) {
            std::cerr<<"out_len= "<<out_len<<std::endl;
            std::cerr<<"out_str= "<<out_str<<std::endl;
            std::exit(2);
        } else {
            std::cerr<<"DumpModel() error;"<<std::endl;
            std::exit(EXIT_FAILURE);
        }
        */
        /** TESTING_CODE::end */
        /** TESTING_CODE-20200814a::beginning */
        #if 0
        indptr.clear(); 
        indices.clear(); 
        data.clear(); 

        indptr.push_back(0);
        indptr.push_back(53);
        for(int i20200807=0; i20200807<53; i20200807++) { 
            indices.push_back(i20200807);
        }
        for(int i20200807=0; i20200807<50; i20200807++) { 
            if(i20200807%2) 
                data.push_back((double)3);
            else
                data.push_back((double)5);
        }
        // add object size
        data[0]=(double)-7;
        data.push_back(std::round(100*std::log2(req._size)));
        // available cache space
        uint64_t cacheAvailBytes = getSize()-getCurrentSize();
        double currentSize = cacheAvailBytes <= 0 ? 0 :
            std::round(100*std::log2(cacheAvailBytes));
        data.push_back(std::round(100*std::log2(currentSize)));
        // cost
        if(_objective==LFO::OHR) 
            data.push_back((double)1);
        else if(_objective==LFO::BHR)
            data.push_back((double)req._size); 
        else {
            std::cerr<<"Invalid LFOCache::_objective " << _objective 
                <<std::endl; 
            std::exit(EXIT_FAILURE);
        }
        #endif
        /** TESTING_CODE-20200814a::end */

        /** TESTING_CODE::beginning */
        #if 0
        std::cerr
        << "before LGBM_BoosterPredictForCSR(): "
        << indptr.size() << " " 
        << indices.size() << " " 
        << data.size() 
        << std::endl; 
        std::cerr<<"indptr: ";
        for(int iter0=0; iter0<indptr.size(); iter0++) 
            std::cerr<<indptr[iter0];
        std::cerr<<std::endl;
        std::cerr<<"indices: ";
        for(int iter0=0; iter0<4; iter0++) 
            std::cerr<<indices[iter0]<<" ";
        std::cerr<<std::endl;
        for(int iter0=indices.size()-5; iter0<indices.size(); iter0++) 
            std::cerr<<indices[iter0]<<" ";
        std::cerr<<std::endl;
        std::cerr<<"data: ";
        for(int iter0=0; iter0<4; iter0++) 
            std::cerr<<data[iter0]<<" ";
        std::cerr<<std::endl;
        for(int iter0=data.size()-5; iter0<data.size(); iter0++) 
            std::cerr<<data[iter0]<<" ";
        std::cerr<<std::endl;
        #endif
        /** TESTING_CODE::end */
        // so that result.data() does not return nullptr 
        result.reserve((size_t)4); 
        if((
            (LFO::train_seq-1)%(LFO::windowSize/10)
            )==0) {
            std::cerr
                <<"Object ID= "      << req._id        << ", "
                <<"Object Size= "    << req._size      << ", "
                <<"indptr.size()= "  << indptr.size()  << ", "
                <<"indices.size()= " << indices.size() << ", "
                <<"curQueue.size()= " << curQueue.size() << ", "
                <<"data.size()= "    << data.size()
                <<std::endl;
            std::cerr<<"indptr= ";
            for(int i=0; i<indptr.size(); i++) 
                std::cerr<<indptr[i]<<" ";
            std::cerr<<std::endl;
            std::cerr<<"indices= ";
            for(int i=0; i<indices.size(); i++) 
                std::cerr<<indices[i]<<" ";
            std::cerr<<std::endl;
            std::cerr<<"curQueue= ";
            for(auto &lit: curQueue) 
                std::cerr<<lit<<" ";
            std::cerr<<std::endl;
            std::cerr<<"data= ";
            for(int i=0; i<data.size(); i++) 
                std::cerr<<data[i]<<" ";
            std::cerr<<std::endl;
        } 

	//
	// Added by David on 2021-05-10
	//
	
	std::string params = param_map_to_string(LFO::trainParams);


        int return_LGBM_BPFC= LGBM_BoosterPredictForCSR(
                LFO::booster, 
                static_cast<void *>(indptr.data()), 
                C_API_DTYPE_INT32, 
                indices.data(),
                static_cast<void *>(data.data()), 
                C_API_DTYPE_FLOAT64,
                indptr.size(), 
                data.size(), 
                HISTFEATURES+NR_NON_TIMEGAP_ELMNT, /** num_col */
                C_API_PREDICT_NORMAL, 
                //since only 1 row HISTFEATURES + NR_NON_TIMEGAP_ELMNT,
		        0, /** start_iteration */
                0, /** num_iteration */
                params.c_str(), 
                &len_result, 
                result.data()
                );
        if(return_LGBM_BPFC == 0) { 
            // predication succeeded 
            /** TESTING_CODE:: beginning */
            /** 
            std::cerr
                << "after LGBM_BoosterPredictForCSR(): "
                << len_result << " "
                << result.size() 
                << std::endl;
            */
            /** TESTING_CODE::cnt_quartile::beginning */
            #if 0
            if(result[0]<(double)0.0) {
                std::cerr<<"result[0]= "<<result[0]<<"<0"<<std::endl;
                std::exit(EXIT_FAILURE);
            } else if(result[0]<(double).25) {
                LFO::cnt_quartile0++;
            } else if(result[0]<(double).50) {
                LFO::cnt_quartile1++;
            } else if(result[0]<(double).75) {
                LFO::cnt_quartile2++;
            } else if(result[0]<=(double)1.0) {
                LFO::cnt_quartile3++;
            } else {
                std::cerr<<"result[0]= "<<result[0]<<" out of range"<<std::endl;
                std::exit(EXIT_FAILURE);
            }
            #endif
            /** TESTING_CODE::cnt_quartile::end */
    
            /** TESTING_CODE::end */
        } else if (return_LGBM_BPFC == -1) {
            //return value not 0
            std::cerr<<"prediction failed"<<std::endl; 
            std::exit(EXIT_FAILURE);
        } else {
            std::cerr<<"invalid LGBM_BoosterPredictForCSR return value: " 
                << return_LGBM_BPFC << std::endl; 
            std::exit(EXIT_FAILURE);
        }
        /** 
        if(req._id==4294793459) {
            std::cerr
                <<"Object ID= "      << req._id        << ", "
                <<"Object Size= "    << req._size      << ", "
                <<"indptr.size()= "  << indptr.size()  << ", "
                <<"indices.size()= " << indices.size() << ", "
                <<"data.size()= "    << data.size()
                <<std::endl;
            std::cerr<<"indptr= ";
            for(int i=0; i<indptr.size(); i++) 
                std::cerr<<indptr[i]<<" ";
            std::cerr<<std::endl;
            std::cerr<<"indices= ";
            for(int i=0; i<indices.size(); i++) 
                std::cerr<<indices[i]<<" ";
            std::cerr<<std::endl;
            std::cerr<<"data= ";
            for(int i=0; i<data.size(); i++) 
                std::cerr<<data[i]<<" ";
            std::cerr<<std::endl;

            std::exit(EXIT_SUCCESS);
        }
        */
        #if 0
        std::cerr<<"len_result= "<<len_result<< ", "<<result.data()
            <<", result[0]= "<<result[0] 
            /** 
            Normal for result.size()=0 because the result is not added using 
            any function in std::vector
            Note: datatype of result is std::vector
            */
            <<", result.size()= "<<result.size()
            <<", result.capacity()= "<<result.capacity()
            <<std::endl;
        // std::exit(2);
        #endif

        indptr.clear(); 
        indices.clear(); 
        data.clear(); 

        double rehit_probability = result[0]; 
        result.clear(); 
        /** predicting rehit probability::end */
        return rehit_probability; 
}

void LFO::calculateOPT(uint64_t cacheSize, int optimization_objective) {
    /**
    Note: cacheSize=physical cache size - size of metadata
    i.e., size of the cache used to store objects 

    */ 

    if(optimization_objective == LFO::OHR) {
        sort(LFO::windowOpt.begin(), LFO::windowOpt.end(), 
            [](const struct optEntry &lhs, const struct optEntry &rhs) {
                return lhs.volume < rhs.volume;
            });
    } else if(optimization_objective == LFO::BHR) {
        sort(LFO::windowOpt.begin(), LFO::windowOpt.end(), 
            [](const struct optEntry &lhs, const struct optEntry &rhs) {
                return (lhs.volume/lhs.size) < (rhs.volume/rhs.size);
            });
    } else {
        std::cerr<<"invalid optimization_objective= " << optimization_objective 
            << std::endl; 
        std::exit(1);
    }

    uint64_t cacheVolume = cacheSize * LFO::windowSize;
    uint64_t currentVolume = 0;
    uint64_t hitc = 0;
    uint64_t bytehitc = 0;
    for (auto &it: LFO::windowOpt) {
        if (currentVolume > cacheVolume) {
            break;
        }
        if (it.hasNext) {
            LFO::windowTrace[it.idx].toCache = true;
            hitc++;
            bytehitc += LFO::windowTrace[it.idx].size;
            currentVolume += it.volume;
        }
    }
    /** TESTING_CODE::beginning */
    /**
    */
    std::cerr<<"LFO::calculateOPT: cacheSize = " <<cacheSize
        <<", hitc = "<< hitc <<", bytehitc = "<< bytehitc <<std::endl; 
    /** TESTING_CODE::end */
}

void LFO::deriveFeatures(std::vector<float> &labels, 
    std::vector<int32_t> &indptr, std::vector<int32_t> &indices, 
    std::vector<double> &data, int sampling, uint64_t cacheSize) {
    int64_t cacheAvailBytes = (int64_t)cacheSize;
    /** TESTING_CODE::beginning */
    /**
    */
    std::cerr << "LFO::deriveFeatures(), cacheAvailBytes= " << cacheAvailBytes
        << ", sampling = " << sampling
        << ", LFO::sampleSize = "<< LFO::sampleSize 
        << ", HISTFEATURES = " << HISTFEATURES 
        << ", sampling = " << sampling 
        << ", sampleSize = " << sampleSize 
        <<std::endl; 
    /** TESTING_CODE::end */

    // unordered_map<object ID, object size> cache 
    std::unordered_map<uint64_t, uint64_t> cache;

    uint64_t negCacheSize = (uint64_t)0;
    // -ve cache size below 0 by how much
    uint64_t negCacheSizeMax = (uint64_t)0;

    uint64_t iter0 = (uint64_t)0;
    indptr.push_back(0);

    // nmbr of samples taken during each window
    uint64_t flagCnt=(uint64_t)0;
    /** TESTING_CODE::beginning */
    /**
    */
    uint64_t expected_indices_size=(uint64_t)0;
    /** TESTING_CODE::end */

    /** 
        unordered_map latest_decision stores 
        the latest caching decision of each object as traversing 
        windowTrace. 
        <std::uint64_t ObjectID, bool decision> 
    */
    std::unordered_map<std::uint64_t, bool> latest_decision; 
    /**
        for calculating remaining cache space at each request in windowTrace
    */
    uint64_t remaining_cacheSize = cacheSize;
    for(auto &it: windowTrace) {
        auto &curQueue = statistics[it.id];
        const auto curQueueLen = curQueue.size();
        if(curQueueLen > HISTFEATURES) {
            if(!(curQueueLen==HISTFEATURES+1)) {
                std::cerr<<"curQueueLen should== HISTFEATURES+1";
                std::exit(-1);
            }
            curQueue.pop_back();
        }

        bool flag = true;
        if(sampling == 1) {
            // YK: iter0 incremented in each for-loop iteration
            // sampleSize is an element in map<string, string> params, an 
            //  argument of _simulation_lfo(). As on 2020071, since 
            // _simulation_lfo() not invoked in the code provided, I do not know
            // the value of sampleSize 
            flag = iter0 >= (windowSize - LFO::sampleSize);
        } else if (sampling == 2) {
            // uniform_real_distribution<> dis(0.0, 1.0);
            double rand = LFO::dis(LFO::gen);
            /** TESTING_CODE::beginning */
            if(rand < 0.0) {
                std::cerr<<"invalid: rand= "<<rand<<std::endl;
                std::exit(EXIT_FAILURE);
            } else if(rand > 1.0) {
                std::cerr<<"invalid: rand= "<<rand<<std::endl;
                std::exit(EXIT_FAILURE);
            }
            /** TESTING_CODE::end */
            flag = rand < (double) sampleSize / windowSize;
        } else {
            std::cerr<<"Invalid sampling type: "<< sampling << std::endl;
            std::exit(EXIT_FAILURE);
        }

        if (flag) {
            flagCnt++;
            // std::vector::push_back: add element to the end 
            labels.push_back(it.toCache ? 1 : 0);

            // derive features
            int32_t idx = 0;
            uint64_t lastReqTime = iter0;

            /** TESTING_CODE::INDICES_GROWTH::beginning */
            uint64_t indices_size_old = (uint64_t)indices.size();
            /** TESTING_CODE::INDICES_GROWTH::end */

            //YK: curQueue: feature list of an object 

            /** 
            // indices[0]=data[0]=0
            indices.push_back(idx); 
            idx++;
            data.push_back(0);
            */
            for (auto &lit: curQueue) { 
                if(!(lastReqTime > lit)) break;
                const uint64_t dist = lastReqTime - lit; // distance
                // YK: std::vector::push_back(): append to the end 
                indices.push_back(idx);
                data.push_back(dist);

                /** TESTING_CODE::beginning */
                if(idx==0) {
                    // no TESTING_CODE block is not necessary. 
                    //  RSN: idx==0 is noise anyway 
                } else {
                    if(!(dist>=(uint64_t)0)) {
                        // because of the curQueue.push_front(i++) statement 
                        //  at end of the outer for loop, dist=0 is OK for 
                        //  an object accessed by 2 consecutive requests. 
                        std::cerr
                            <<"idx= "<<idx<<": dist= "<<dist<<"<=0"
                            <<", lastReqTime= "<<lastReqTime
                            <<", lit= "<<lit
                            <<std::endl;
                        std::cerr<<"curQueue: ";
                        for(uint64_t elmnt: curQueue) 
                            std::cerr<<elmnt<<" ";
                        std::cerr<<std::endl;
                        std::exit(EXIT_FAILURE);
                    }
                }
                /** TESTING_CODE::end */
                idx++;
                lastReqTime = lit;
            }
            /** TESTING_CODE::INDICES_GROWTH::beginning */
            uint64_t indices_growth = indices.size()-indices_size_old;
            if(indices_growth!=(uint64_t)idx) {
                std::cerr<<"TESTING_CODE::INDICES_GROWTH failed"<<std::endl;
                std::exit(EXIT_FAILURE);
            }
            if(!(indices_growth<=HISTFEATURES)) {
                std::cerr<<"indices_growth should <= HISTFEATURES"<<std::endl;
                std::cerr<<"indices_growth= "<<indices_growth<<std::endl;
                std::exit(EXIT_FAILURE);
            }
            /** TESTING_CODE::INDICES_GROWTH::end */

            /** 202102141439::beginning */
            if(idx==0) {
                indices.push_back((int32_t)0);
                data.push_back((double)0);
                idx++;
            }
            /** 202102141439::end */

            // object size
            indices.push_back(HISTFEATURES);
            data.push_back(round(100.0 * log2(it.size)));

            // remaining cache space 
            indices.push_back(HISTFEATURES + 1);
            data.push_back(round(100.0 * log2(remaining_cacheSize)));
            if(it.toCache) {
                if(latest_decision.find(it.id)==latest_decision.end()) {
                    // 1st encounter of the object on windowTrace
                    remaining_cacheSize -= it.size;
                } else {
                    if(latest_decision[it.id]==false) {
                        // OPT decided to not cache object in latest encounter 
                        remaining_cacheSize -= it.size;
                    } else if(latest_decision[it.id]==true) {
                        // OPT decided to cache object in latest encounter 
                        //  OPT decides to continue caching the object
                        //remaining_cacheSize unchanged
                    } else {
                        std::cerr<<"undefine latest_decision[], it.id= "
                            <<it.id<<", latest_decision[]= "
                            <<latest_decision[it.id]<<std::endl;
                        std::exit(1);
                    }
                }
            } else if (!it.toCache) {
                if(latest_decision.find(it.id)==latest_decision.end()) {
                    // 1st encounter of the object on windowTrace
                    // remaining_cacheSize unchanged 
                } else {
                    if(latest_decision[it.id]==false) {
                        // OPT decided to not cache object in latest encounter 
                        // remaining_cacheSize unchanged
                    } else if(latest_decision[it.id]==true) {
                        // OPT decided to cache object in latest encounter 
                        //  OPT decides to not cache the object
                        remaining_cacheSize += it.size; 
                    } else {
                        std::cerr<<"undefine latest_decision[], it.id= "
                            <<it.id<<", latest_decision[]= "
                            <<latest_decision[it.id]<<std::endl;
                        std::exit(1);
                    }
                }
            } else {
                std::cerr<<"no defined value for it.toCache"<<std::endl;
                std::exit(1);
            } // if(it.toCache) 
            latest_decision[it.id]=it.toCache; 
            // cost
            indices.push_back(HISTFEATURES + 2);
            data.push_back(it.cost);

            /**
            from the code I was provided
            indptr.push_back(indptr[indptr.size() - 1] + idx + 2);
            */
            // For each element in windowTrace, following are appended in 
            // std::vector indices
            //  * upto HISTFEATURES of elements from 0, 1, ... 
            //  * new element of value 50 corresponding to object size in 
            //      vector data
            //  * new element of value 51 corresponding to remaining cache size 
            //      in vector data
            //  * new element of value 50 corresponding to object size in 
            //      vector data
            //  that is why NR_NON_TIMEGAP_ELMNT=3
            /** TESTING_CODE::verifying indices.size()::beginning */
            if(NR_NON_TIMEGAP_ELMNT!=3) {
                std::cerr<<"NR_NON_TIMEGAP_ELMNT!=3"<<std::endl;
                std::exit(EXIT_FAILURE);
            }

            /** 
            */
            uint64_t indices_size_expected = (uint64_t)0;
            indices_size_expected += indices_size_old;
            indices_size_expected += (uint64_t)idx;
            indices_size_expected += (uint64_t)NR_NON_TIMEGAP_ELMNT;
            if(indices.size() != indices_size_expected) {
                std::cerr
                    <<"indices.size() != indices_size_expected, "
                    <<"indices.size() = " << indices.size() << ", "
                    <<"indices_size_expected = " << indices_size_expected
                    <<std::endl;
                std::exit(EXIT_FAILURE);
            }
            /**
            */
            expected_indices_size+= (uint64_t)idx;
            expected_indices_size+= (uint64_t)NR_NON_TIMEGAP_ELMNT;
            /** TESTING_CODE::verifying indices.size()::end */
            indptr.push_back(indptr[indptr.size() - 1] + idx 
                + NR_NON_TIMEGAP_ELMNT);
        } // if (flag) 

        // update cache size
        if (cache.count(it.id) == 0) {
            // we have never seen this id
            if (it.toCache) {
                cacheAvailBytes -= it.size;
                cache[it.id] = it.size;
            }
        } else {
            // repeated request to this id
            if (!it.toCache) {
                // used to be cached, but not any more
                cacheAvailBytes += cache[it.id];
                cache.erase(it.id);
            }
        }

        if (cacheAvailBytes < 0) {
            negCacheSize++; // that's bad
            negCacheSizeMax = (-1 * cacheAvailBytes) > negCacheSizeMax? 
                -1 * cacheAvailBytes : negCacheSizeMax; 
        }

        // update queue
        curQueue.push_front(iter0++);
    } //for(auto &it: windowTrace) 

    /** TESTING_CODE::beginning */
    /**
    */
    if(indices.size() != expected_indices_size) {
        std::cerr
            <<"indices.size() != expected_indices_size, "
            <<"indices.size() = " << indices.size() << ", "
            <<"expected_indices_size = " << expected_indices_size
            <<std::endl;
        std::exit(EXIT_FAILURE);
    }
    std::cerr<<"deriveFeatures(): expected_indices_size = " 
        << expected_indices_size << std::endl;
    /** TESTING_CODE::end */

    /** TESTING_CODE::beginning */
    /**
    */
    // -ve cache size is undesirable. See "LFO Implementation Note"
    if (negCacheSize > 0) {
        std::cerr 
            << "deriveFeatures(): Negative cache size event: " << negCacheSize 
            << ", " 
            << "negCacheSizeMax= " << negCacheSizeMax
            << ", " 
            << "flagCnt= " << flagCnt
            << std::endl;
    }
    /** TESTING_CODE::end */

    std::cerr<<"LFO::deriveFeatures() summary"<<std::endl; 
    std::cerr<<"* windowTrace.size()= "<<windowTrace.size()<<std::endl; 
    std::cerr<<"* labels.size()= "<<labels.size()<<std::endl; 
    std::cerr<<"* indptr.size()= "<<indptr.size()<<std::endl;
    std::cerr<<"* indices.size()= "<<indices.size()<<std::endl;
    std::cerr<<"* data.size()= "<<data.size()<<std::endl;

    #if 0
    /** 202103030909::beginning */
    int window_id0=(LFO::train_seq-1)/windowSize;
    int sampled_windowTrace_index0=window_id0*(windowSize/10);
    std::cerr<<"202103041315:";
    for(
        uint64_t index0=indptr[sampled_windowTrace_index0]; 
        index0<indptr[sampled_windowTrace_index0+1]; 
        index0++) { 
        std::cerr<<'\t'<<data[index0];
    }
    std::cerr<<'\n';
    /** 202103030909::end */
    #endif

}

void LFO::trainModel(vector<float> &labels, vector<int32_t> &indptr, 
    vector<int32_t> &indices, vector<double> &data) {

    DatasetHandle trainData;
    /** TESTING_CODE::beginning */
    /**
    std::cerr
        << "size: "
        << "labels "  << labels.size()  << " "
        << "indptr "  << indptr.size()  << " "
        << "indices " << indices.size() << " "
        << "data "    << data.size()    << " "
        << std::endl;
    */
    /** TESTING_CODE::end */

    /** EXPORTING TRAINING DATA::beginning */
    std::stringstream stringstream0; 
    stringstream0<<"training_data."<<LFO::timestamp
        <<".window"<<(LFO::train_seq/LFO::windowSize)-1
        <<".txt";
    std::string training_data_filename;
    stringstream0>>training_data_filename;
    std::ofstream training_data_ofstream;
    training_data_ofstream.open(training_data_filename, std::ios_base::app);
    if(!(indptr.size() == labels.size()+1)) {
        std::cerr<<"indptr.size() should == labels.size()+1";
        std::exit(EXIT_FAILURE);
    }
    #if 0
    int loop_index_level0=0;
    for(std::int32_t indptr0: indptr) {
        if(indptr0==indptr.back()) break;
        std::cerr<<"loop_index_level= "<<loop_index_level0++<<" ";
        int loop_index_level1=0;
        for(float label: labels) {
            std::cerr<<loop_index_level1++<<" ";
            training_data_ofstream<<label;
            std::int32_t index0;
            for(index0=indptr0; indices[index0]<HISTFEATURES; index0++) {
                std::cerr<<index0<<" ";
                training_data_ofstream<<'\t'<<data[index0];
            }
            for(int j=0; j<NR_NON_TIMEGAP_ELMNT; j++) {
                if(!(indices[index0]>=HISTFEATURES)) {
                    std::cerr<<"indices[index0]>=HISTFEATURES";
                    std::exit(EXIT_FAILURE);
                }
                std::cerr<<index0+j<<" ";
                training_data_ofstream<<'\t'<<data[index0+j];
            }
            training_data_ofstream<<std::endl;
        }
        std::cerr<<std::endl;
    }
    #endif
    size_t window_index = (size_t)0;
    for(float label: labels) {
        training_data_ofstream<<label;
        size_t iter0;
        for(iter0=indptr[window_index]; indices[iter0]<HISTFEATURES; iter0++) {
            training_data_ofstream<<'\t'<<data[iter0];
        }
        if(!(indices[iter0]==HISTFEATURES)) {
            std::cerr<<"indices[iter0] should ==HISTFEATURES";
            std::exit(EXIT_FAILURE);
        }
        for(; iter0<indptr[window_index]+HISTFEATURES; iter0++) {
            training_data_ofstream<<'\t'<<"NA";
        }
        if(!(iter0==indptr[window_index]+HISTFEATURES)) {
            std::cerr<<"iter0 should== indptr[window_index]+HISTFEATURES";
            std::exit(EXIT_FAILURE);
        }
        if(!(indices[indptr[window_index+1]-NR_NON_TIMEGAP_ELMNT]==HISTFEATURES
            )) {
            std::cerr<<"indices[indptr[window_index+1]-NR_NON_TIMEGAP_ELMNT] ";
            std::cerr<<"should==HISTFEATURES";
            std::exit(EXIT_FAILURE);
        }
        training_data_ofstream<<'\t'
            <<data[indptr[window_index+1]-NR_NON_TIMEGAP_ELMNT];
        training_data_ofstream<<'\t'
            <<data[indptr[window_index+1]-NR_NON_TIMEGAP_ELMNT+1];
        training_data_ofstream<<'\t'
            <<data[indptr[window_index+1]-NR_NON_TIMEGAP_ELMNT+2];
        training_data_ofstream<<std::endl;
        window_index++;
    }
    if(!(window_index==LFO::windowSize)) {
        std::cerr<<"window_index should==LFO::windowSize";
        std::exit(EXIT_FAILURE);
    }
    training_data_ofstream.close();
    /** EXPORTING TRAINING DATA::end */
    std::string params = param_map_to_string(LFO::trainParams);
    LGBM_DatasetCreateFromCSR(
        static_cast<void *>(indptr.data()), C_API_DTYPE_INT32, 
        indices.data(), 
        static_cast<void *>(data.data()), C_API_DTYPE_FLOAT64, 
        indptr.size(), 
        data.size(), 
        HISTFEATURES + 3, 
        params.c_str(), nullptr, &trainData);
    /** TESTING_CODE::Dumping DatasetHandle trainData::beginning */
    int num_data; // no. of data points
    LGBM_DatasetGetNumData(trainData, &num_data); 
    std::cerr<<"LFO::trainModel(): num_data= "<<num_data<<std::endl;
    int num_feature; // no. of features
    LGBM_DatasetGetNumFeature(trainData, &num_feature); 
    std::cerr<<"LFO::trainModel(): num_feature= "<<num_feature<<std::endl;
    /** TESTING_CODE::Dumping DatasetHandle trainData::end */

    LGBM_DatasetSetField(
        trainData, "label", 
        static_cast<void *>(labels.data()), labels.size(), 
        C_API_DTYPE_FLOAT32);

    if(LFO::init) {
        /** TESTING_CODE::beginning */
        /**
        std::cerr<<"LFO::trainModel(): LFO::init==true"<<std::endl; 
        */
        /** TESTING_CODE::end */
        // init booster //YK: create a new boosting learner
    	std::string params = param_map_to_string(LFO::trainParams);
        LGBM_BoosterCreate(trainData, params.c_str(), &LFO::booster);
        // train
        for (int i = 0; i < std::stoi(LFO::trainParams["num_iterations"]); 
            i++) {
            int isFinished;
            // YK: update the model for 1 iteration 
            LGBM_BoosterUpdateOneIter(LFO::booster, &isFinished);
            if (isFinished) {
                break;
            }
        }
        char booster_dump[1000000];
        int64_t out_len;
        if(!LGBM_BoosterDumpModel(
            LFO::booster, 
            0, 
            -1, 
            C_API_FEATURE_IMPORTANCE_SPLIT, 
            (int64_t)1000000, 
            &out_len, 
            booster_dump)) {
            std::cerr
                <<"LFO::trainModel()::init==true::"
                <<"out_len= "<<out_len<<", "
                <<"booster_dump= "<<booster_dump
                <<std::endl;
        } else {
            std::cerr
                <<"LFO::trainModel()::init==true::"
                <<"LGBM_BoosterDumpModel() failed"
                <<std::endl;
            std::exit(EXIT_FAILURE);
        }
    } else {
        /** TESTING_CODE::beginning */
        /**
        std::cerr<<"LFO::trainModel(): LFO::init==false"<<std::endl; 
        */
        /** TESTING_CODE::end */
        BoosterHandle newBooster;
        std::string params = param_map_to_string(LFO::trainParams);
        LGBM_BoosterCreate(trainData, params.c_str(), &newBooster);
        #ifdef MODEL_REFITTING 
        int64_t len;
        //YK: get number of predictions 
        LGBM_BoosterCalcNumPredict(booster, indptr.size() - 1, 
            C_API_PREDICT_LEAF_INDEX, 0, &len);
        vector<double> tmp(len);
        //YK: make prediction for a new dataset in CSR format 
        LGBM_BoosterPredictForCSR(booster, static_cast<void*>(indptr.data()), 
            C_API_DTYPE_INT32, indices.data(), static_cast<void*>(data.data()), 
            C_API_DTYPE_FLOAT64, indptr.size(), data.size(), HISTFEATURES + 3,
            C_API_PREDICT_LEAF_INDEX, 0, trainParams, &len, tmp.data());
        //YK: iterating through vector<double> tmp
        vector<int32_t> predLeaf(tmp.begin(), tmp.end());
        tmp.clear();
        //YK: merging model from booster to newBooster 
        LGBM_BoosterMerge(newBooster, booster);
        //YK: refit the tree model using the new data (online learning) 
        std::cerr<<"Refitting model ..."<<std::endl; 
        LGBM_BoosterRefit(newBooster, predLeaf.data(), indptr.size() - 1, 
            predLeaf.size() / (indptr.size() - 1));

        #else 
        // train a new booster
        for (int i = 0; i < std::stoi(LFO::trainParams["num_iterations"]); 
            i++) {
            int isFinished;
            LGBM_BoosterUpdateOneIter(newBooster, &isFinished);
            if (isFinished) {
                break;
            }
        }
        #endif
        LGBM_BoosterFree(LFO::booster);
        std::cerr<<"Replacing model ..."<<std::endl; 
        LFO::booster = newBooster;

        char booster_dump[1000000];
        int64_t out_len;
        if(!LGBM_BoosterDumpModel(
                LFO::booster, 
                0, 
                -1, 
                C_API_FEATURE_IMPORTANCE_SPLIT, 
                (int64_t)1000000,
                &out_len, 
                booster_dump)) {
            std::cerr
                <<"LFO::trainModel()::init==false::"
                <<"out_len= "<<out_len<<", "
                <<"booster_dump= "<<booster_dump
                <<std::endl;
        } else {
            std::cerr
                <<"LFO::trainModel()::init==false::"
                <<"LGBM_BoosterDumpModel() failed"
                <<std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
    LGBM_DatasetFree(trainData);

    return;
}
