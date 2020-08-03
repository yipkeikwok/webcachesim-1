//#include <unordered_map>
#include <random>
#include <cmath>
#include <string>
/**
// NDEBUG is defined somewhere in this application. I do not know where.
#include <cassert>
*/
#include <cstdlib>
#include <LightGBM/application.h>
#include <LightGBM/c_api.h>
#include "lfo.h"
#include "random_helper.h"

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

#define NR_NON_TIMEGAP_ELMNT 3

/*
  LFO: Learning from OPT
*/
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

    LFO::train_seq++;
    LFO::annotate(LFO::train_seq, req._id, req._size, 1.0); 
    if(!(LFO::train_seq%LFO::windowSize)) {
        uint64_t cacheAvailBytes0 = getSize();
        LFO::calculateOPT(cacheAvailBytes0);

        /** 
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
            }
        }
        if(flag0) {
            std::cerr<<"only 1 element on each list"<<std::endl;
        } else {
            std::cerr<<count0<<" lists have >=1 elements"<<std::endl;
        }
        */

        std::vector<float> labels;
        std::vector<int32_t> indptr;
        std::vector<int32_t> indices;
        std::vector<double> data;
        uint64_t cacheAvailBytes1 = getSize();
   
        /** TESTING_CODE::beginning */
        if(cacheAvailBytes0 != cacheAvailBytes1) {
            std::cerr
                <<"cacheAvailBytes0 = "<<cacheAvailBytes0<<", "
                <<"cacheAvailBytes1 = "<<cacheAvailBytes1<<std::endl;
            std::exit(EXIT_FAILURE);
        }
        /** TESTING_CODE::end */
        LFO::deriveFeatures(labels, indptr, indices, data, LFO::sampling, 
            // getSize());
            cacheAvailBytes1);

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
        std::cerr<<"lookup(): indices.size()= "<<indices.size()<<std::endl;
        if(indices.size()!=data.size()) {
            std::cerr
                <<"indices.size()!=data.size()"<<": "
                <<"indices.size() = "<<indices.size()<<", "
                <<"data.size() = "<< data.size()
                <<std::endl;
        }
        */
        /** TESTING_CODE::end */
        LFO::trainModel(labels, indptr, indices, data);
        labels.clear();
        indptr.clear();
        indices.clear();
        data.clear();

        LFO::statistics.clear();
        LFO::windowLastSeen.clear();
        LFO::windowOpt.clear();
        LFO::windowTrace.clear();
    }

    uint64_t & obj = req._id;
    auto it = _cacheMap.find(obj);
    if (it != _cacheMap.end()) {
        // log hit
        auto & size = _size_map[obj];
        LOG("h", 0, obj.id, obj.size);
        hit(it, size);
        return true;
    }
    return false;
}

void LFOCache::admit(SimpleRequest& req)
{
    const uint64_t size = req.get_size();
    // object feasible to store?
    if (size > _cacheSize) {
        LOG("L", _cacheSize, req._id, size);
        return;
    }
    // check eviction needed
    while (_currentSize + size > _cacheSize) {
        evict();
    }
    // admit new object
    uint64_t & obj = req._id;
    _cacheList.push_front(obj);
    _cacheMap[obj] = _cacheList.begin();
    _currentSize += size;
    _size_map[obj] = size;
    LOG("a", _currentSize, obj.id, obj.size);
}

void LFOCache::evict(SimpleRequest& req)
{
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
}

void LFOCache::evict()
{
    // evict least popular (i.e. last element)
    if (_cacheList.size() > 0) {
        ListIteratorType lit = _cacheList.end();
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
        _cacheMap.erase(obj);
        _cacheList.erase(lit);
    }
}

void LFOCache::hit(lfoCacheMapType::const_iterator it, uint64_t size)
{
    _cacheList.splice(_cacheList.begin(), _cacheList, it->second);
}

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

bool LFOCache::exist(const KeyT &key) {
    return _cacheMap.find(key) != _cacheMap.end();
}

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
    if(LFO::statistics.count(id)) {
        std::list<uint64_t>& list0=LFO::statistics[id];
        list0.push_front(idx);
        if(list0.size() > HISTFEATURES) 
            list0.pop_back();
    } else {
        // first time this object is accessed in this sliding window 
        std::list<uint64_t> list0;
        list0.push_front(id);
        LFO::statistics[id]=list0;
    }
    const auto idsize = std::make_pair(id, size); 
    if(LFO::windowLastSeen.count(idsize)>0) {
        LFO::windowOpt[LFO::windowLastSeen[idsize]].hasNext = true;
        LFO::windowOpt[LFO::windowLastSeen[idsize]].volume = (idx - 
           LFO::windowLastSeen[idsize]) * size;
    }
    LFO::windowByteSum += size;
    LFO::windowLastSeen[idsize]=idx;
    LFO::windowOpt.emplace_back(idx); 
    LFO::windowTrace.emplace_back(id, size, cost);

    return; 
}

void LFO::calculateOPT(uint64_t cacheSize) {
    /**
    Note: cacheSize=physical cache size - size of metadata
    i.e., size of the cache used to store objects 

    */ 

    sort(LFO::windowOpt.begin(), LFO::windowOpt.end(), 
        [](const struct optEntry &lhs, const struct optEntry &rhs) {
            return lhs.volume < rhs.volume;
    });

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
    std::cerr<<"LFO::calculateOPT: cacheSize = " <<cacheSize
        <<", hitc = "<< hitc <<", bytehitc = "<< bytehitc <<std::endl; 
    */
    /** TESTING_CODE::end */
}

void LFO::deriveFeatures(std::vector<float> &labels, 
    std::vector<int32_t> &indptr, std::vector<int32_t> &indices, 
    std::vector<double> &data, int sampling, uint64_t cacheSize) {
    int64_t cacheAvailBytes = (int64_t)cacheSize;
    /** TESTING_CODE::beginning */
    /**
    std::cerr << "LFO::deriveFeatures(), cacheAvailBytes= " << cacheAvailBytes
        << ", sampling = " << sampling
        << ", LFO::sampleSize = "<< LFO::sampleSize 
        << ", HISTFEATURES = " << HISTFEATURES 
        << ", sampling = " << sampling 
        << ", sampleSize = " << sampleSize 
        <<std::endl; 
    */
    /** TESTING_CODE::end */

    // unordered_map<object ID, object size> cache 
    std::unordered_map<uint64_t, uint64_t> cache;

    uint64_t negCacheSize = (uint64_t)0;
    // -ve cache size below 0 by how much
    uint64_t negCacheSizeMax = (uint64_t)0;

    uint64_t i = (uint64_t)0;
    indptr.push_back(0);

    // nmbr of samples taken during each window
    uint64_t flagCnt=(uint64_t)0;
    /** TESTING_CODE::beginning */
    /**
    uint64_t expected_indices_size=(uint64_t)0;
    */
    /** TESTING_CODE::end */
    for(auto &it: windowTrace) {
        auto &curQueue = statistics[it.id];
        const auto curQueueLen = curQueue.size();
        if(curQueueLen > HISTFEATURES) {
            curQueue.pop_back();
        }

        bool flag = true;
        if(sampling == 1) {
            // YK: i incremented in each for-loop iteration
            // sampleSize is an element in map<string, string> params, an 
            //  argument of _simulation_lfo(). As on 2020071, since 
            // _simulation_lfo() not invoked in the code provided, I do not know
            // the value of sampleSize 
            flag = i >= (windowSize - LFO::sampleSize);
        } else if (sampling == 2) {
            // uniform_real_distribution<> dis(0.0, 1.0);
            double rand = LFO::dis(LFO::gen);
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
            uint64_t lastReqTime = i;

            uint64_t indices_size_old = (uint64_t)indices.size();

            //YK: curQueue: feature list of an object 
            for (auto &lit: curQueue) { 
                const uint64_t dist = lastReqTime - lit; // distance
                // YK: std::vector::push_back(): append to the end 
                indices.push_back(idx);
                data.push_back(dist);
                if(idx==0)
                    assert(data[idx]<0.0);
                else
                    assert(dist>0.0);
                idx++;
                lastReqTime = lit;
            }

            // object size
            indices.push_back(HISTFEATURES);
            data.push_back(round(100.0 * log2(it.size)));

            //YK: remaining cache space is a feature. Why did they comment the
            // code? 
            double currentSize = cacheAvailBytes <= 0 ? 
                0 : round(100.0 * log2(cacheAvailBytes));
            indices.push_back(HISTFEATURES + 1);
            data.push_back(currentSize);
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
            /** TESTING_CODE::beginning */
            if(NR_NON_TIMEGAP_ELMNT!=3) {
                std::cerr<<"NR_NON_TIMEGAP_ELMNT!=3"<<std::endl;
                std::exit(EXIT_FAILURE);
            }

            /** 
            uint64_t indices_size_expected = (uint64_t)0;
            indices_size_expected += indices_size_old;
            indices_size_expected += (uint64_t)curQueue.size();
            indices_size_expected += (uint64_t)NR_NON_TIMEGAP_ELMNT;
            if(indices.size() != indices_size_expected) {
                std::cerr
                    <<"indices.size() != indices_size_expected, "
                    <<"indices.size() = " << indices.size() << ", "
                    <<"indices_size_expected = " << indices_size_expected
                    <<std::endl;
                std::exit(EXIT_FAILURE);
            }
            */
            /**
            expected_indices_size+= (uint64_t)curQueue.size();
            expected_indices_size+= (uint64_t)NR_NON_TIMEGAP_ELMNT;
            */
            /** TESTING_CODE::end */
            indptr.push_back(indptr[indptr.size() - 1] + idx 
                + NR_NON_TIMEGAP_ELMNT);
        } // if (flag) {

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
        curQueue.push_front(i++);
    } //for(auto &it: windowTrace) 

    /** TESTING_CODE::beginning */
    /**
    if(indices.size() != expected_indices_size) {
        std::cerr
            <<"indices.size() != expected_indices_size, "
            <<"indices.size() = " << indices.size() << ", "
            <<"indices_size_expected = " << expected_indices_size
            <<std::endl;
        std::exit(EXIT_FAILURE);
    }
    std::cerr<<"deriveFeatures(): expected_indices_size = " 
        << expected_indices_size << std::endl;
    */
    /** TESTING_CODE::beginning */

    /** TESTING_CODE::beginning */
    /**
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
    */
    /** TESTING_CODE::beginning */

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
    LGBM_DatasetCreateFromCSR(
        static_cast<void *>(indptr.data()), C_API_DTYPE_INT32, 
        indices.data(), 
        static_cast<void *>(data.data()), C_API_DTYPE_FLOAT64, 
        indptr.size(), 
        data.size(), 
        HISTFEATURES + 3, 
        LFO::trainParams, nullptr, &trainData);

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
        LGBM_BoosterCreate(trainData, LFO::trainParams, &LFO::booster);
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
        LFO::init = false;
    } else {
        /** TESTING_CODE::beginning */
        /**
        std::cerr<<"LFO::trainModel(): LFO::init==false"<<std::endl; 
        */
        /** TESTING_CODE::end */
        BoosterHandle newBooster;
        LGBM_BoosterCreate(trainData, LFO::trainParams, &newBooster);
        // train a new booster
        for (int i = 0; i < std::stoi(LFO::trainParams["num_iterations"]); 
            i++) {
            int isFinished;
            LGBM_BoosterUpdateOneIter(newBooster, &isFinished);
            if (isFinished) {
                break;
            }
        }
        LGBM_BoosterFree(LFO::booster);
        LFO::booster = newBooster;
    }
    LGBM_DatasetFree(trainData);

    return;
}
