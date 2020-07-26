//#include <unordered_map>
#include <random>
#include <cmath>
#include <cassert>
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
    LFO::annotate(LFO::train_seq, req._id, req._size, 0.0); 
    if(!(LFO::train_seq%LFO::windowSize)) {
        LFO::calculateOPT(getSize());

        std::vector<float> labels;
        std::vector<int32_t> indptr;
        std::vector<int32_t> indices;
        std::vector<double> data;
        int sampling=2; // 1: select later requested object, 2: select randomly
        LFO::deriveFeatures(labels, indptr, indices, data, sampling, getSize());
        // trainModel();
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
            std::cerr<<count0<<" elements have >=1 elements"<<std::endl;
        }

        LFO::statistics.clear();
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
    if(!(seq % LFO::windowSize)) {
        std::cerr<<"End Window"<<(seq/LFO::windowSize-1)<<std::endl; 
    }

    const uint64_t idx= (seq-1) % LFO::windowSize; 
    // store access timestamps
    if(LFO::statistics.count(id)) {
        std::list<uint64_t>& list0=LFO::statistics[id];
        list0.push_front(idx);
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
    std::cerr<<"LFO::calculateOPT: cacheSize = " <<cacheSize
        <<", hitc = "<< hitc <<", bytehitc = "<< bytehitc <<std::endl; 
}

void LFO::deriveFeatures(vector<float> &labels, vector<int32_t> &indptr, 
    vector<int32_t> &indices, vector<double> &data, int sampling, 
    uint64_t freeCacheSizeBeginningWindow) {
    int64_t cacheAvailBytes = freeCacheSizeBeginningWindow;
    std::cerr<<"LFO::deriveFeatures(), cacheAvailBytes= "<<cacheAvailBytes
        <<std::endl; 
    
    return;
}

