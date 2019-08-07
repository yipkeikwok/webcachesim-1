#include <unordered_map>
#include <limits>
#include <cmath>
#include <cassert>
#include <cmath>
#include <cassert>
#include "lhd_variants.h"
#include "../random_helper.h"

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
  LHD: Lowest Hit-Density eviction
*/
bool LHDCache::lookup(SimpleRequest* req)
{
    // CacheObject: defined in cache_object.h 
    CacheObject obj(req);
    // _cacheMap defined in class LRUCache in lru_variants.h 
    auto it = _cacheMap.find(obj);
    if (it != _cacheMap.end()) {
        // log hit
        LOG("h", 0, obj.id, obj.size);
        hit(it, obj.size);
        return true;
    }
    return false;
}

void LHDCache::admit(SimpleRequest* req)
{
    const uint64_t size = req->getSize();
    // object feasible to store?
    if (size > _cacheSize) {
        LOG("L", _cacheSize, req->getId(), size);
        return;
    }
    // check eviction needed
    while (_currentSize + size > _cacheSize) {
        evict();
    }
    // admit new object
    CacheObject obj(req);
	/** 
	TODO: add obj into _cacheMap 
	TODO: deduce classID 
	TODO: update classes[classID].hit[age] 
    _cacheList.push_front(obj);
    _cacheMap[obj] = _cacheList.begin();
	*/ 
    /** begin: FIFO */
    _cacheList.push_back(obj); 
    _cacheMap[obj]=_cacheList.end();   
    /** end: FIFO */
    _currentSize += size;
    LOG("a", _currentSize, obj.id, obj.size);
}

void LHDCache::evict(SimpleRequest* req)
{
	#if 0
    CacheObject obj(req);
    auto it = _cacheMap.find(obj);
    if (it != _cacheMap.end()) {
        ListIteratorType lit = it->second;
        LOG("e", _currentSize, obj.id, obj.size);
        _currentSize -= obj.size;
        _cacheMap.erase(obj);
	/** 
	TODO: update classes[classID].hit[age] 
        _cacheList.erase(lit);
	*/
    } else {
	std::cerr << "Eviction victim cannot be found in cache" << std::endl;
	std::exit(1);
    }
	#endif
}

SimpleRequest* LHDCache::evict_return()
{
	#if 0
    // evict least popular (i.e. last element)
    if (_cacheList.size() > 0) {
        ListIteratorType lit = _cacheList.end();
        lit--;
        CacheObject obj = *lit;
        LOG("e", _currentSize, obj.id, obj.size);
        SimpleRequest* req = new SimpleRequest(obj.id, obj.size);
        _currentSize -= obj.size;
        _cacheMap.erase(obj);
        _cacheList.erase(lit);
        return req;
    }
    return NULL;
	#endif

    /** begin: FIFO */
    if(_cacheList.size()>0) {
        ListIteratorType lit=_cacheList.begin(); 
        CacheObject obj=*lit;
        LOG("e", _currentSize, obj.id, obj.size);
        SimpleRequest* req=new SimpleRequest(obj.appId, obj.id, obj.size); 
        _currentSize -= obj.size;
        _cacheMap.erase(obj); 
        _cacheList.erase(lit);
        return req;
    }
    /** end: FIFO */
   
    rank_t victimRank = std::numeric_limits<rank_t>::max();
    victimRank+=.0; // for working around the 'unused-variable' compiler error 

    // Sample few candidates early in the trace so that we converge
    // quickly to a reasonable policy.
    //
    // This is a hack to let us have shorter warmup so we can evaluate
    // a longer fraction of the trace; doesn't belong in a real
    // system.
    uint32_t candidates =
        (numReconfigurations > 50)?
        ASSOCIATIVITY : 8;
    
    for(uint32_t i=0; i<candidates; i++) {
        auto idx = rand.next() % _cacheMap.size(); 
        auto& tag = tags[idx];
        std::cout << tag.size <<std::endl; 
    }

	return NULL; 
}

void LHDCache::evict()
{
    evict_return();
}

// const_iterator: a forward iterator to const value_type, where 
// value_type is pair<const key_type, mapped_type>
void LHDCache::hit(lhdCacheMapType::const_iterator it, uint64_t size)
{
	#if 0
    // transfers it->second (i.e., ObjInfo) from _cacheList into 
    // 	*this. The transferred it->second is to be inserted before 
    // 	the element pointed to by _cacheList.begin()
    //
    // _cacheList is defined in class LRUCache in lru_variants.h 
    _cacheList.splice(_cacheList.begin(), _cacheList, it->second);
	#endif
}

