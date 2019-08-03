#ifndef LHD_VARIANTS_H
#define LHD_VARIANTS_H

#include <unordered_map>
#include <list>
#include <random>
#include "cache.h"
#include "cache_object.h"


typedef std::list<CacheObject>::iterator ListIteratorType;
typedef std::unordered_map<CacheObject, ListIteratorType> lruCacheMapType;

/*
  LHD: Lowest Hit-Density eviction
*/
class LHDCache : public Cache
{
protected:
    // list for recency order
    // std::list is a container, usually, implemented as a doubly-linked list 
    std::list<CacheObject> _cacheList;
    // map to find objects in list
    lruCacheMapType _cacheMap;

    virtual void hit(lruCacheMapType::const_iterator it, uint64_t size);

public:
    LHDCache()
        : Cache()
    {
    }
    virtual ~LHDCache()
    {
    }

    virtual bool lookup(SimpleRequest* req);
    virtual void admit(SimpleRequest* req);
    virtual void evict(SimpleRequest* req);
    virtual void evict();
    virtual SimpleRequest* evict_return();
};

static Factory<LHDCache> factoryLHD("LHD");

#endif
