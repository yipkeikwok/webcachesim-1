#ifndef LHD_VARIANTS_H
#define LHD_VARIANTS_H

#include <unordered_map>
#include <list>
#include <vector>
#include "cache.h"
#include "cache_object.h"
#include "rand.hpp"

/*
  LHD: Lowest Hit-Density eviction
*/
class LHDCache : public Cache
{
public:
    LHDCache()
        : Cache()
    {
        for(uint32_t i=0; i<NUM_CLASSES; i++) {
            classes.push_back(Class()); 
            auto& cl=classes.back(); 
            //  static constexpr age_t MAX_AGE = 20000;
            //  Note: MAX_AGE is a coarsened age
            cl.hits.resize(MAX_AGE, 0); 
            cl.evictions.resize(MAX_AGE, 0); 
            cl.hitDensities.resize(MAX_AGE, 0); 
        }

        // Initialize policy to ~GDSF by default.
        for (uint32_t c = 0; c < NUM_CLASSES; c++) {
            for (age_t a = 0; a < MAX_AGE; a++) {
                classes[c].hitDensities[a] =
                    1. * (c + 1) / (a + 1);
            }
        }
    }
    virtual ~LHDCache()
    {
    }

    virtual bool lookup(SimpleRequest* req);
    virtual void admit(SimpleRequest* req);
    virtual void evict(SimpleRequest* req);
    virtual void evict();
    virtual SimpleRequest* evict_return();

protected:

	// TYPES 
	typedef float rank_t; 
	typedef uint64_t timestamp_t; 
	typedef uint64_t age_t; 
    typedef std::vector<struct Tag>::iterator VectorIteratorType;
    // typedef std::unordered_map<CacheObject, VectorIteratorType> lhdCacheMapType;
    /** begin: FIFO */
    typedef std::list<CacheObject>::iterator ListIteratorType;
    typedef std::unordered_map<CacheObject, ListIteratorType> lhdCacheMapType;
    /** end: FIFO */

    struct candidate_t {
        int appId;
        int64_t id;

        /** 
        static candidate_t make(const parser::Request& req) {
            return candidate_t{req.appId, req.id};
        }
        */ 

inline bool operator==(const candidate_t& that) const {
return (id == that.id) && (appId == that.appId);
}

inline bool operator!=(const candidate_t& that) const {
return !(operator==(that));
}

inline bool operator<(const candidate_t& that) const {
if (this->appId == that.appId) {
return this->id < that.id;
} else {
return this->appId < that.appId;
}
}
};

    // info we track about each object
    struct Tag {
    // the un-coarsened time of the beginning of the block (single object's
    //  lifetime)
        age_t timestamp;
        age_t lastHitAge;
        age_t lastLastHitAge;
    // not actual appId
    // LHD::update(candidate_t id, const parser::Request& req) {
    //  tag->app = req.appId % APP_CLASSES;
    // }
        uint32_t appClassId;

        candidate_t id;
        rank_t size; // stored redundantly with cache
        bool explorer;
    };

	// info LHD traces about each class of objects 
	struct Class {
		std::vector<rank_t> hits; 
		std::vector<rank_t> evictions; 
		std::vector<rank_t> hitDensities; 
		rank_t totalHits = 0; 
		rank_t totalEvictions = 0; 
	};

	// CONSTANTS 
    // how to sample candidates; can significantly impact hit
    // ratio. want a value at least 32; diminishing returns after
    // that.
    const uint32_t ASSOCIATIVITY = 32;

    // since our cache simulator doesn't bypass objects, we always
    // consider the last ADMISSIONS objects as eviction candidates
    // (this is important to avoid very large objects polluting the
    // cache.) alternatively, you could do bypassing and randomly
    // admit objects as "explorers" (see below).
    const uint32_t ADMISSIONS = 8;

    // escape local minima by having some small fraction of cache
    // space allocated to objects that aren't evicted. 1% seems to be
    // a good value that has limited impact on hit ratio while quickly
    // discovering the shape of the access pattern.
    static constexpr rank_t EXPLORER_BUDGET_FRACTION = 0.01;
    // if "explorer" space budget has not been reached, in lhd.cpp::update(),
    // objects are being randomly put into the "explorer" space. The
    // probability is 1/EXPLORE_INVERSE_PROBABILITY
    static constexpr uint32_t EXPLORE_INVERSE_PROBABILITY = 32;

	// these parameters determine how aggressively to classify objects.
	// diminishing returns after a few classes; 16 is safe.
	static constexpr uint32_t HIT_AGE_CLASSES = 16;
	static constexpr uint32_t APP_CLASSES = 16;
	static constexpr uint32_t NUM_CLASSES = HIT_AGE_CLASSES * APP_CLASSES;

	// these parameters are tuned for simulation performance, and hit
	// ratio is insensitive to them at reasonable values (like these)
	static constexpr rank_t AGE_COARSENING_ERROR_TOLERANCE = 0.01;
	static constexpr age_t MAX_AGE = 20000;
	static constexpr timestamp_t ACCS_PER_RECONFIGURATION = (1 << 20);
	static constexpr rank_t EWMA_DECAY = 0.9;

	// FIELDS 
	// map to find objects in list
    /** begin: FIFO */
    std::list<CacheObject> _cacheList; 
    /** end: FIFO */
	lhdCacheMapType _cacheMap;
    std::vector<Tag> tags;
	std::vector<Class> classes; 

    int numReconfigurations = 0;
    misc::Rand rand;

	// METHODS 
	virtual void hit(lhdCacheMapType::const_iterator it, uint64_t size);
};

static Factory<LHDCache> factoryLHD("LHD");

#endif
