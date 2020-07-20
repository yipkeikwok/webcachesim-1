#ifndef LFO_VARIANTS_H
#define LFO_VARIANTS_H

#include <unordered_map>
#include <boost/functional/hash.hpp>
#include <unordered_set>
#include <utility>
#include <list>
#include <random>
#include "cache.h"
#include "adaptsize_const.h" /* AdaptSize constants */

#ifdef EVICTION_LOGGING
#include "mongocxx/client.hpp"
#endif

typedef std::list<uint64_t >::iterator ListIteratorType;
typedef std::unordered_map<uint64_t , ListIteratorType> lfoCacheMapType;


using namespace std;
using namespace webcachesim;

struct optEntry {
    uint64_t idx; 
    uint64_t volume;
    bool hasNext;

    optEntry(uint64_t idx) : 
        idx(idx), 
        volume(std::numeric_limits<uint64_t>::max()), 
        hasNext(false) {
    };
};

struct trEntry {
    uint64_t id;
    uint64_t size;
    double cost; 
    bool toCache; 

    trEntry(uint64_t id, uint64_t size, double cost) : 
        id(id), size(size), cost(cost), toCache(false) {
    };
};

namespace LFO {
    //nmbr of rqst arrived so far
    uint64_t train_seq=(uint64_t)0;
    uint64_t windowSize=(uint64_t)1000000; 
    //std::pair<uint64_t, uint64_t> idsize;
    std::unordered_map<
        std::pair<uint64_t,uint64_t>, 
        uint64_t, 
        boost::hash<std::pair<uint64_t,uint64_t>>
        > windowLastSeen; 
    std::vector<optEntry> windowOpt;
    std::vector<trEntry> windowTrace;
    uint64_t windowByteSum = 0;

    void annotate(uint64_t seq, uint64_t id, uint64_t size, double cost);
    void calculateOPT(uint64_t cacheSize); 
}

/*
  LFO: Learning from OPT
*/
class LFOCache : public Cache
{
protected:
    // list for recency order
    std::list<uint64_t > _cacheList;
    // map to find objects in list
    lfoCacheMapType _cacheMap;
    unordered_map<uint64_t , uint64_t > _size_map;
#ifdef EVICTION_LOGGING
    uint32_t current_t;
    unordered_map<uint64_t, uint32_t> future_timestamps;
    vector<uint8_t> eviction_qualities;
    vector<uint16_t> eviction_logic_timestamps;
    unordered_map<uint64_t, uint32_t> last_timestamps;
    vector<uint8_t> hits;
    vector<uint16_t> hit_timestamps;
    uint64_t byte_million_req;
    string task_id;
    string dburi;
#endif


#ifdef EVICTION_LOGGING

    void init_with_params(const map<string, string> &params) override {
        //set params
        for (auto &it: params) {
            if (it.first == "byte_million_req") {
                byte_million_req = stoull(it.second);
            } else if (it.first == "task_id") {
                task_id = it.second;
            } else if (it.first == "dburi") {
                dburi = it.second;
            } else {
                cerr << "unrecognized parameter: " << it.first << endl;
            }
        }
    }
#endif


#ifdef EVICTION_LOGGING
    void update_stat(bsoncxx::builder::basic::document &doc) override {
        //Log to GridFs because the value is too big to store in mongodb
        try {
            mongocxx::client client = mongocxx::client{mongocxx::uri(dburi)};
            mongocxx::database db = client["webcachesim"];
            auto bucket = db.gridfs_bucket();

            auto uploader = bucket.open_upload_stream(task_id + ".evictions");
            for (auto &b: eviction_qualities)
                uploader.write((uint8_t *) (&b), sizeof(uint8_t));
            uploader.close();
            uploader = bucket.open_upload_stream(task_id + ".eviction_timestamps");
            for (auto &b: eviction_logic_timestamps)
                uploader.write((uint8_t *) (&b), sizeof(uint16_t));
            uploader.close();
            uploader = bucket.open_upload_stream(task_id + ".hits");
            for (auto &b: hits)
                uploader.write((uint8_t *) (&b), sizeof(uint8_t));
            uploader.close();
            uploader = bucket.open_upload_stream(task_id + ".hit_timestamps");
            for (auto &b: hit_timestamps)
                uploader.write((uint8_t *) (&b), sizeof(uint16_t));
            uploader.close();
        } catch (const std::exception &xcp) {
            cerr << "error: db connection failed: " << xcp.what() << std::endl;
            abort();
        }
    }
#endif

    virtual void hit(lfoCacheMapType::const_iterator it, uint64_t size);

public:
    LFOCache()
        : Cache()
    {
    }
    virtual ~LFOCache()
    {
    }

    bool lookup(SimpleRequest &req) override;

    bool exist(const KeyT &key) override;

    void admit(SimpleRequest &req) override;

    void evict(SimpleRequest &req);

    void evict();
    virtual const SimpleRequest & evict_return();
};

static Factory<LFOCache> factoryLFO("LFO");

#endif
