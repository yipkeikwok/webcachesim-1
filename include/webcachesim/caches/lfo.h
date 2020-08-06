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
    // objective: 0 for OHR, 1 for BHR
    int8_t OHR=(int8_t)0; 
    int8_t BHR=(int8_t)1; 
    int8_t objective = LFO::OHR; 

    //Will be used to obtain a seed for the random number engine
    std::random_device rd;
    //Standard mersenne_twister_engine seeded with rd()
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<> dis(0.0, 1.0);
    //nmbr of rqst arrived so far
    uint64_t train_seq=(uint64_t)0;
    uint64_t windowSize=(uint64_t)1000000; 
    int sampling=1;
    bool init = true;
    BoosterHandle booster;
    // learning samples for model training
    // Sampling Mode 1: The number of requests at the end of windowTrace that 
    //  are used for deriving features (deriveFeatures())
    // Sampling Mode 2: The number of requests in windowTrace that are used for 
    //  deriving features. Those requests are evenly distributed in windowTrace
    uint64_t sampleSize = windowSize; 
    //std::pair<uint64_t, uint64_t> idsize;
    // stored the index (starting from 0) within the current sliding window
    //  when an object was last accessed
    //  <<object ID, object size>, index>
    std::unordered_map<
        std::pair<uint64_t,uint64_t>, 
        uint64_t, 
        boost::hash<std::pair<uint64_t,uint64_t>>
        > windowLastSeen; 
    std::vector<optEntry> windowOpt;
    std::vector<trEntry> windowTrace;
    uint64_t windowByteSum = 0;
    // unordered_map<uint64_t objectID, list<uint64_t> accessTimestamp> 
    //  accessTimestamp: since beginning of each sliding window 
    //      in descending order: later timestamps before older 
    std::unordered_map<uint64_t, std::list<uint64_t>> statistics; 
    /**
    default values in trainParams can be found in 
    https://lightgbm.readthedocs.io/en/latest/Parameters.html

    I change the values of the following from the provided code. 
    num_threads: for the best speed, set this to the number of real CPU cores, 
        not the number of threads (most CPUs use hyper-threading to generate 2 
        threads per CPU core)
    num_iterations: the paper suggests 30
    */
    std::unordered_map<string, string> trainParams = {
            {"boosting",                   "gbdt"},
            {"objective",                  "binary"},
            {"metric",                     "binary_logloss,auc"},
            {"metric_freq",                "1"},
            {"is_provide_training_metric", "true"},
            {"max_bin",                    "255"},
            {"num_iterations",             "30"},
            {"learning_rate",              "0.1"},
            {"num_leaves",                 "31"},
            {"tree_learner",               "serial"},
            {"num_threads",                "2"},
            {"feature_fraction",           "0.8"},
            {"bagging_freq",               "5"},
            {"bagging_fraction",           "0.8"},
            {"min_data_in_leaf",           "50"},
            {"min_sum_hessian_in_leaf",    "5.0"},
            {"is_enable_sparse",           "true"},
            {"two_round",                  "false"},
            {"save_binary",                "false"}
    };

    void annotate(uint64_t seq, uint64_t id, uint64_t size, double cost);
    void calculateOPT(uint64_t cacheSize); 
    void deriveFeatures(vector<float> &labels, vector<int32_t> &indptr, 
       vector<int32_t> &indices, vector<double> &data, int sampling, 
        uint64_t cacheSize);
    void trainModel(vector<float> &labels, vector<int32_t> &indptr, 
        vector<int32_t> &indices, vector<double> &data);
}

#define HISTFEATURES 50 //online features

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
