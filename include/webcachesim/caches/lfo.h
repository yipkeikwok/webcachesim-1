#ifndef LFO_VARIANTS_H
#define LFO_VARIANTS_H

#include <unordered_map>
#include <boost/bimap.hpp>
#include <boost/bimap/multiset_of.hpp>
#include <boost/functional/hash.hpp>
#include <unordered_set>
#include <utility>
#include <string>
#include <list>
#include <random>
#include <limits>
#include "cache.h"
#include "adaptsize_const.h" /* AdaptSize constants */

#ifdef EVICTION_LOGGING
#include "mongocxx/client.hpp"
#endif

// for LRU. LRU to be used during 1st window 
typedef std::list<uint64_t >::iterator ListIteratorType;
// for LRU. LRU to be used during 1st window 
typedef std::unordered_map<uint64_t , ListIteratorType> lruCacheMapType;
typedef std::pair<std::uint64_t, double> lfoCacheMapKey_t;
typedef boost::bimap<
    boost::bimaps::set_of<lfoCacheMapKey_t>,
    boost::bimaps::multiset_of<double>
    > lfoCacheMapType;
typedef lfoCacheMapType::right_map::const_iterator right_const_iterator;

using namespace std;
using namespace webcachesim;

struct optEntry {
    uint64_t idx; 
    uint64_t volume;
    bool hasNext;
    uint64_t size; 
    uint64_t id;
    uint64_t seq;

    optEntry(uint64_t idx, uint64_t size) : 
        idx(idx), 
        volume(std::numeric_limits<uint64_t>::max()), 
        hasNext(false),
        size(size),
        id(std::numeric_limits<uint64_t>::max()) {
    };

    optEntry(uint64_t idx, uint64_t seq, uint64_t id, uint64_t size) : 
        idx(idx), 
        seq(seq),
        id(id),
        volume(std::numeric_limits<uint64_t>::max()), 
        hasNext(false),
        size(size) {
    };
};

struct trEntry {
    uint64_t id;
    uint64_t size;
    double cost; 
    bool toCache; 
    uint64_t lastSeenIndex;
    int lastSeenCount;
    uint64_t seq;

    trEntry(uint64_t seq, uint64_t id, uint64_t size, double cost, 
        int lastSeenCount) : 
        seq(seq), id(id), size(size), cost(cost), toCache(false), 
        lastSeenIndex(std::numeric_limits<uint64_t>::max()),
        lastSeenCount(lastSeenCount) {
    };

    trEntry(uint64_t seq, uint64_t id, uint64_t size, double cost, 
        uint64_t lastSeenIndex, int lastSeenCount) :
        seq(seq), id(id), size(size), cost(cost), toCache(false), 
        lastSeenIndex(lastSeenIndex), lastSeenCount(lastSeenCount) {
    };
};

namespace LFO {
    int OHR=0;
    int BHR=1;
    /** TESTING_CODE::cnt_quartile::beginning */
    // purpose: observe the distribution of rehit_probabilility 
    //  (i.e., prediction output) 
    /** 
    uint64_t cnt_quartile0 = (uint64_t)0;
    uint64_t cnt_quartile1 = (uint64_t)0;
    uint64_t cnt_quartile2 = (uint64_t)0;
    uint64_t cnt_quartile3 = (uint64_t)0;
    */
    /** TESTING_CODE::cnt_quartile::end */

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
    void conclude_window(int objective, uint64_t cache_size); 
    double calculate_rehit_probability(
        SimpleRequest& req, 
        uint64_t cacheAvailBytes, 
        int optimization_objective // 0 for OHR, 1 for BHR 
        ); 
    void calculateOPT(uint64_t cacheSize, int optimization_objective); 
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
    // for LRU. LRU to be used during 1st window 
    // list for LRU recency order
    std::list<uint64_t > _lruCacheList;
    // map to find objects in LRU recency list
    lruCacheMapType _lruCacheMap;
    // map to find objects in list
    lfoCacheMapType _cacheMap;
    unordered_map<uint64_t , uint64_t > _size_map;
    int _objective; // 0: OHR(default), 1: BHR
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

/** 
    virtual void hit(lfoCacheMapType::left_map::const_iterator it, 
        double rehit_probability);
*/

public:
    LFOCache();
    virtual ~LFOCache()
    {
    }

    void init_with_params(const map<string, string> &params) override {
        //set params
        std::cerr<<"LFOCache(): init_with_params(...): params.size()= "
            <<params.size()<<": ";
        for (auto& it: params) {
            std::cerr<<it.first<<"="<<it.second<<", "; 
            if (it.first == "objective") {
                _objective = std::stoi(it.second);
#ifdef EVICTION_LOGGING
            } else if (it.first == "byte_million_req") {
                byte_million_req = stoull(it.second);
            } else if (it.first == "task_id") {
                task_id = it.second;
            } else if (it.first == "dburi") {
                dburi = it.second;
#endif
            } else if (it.first == "n_extra_fields") {
                /** 
                    n_extra_fields refers the the number of columns after the 
                    3rd column
                */
                if(std::stoi(it.second)!=0) { 
                    std::cerr<<"n_extra_fields=="<<it.second<<"!=0"<<std::endl; 
                    std::exit(EXIT_FAILURE);
                }
            } else {
                cerr 
                    << "LFO::unrecognized parameter: key= " << it.first 
                    << ", value= "<<it.second<< endl;
                std::exit(EXIT_FAILURE); 
            }
        }
        std::cerr<<std::endl;
    }

    bool lookup(SimpleRequest &req) override;

    /**
    bool exist(const KeyT &key) override;
    */

    void admit(SimpleRequest &req) override;

    void evict(SimpleRequest &req);

    KeyT evict();

    /**
    // TODO: remove this function
    virtual const SimpleRequest & evict_return();
    */
};

static Factory<LFOCache> factoryLFO("LFO");

#endif
