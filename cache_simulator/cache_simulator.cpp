/** 
Cache Simulator
What it does: Given a trace, calculate overall and per-window OHR and BHR

To compile: 
g++ -g -std=c++11 -o ./cache_simulator.exe ./cache_simulator.cpp

To run: 
cache_simulator.exe [calculateOPT trace] [cache size] [window size]

trace format: [object ID] [object size] [caching decision: 0(not cache)/1(cache)
778030675 13539 0
3484637698 17635 0
1130267045 152 1
3423744231 616 1

Note: If the size of an object has changes, it is regarded as a NEW object.
*/

//#define NDEBUG
//#define  SAFE_CHECK0
#define DEBUG_LEVEL 0
#include <cassert> 
#include <fstream> 
#include <iostream> 
#include <cstdlib> // size_t
#include <unordered_map> 
#include <vector> 
#include <utility> // std::pair
#include <boost/functional/hash.hpp> // boost::hash<>

bool lookup(uint64_t id, uint64_t size, uint64_t timestamp, 
    std::unordered_map<
            std::pair<uint64_t,uint64_t>, 
            uint64_t, 
            boost::hash<std::pair<uint64_t, uint64_t>>
        > &cacheMap,
    uint64_t &hit_object, 
    uint64_t *hit_byte
    ) {
    #if DEBUG_LEVEL > 0
    std::cout<<"lookup(..): "<<id<<" "<<size<<" "<<timestamp<<std::endl;
    #endif
    std::pair<uint64_t, uint64_t> id_size = std::make_pair(
        (uint64_t)id, 
        (uint64_t)size
        ); 
    int count;
    if(count = (int)cacheMap.count(id_size)) {
        assert(count==1); 
        hit_object++;
        *hit_byte+=size;
        cacheMap[id_size] = timestamp;
        // std::cout<<"updated timestamp= "<<cacheMap[id_size]<<std::endl; 
        return true;
    } 
    assert(count==0);
    return false;
}

int main(int argc, char**argv) {
    if(argc!=4) { 
        std::cerr
            <<"Usage: cache_simulator.exe [calculateOPT trace] "
            <<"[cache size] [window size]"<<std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::ifstream ifstream_trace; 
    ifstream_trace.open(argv[1], std::ios::in);

    std::unordered_map<
            std::pair<uint64_t, uint64_t>,  // id, size
            uint64_t,                       // timestamp
            boost::hash<std::pair<uint64_t, uint64_t>>
        > cacheMap;
    uint64_t sz_cache_capacity, sz_cache_remaining; 
    // minimum remaining cache size in each window 
    uint64_t sz_cache_remaining_min_window 
        = std::numeric_limits<uint64_t>::max();
    sz_cache_capacity = sz_cache_remaining = std::stoul(argv[2], nullptr, 10); 
    std::cout<<"trace file:     "<<argv[1]<<std::endl; 
    std::cout<<"cache size remaining:     "<<sz_cache_remaining<<std::endl; 
    std::cout<<"cache size capacity:      "<<sz_cache_capacity <<std::endl; 
    uint64_t sz_window;
    sz_window = std::stoul(argv[3], nullptr, 10);
    std::cout<<"window size:    "<<sz_window<<std::endl;
    uint64_t count_object_overall, count_byte_overall, hit_object_overall, 
        hit_byte_overall;
    count_object_overall=count_byte_overall=hit_object_overall=hit_byte_overall
        =(uint64_t)0; 
    uint64_t count_object_window, count_byte_window, hit_object_window, 
        hit_byte_window;
    count_object_window=count_byte_window=hit_object_window=hit_byte_window
        =(uint64_t)0; 
    #ifdef SAFE_CHECK0
    uint64_t count_object_window_sum, count_byte_window_sum;
    uint64_t hit_object_window_sum,   hit_byte_window_sum;
    count_object_window_sum=count_byte_window_sum=(uint64_t)0; 
    hit_object_window_sum  =hit_byte_window_sum  =(uint64_t)0;
    #endif

    struct trace_entry {
        uint64_t id; 
        uint64_t size;
        int decision;

        trace_entry(uint64_t id, uint64_t size, int decision) : 
            id(id), size(size), decision(decision) {
        };
    };
    std::vector<struct trace_entry> window_trace;

    std::unordered_map<
        std::pair<uint64_t, uint64_t>,  //<id, size>
        uint64_t,                       // last_seen_index
        boost::hash<std::pair<uint64_t, uint64_t>>
        > window_last_seen;

    uint64_t id, size; 
    int decision; 
    uint64_t timestamp; 
    timestamp=(uint64_t)0; 
    uint64_t index = (uint64_t)0;
    // smaller the volume, higher the hit density, more likely to cache
    // min volume of an object that LFO::calculateOPT() decided not to cache
    uint64_t min_volume = std::numeric_limits<uint64_t>::max();
    // max volume of an object that LFO::calculateOPT() decided to cache
    uint64_t max_volume = (uint64_t)0;
    while(ifstream_trace >> id >> size >> decision) {
        if((decision<0)||(decision>1)) {
            std::cerr<<"Invalid decision "<<decision<<std::endl;
            std::exit(EXIT_FAILURE);
        }
        // std::cout<<id<<" "<<size<<" "<<decision<<std::endl;

        window_trace.emplace_back(id, size, decision);
        count_object_overall++;
        count_object_window ++;
        count_byte_overall+=size;
        count_byte_window +=size;
        std::pair<uint64_t, uint64_t> id_size = std::make_pair(
            (uint64_t)id, (uint64_t)size
            );
        if(window_last_seen.count(id_size)==(size_t)0) {
            // object accessed first 1st time in this window 
        } else {
            // object accessed early in this window 
            assert(window_last_seen.count(id_size)==(size_t)1);
            uint64_t last_seen_index=window_last_seen[id_size];
            uint64_t volume=size*(index-last_seen_index);
            if(window_trace[last_seen_index].decision==0) {
                if(volume<min_volume) {
                    min_volume=volume;
                }
            } else {
                assert(window_trace[last_seen_index].decision==1);
                if(volume>max_volume) {
                    max_volume=volume;
                }
            }
        }
        window_last_seen[id_size]=index;
        index++;
        if(!decision) {
            if(!lookup(id, size, timestamp, cacheMap, hit_object_window, &hit_byte_window
                    )
                ) {
                // not in cache 
            } else {
                // in cache 
                if(!(sz_cache_remaining + size <= sz_cache_capacity)) {
                    std::cerr
                        <<"sz_cache_remaining + size should <= "
                        <<"sz_cache_capacity: "
                        <<"sz_cache_remaining= "<<sz_cache_remaining<<", "
                        <<"sz_object= "<<size<<", "
                        <<"sz_cache_capacity= "<<sz_cache_capacity<<std::endl;
                    std::exit(EXIT_FAILURE);
                }
                #if DEBUG_LEVEL > 0
                std::cout<<id<<" "<<size<<" in cache"<<std::endl;
                #endif
                hit_object_overall++; 
                hit_byte_overall  +=size;
                if(cacheMap.erase(id_size) != (size_t)1) {
                    std::cerr
                        <<"failed to erase ("<<id<<", "<<size<<") from cache"
                        <<std::endl;
                    std::exit(EXIT_FAILURE);
                }
                sz_cache_remaining += size;
            }
        } else if(decision==1) {
            if(!lookup(id, size, timestamp, cacheMap, hit_object_window, &hit_byte_window
                    )
                ) {
                // not in cache 
                assert(
                    cacheMap.count(id_size)==(size_t)0
                    );
                if(!(sz_cache_remaining >= size)) {
                    std::cerr<<"sz_cache_remaining should >= sz_object: "
                        <<"sz_cache_remaining= "<<sz_cache_remaining<<", "
                        <<"sz_object= "<<size<<std::endl;
                    std::exit(EXIT_FAILURE);
                }
                cacheMap[id_size] = timestamp; 
                assert(
                    cacheMap.count(id_size)==(size_t)1
                    );
                sz_cache_remaining -= size;
                if(sz_cache_remaining < sz_cache_remaining_min_window) {
                    sz_cache_remaining_min_window=sz_cache_remaining; 
                }
                assert(sz_cache_remaining_min_window<=sz_cache_remaining);
            } else {
                // in cache 
                // action: none
                #if DEBUG_LEVEL > 0
                std::cout<<id<<" "<<size<<" in cache"<<std::endl;
                #endif
                hit_object_overall++;
                hit_byte_overall  +=size;
            }
        } else {
            std::cerr<<"Invalid decision (2nd check): "<<decision<<std::endl;
            std::exit(EXIT_FAILURE);
        }
        timestamp++;
        if(!(timestamp%sz_window)) {
            std::cout 
                <<"Window "<<timestamp/sz_window-1<<": "
                <<"min volume TO NOT CACHE= "<<min_volume<<", "
                <<"max volume to CACHE= "<<max_volume<<std::endl;
            assert(max_volume < min_volume);
            window_trace.clear();
            window_last_seen.clear();
            min_volume=std::numeric_limits<uint64_t>::max();
            max_volume=(uint64_t)0;
            index=(uint64_t)0;
            std::cout 
                    <<"Window "<<timestamp/sz_window-1<<": "
                    <<"OHR= "<< (float)hit_object_window/count_object_window
                    <<" ("<<hit_object_window<<"/"<<count_object_window<<")"
                    <<", "
                    <<"BHR= "<< (float)hit_byte_window/count_byte_window
                    <<" ("<<hit_byte_window<<"/"<<count_byte_window<<")"
                    <<", "
                    // If calculateOPT() is implemented correctly, 
                    //  cacheMap.size() should ==0 at end of each window 
                    <<"cacheMap.size()= "<<cacheMap.size()
                    <<", "
                    <<"min remaining cache sz= " 
                        << sz_cache_remaining_min_window;
            if(cacheMap.size()) {
                    std::cout
                        <<", "
                        <<cacheMap.begin()->first.first     // id
                        <<" "
                        <<cacheMap.begin()->first.second    // size
                        <<" "
                        // latest access timestamp
                        <<cacheMap.begin()->second;
            }
            std::cout<<std::endl;
            #ifdef SAFE_CHECK0
            count_object_window_sum+=count_object_window;
            count_byte_window_sum  +=count_byte_window  ;
            hit_object_window_sum  +=hit_object_window  ;
            hit_byte_window_sum    +=hit_byte_window    ;
            #endif
            count_object_window=count_byte_window=hit_object_window
                =hit_byte_window=(uint64_t)0; 
            sz_cache_remaining_min_window=std::numeric_limits<uint64_t>::max();
        }
    } // while(ifstream_trace >> id >> size >> decision) 
    if(timestamp%sz_window) {
        std::cout 
                <<"Window "<<timestamp/sz_window<<": "
                <<"OHR= "<< (float)hit_object_window/count_object_window
                <<" ("<<hit_object_window<<"/"<<count_object_window<<")"
                <<", "
                <<"BHR= "<< (float)hit_byte_window/count_byte_window
                <<" ("<<hit_byte_window<<"/"<<count_byte_window<<")"
                <<", "
                <<"cacheMap.size()= "<<cacheMap.size();
        if(cacheMap.size()) {
            std::cout
                <<", "
                <<cacheMap.begin()->first.first     // id
                <<" "
                <<cacheMap.begin()->first.second    // size
                <<" "
                // latest access timestamp
                <<cacheMap.begin()->second;
        }
        std::cout<<std::endl;
        #ifdef SAFE_CHECK0
        count_object_window_sum+=count_object_window;
        count_byte_window_sum  +=count_byte_window  ;
        hit_object_window_sum  +=hit_object_window  ;
        hit_byte_window_sum    +=hit_byte_window    ;
        #endif
    }
    std::cout<<"trace.size()= "<<window_trace.size()<<std::endl;
    ifstream_trace.close();

    #ifdef SAFE_CHECK0
    assert(count_object_window_sum == count_object_overall); 
    assert(count_byte_window_sum   == count_byte_overall); 
    assert(hit_object_window_sum   == hit_object_overall); 
    assert(hit_byte_window_sum     == hit_byte_overall); 
    #endif

    std::cout 
            <<"Overall: "
            <<"OHR= "<< (float)hit_object_overall/count_object_overall
            <<" ("<<hit_object_overall<<"/"<<count_object_overall<<")"
            <<", "
            <<"BHR= "<< (float)hit_byte_overall  /count_byte_overall
            <<" ("<<hit_byte_overall  <<"/"<<count_byte_overall  <<")"
            <<", "
            <<"cacheMap.size()= "<<cacheMap.size()
            <<std::endl;

    #if DEBUG_LEVEL > 0
    std::cout<<"cached object (id size latest_access_timestamp)"<<std::endl;
    for(const auto &it : cacheMap) {
        std::pair<uint64_t, uint64_t> pair0 = it.first; 
        std::cout<<pair0.first<<" "<<pair0.second<<" "<<it.second<<std::endl;
    }
    #endif

    return 0;
}
