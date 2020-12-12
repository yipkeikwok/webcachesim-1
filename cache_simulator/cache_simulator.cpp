//#define NDEBUG
#include <cassert> 
#include <fstream> 
#include <iostream> 
#include <unordered_map> 
#include <utility> // std::pair
#include <boost/functional/hash.hpp> // boost::hash<>

bool lookup(uint64_t id, uint64_t size, uint64_t timestamp, 
    std::unordered_map<
            std::pair<uint64_t,uint64_t>, 
            uint64_t, 
            boost::hash<std::pair<uint64_t, uint64_t>>
        > cacheMap
    ) {
    std::cout<<"lookup(..): "<<id<<" "<<size<<" "<<timestamp<<std::endl;
    std::pair<uint64_t, uint64_t> id_size = std::make_pair(
        (uint64_t)id, 
        (uint64_t)size
        ); 
    int count;
    if(count = (int)cacheMap.count(id_size)) {
        assert(count==1); 
        cacheMap[id_size] = timestamp;
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
    uint64_t sz_cache_remaining; 
    sz_cache_remaining = std::stoul(argv[2], nullptr, 10); 
    std::cout<<"trace file:     "<<argv[1]<<std::endl; 
    std::cout<<"cache size:     "<<sz_cache_remaining<<std::endl; 
    uint64_t sz_window;
    sz_window = std::stoul(argv[3], nullptr, 10);
    std::cout<<"window size:    "<<sz_window<<std::endl;
    uint64_t count_object, count_byte, hit_object, hit_byte;
    count_object=count_byte=hit_object=hit_byte=(uint64_t)0; 

    uint64_t id, size; 
    int decision; 
    uint64_t timestamp; 
    timestamp=(uint64_t)0; 
    while(ifstream_trace >> id >> size >> decision) {
        if((decision<0)||(decision>1)) {
            std::cerr<<"Invalid decision "<<decision<<std::endl;
            std::exit(EXIT_FAILURE);
        }
        // std::cout<<id<<" "<<size<<" "<<decision<<std::endl;
        count_object++;
        count_byte+=size;
        if(!decision) {
            /**
            if(!lookup(id, size, timestamp, cacheMap)) {
                // not in cache 
            } else {
                // in cache 
                hit_object++;
                hit_byte+=size;
            }
            */
        } else if(decision==1) {
            if(!lookup(id, size, timestamp, cacheMap)) {
                // not in cache 
                std::pair<uint64_t, uint64_t> id_size = std::make_pair(
                    (uint64_t)id, (uint64_t)size
                    );
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
            } else {
                // in cache 
                // action: none
                std::cout<<id<<" "<<size<<" in cache"<<std::endl;
                hit_object++;
                hit_byte+=size;
            }
        } else {
            std::cerr<<"Invalid decision (2nd check): "<<decision<<std::endl;
            std::exit(EXIT_FAILURE);
        }
        timestamp++;           
    } // while(ifstream_trace >> id >> size >> decision) 
    ifstream_trace.close();

    std::cout
        <<"count_object= "<<count_object<<", hit_object= "<<hit_object
        <<std::endl
        <<"count_byte= "<<count_byte<<", hit_byte= "<<hit_byte
        <<std::endl;
    return 0;

    #if 0
    std::unordered_map<
        std::pair<uint64_t, uint64_t>, //<id, size> 
        uint8_t, // decision
        boost::hash<std::pair<uint64_t, uint64_t>>
        > cacheMap; 

    while(std::cin >> id >> size >> decision) {
        const std::pair<uint64_t, uint64_t> id_size = std::make_pair(id, size); 
        cacheMap[id_size] = decision; 
    }
    for(const auto& it: cacheMap) {
        // std::cout<<it.first<<" "<<it.second<<std::endl;
        const auto& pair0 = it.first;
        std::cout<<pair0.first<<" "<<pair0.second<<" "<<it.second<<std::endl;
        //uint64_t uint0 = it.second;
        //std::cout<<uint0<<std::endl;
        //std::cout<<cacheMap[it.first]<<std::endl;
    }
    #endif

    return 0;
}
