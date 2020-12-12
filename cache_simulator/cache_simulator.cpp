#include <fstream> 
#include <iostream> 
#include <unordered_map> 
#include <utility> // std::pair
#include <boost/functional/hash.hpp> // boost::hash<>

int main(void) {
    uint64_t id; 
    uint64_t size; 
    uint8_t decision;
    // uint64_t timestamp;
    std::unordered_map<
        std::pair<uint64_t, uint64_t>, //<id, size> 
        uint8_t, // decision
        boost::hash<std::pair<uint64_t, uint64_t>>
        > cacheMap; 

    uint64_t timestamp=(uint64_t)0; 
    while(std::cin >> id >> size >> decision) {
        const std::pair<uint64_t, uint64_t> id_size = std::make_pair(id, size); 
        cacheMap[id_size] = decision; 
        timestamp++; 
    }
    for(const auto& it: cacheMap) {
        // std::cout<<it.first<<" "<<it.second<<std::endl;
        const auto& pair0 = it.first;
        std::cout<<pair0.first<<" "<<pair0.second<<" "<<it.second<<std::endl;
        //uint64_t uint0 = it.second;
        //std::cout<<uint0<<std::endl;
        //std::cout<<cacheMap[it.first]<<std::endl;
    }

    return 0;
}
