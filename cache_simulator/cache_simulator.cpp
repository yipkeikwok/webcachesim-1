#include <fstream> 
#include <iostream> 

int main(void) {
    uint64_t id; 
    uint64_t size; 
    uint8_t decision;

    while(std::cin >> id >> size >> decision) {
        std::cout<<id<<" "<<size<<" "<<decision<<std::endl;
    }
    return 0;
}
