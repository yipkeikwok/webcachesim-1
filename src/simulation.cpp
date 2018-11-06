//
// Created by Zhenyu Song on 10/30/18.
//

#include "simulation.h"

#include <fstream>
#include <string>
#include <regex>
#include "lru_variants.h"
#include "gd_variants.h"
#include "request.h"
#include <vector>
#include <stdio.h>
#include <sys/time.h>

using namespace std;


void annotate(string trace_file) {
    //todo: there is a risk that multiple process write a same file

    auto expect_file = trace_file+".ant";
    ifstream cachefile(expect_file);
    if (cachefile.good()) {
        cerr<<"file has been annotated, so skip annotation"<<endl;
        return;
    }


    // parse trace file
    vector<tuple<uint64_t, uint64_t , uint64_t, uint64_t >> trace;
    uint64_t t, id, size;

    ifstream infile;
    infile.open(trace_file);
    if (!infile) {
        cerr << "Exception opening/reading annotate original file";
        return;
    }
    while(infile>> t >> id >> size) {
        //default with infinite future interval
        trace.emplace_back(t, id, size, numeric_limits<uint64_t >::max()-1);
    }


    uint64_t totalReqc = trace.size();
    std::cerr << "scanned trace n=" << totalReqc << std::endl;

    // get nextSeen indices
    map<pair<uint64_t, uint64_t>, uint64_t > lastSeen;
    for (auto it = trace.rbegin(); it != trace.rend(); ++it) {
        auto lit = lastSeen.find(make_pair(get<1>(*it), get<2>(*it)));
        if (lit != lastSeen.end())
            get<3>(*it) = lit->second;
        lastSeen[make_pair(get<1>(*it), get<2>(*it))] = get<0>(*it);
    }

    // get current time
    string now;
    {
        struct timeval tp;
        gettimeofday(&tp, NULL);
        now = string(to_string(tp.tv_sec) + to_string(tp.tv_usec));
    }


    ofstream outfile;


    auto tmp_file = "/tmp/" + now;

    outfile.open(tmp_file);
    for (auto & it: trace) {
        outfile << get<0>(it) << " " << get<1>(it) << " " << get<2>(it) << " " << get<3>(it) <<endl;
    }
    
    rename(tmp_file.c_str(), expect_file.c_str());

}

map<string, string> _simulation_belady(string trace_file, string cache_type, uint64_t cache_size,
                                       map<string, string> params){
    //annotate a file
    annotate(trace_file);


    // create cache
    unique_ptr<Cache> webcache = move(Cache::create_unique(cache_type));
    if(webcache == nullptr) {
        cerr<<"cache type not implemented"<<endl;
        return {};
    }

    // configure cache size
    webcache->setSize(cache_size);

    for (auto& kv: params) {
        webcache->setPar(kv.first, kv.second);
    }

    //suppose already annotated
    ifstream infile;
    uint64_t byte_req = 0, byte_hit = 0, obj_req = 0, obj_hit = 0;
    uint64_t t, id, size, next_t;

    trace_file += ".ant";

    infile.open(trace_file);
    if (!infile) {
        cerr << "exception opening/reading file";
        return {};
    }

    AnnotatedRequest req(0, 0, 0);
    int i = 0;
    while (infile >> t >> id >> size >> next_t) {
        byte_req += size;
        obj_req++;

        req.reinit(id, size, next_t);
        if (webcache->lookup(req)) {
            byte_hit += size;
            obj_hit++;
        } else {
            webcache->admit(req);
        }
//        cout << i << " " << t << " " << obj_hit << endl;
        i++;
    }

    infile.close();

    map<string, string> res = {
            {"byte_hit_rate", to_string(double(byte_hit) / byte_req)},
            {"object_hit_rate", to_string(double(obj_hit) / obj_req)},
    };
    return res;
}

map<string, string> _simulation(string trace_file, string cache_type, uint64_t cache_size,
                                map<string, string> params){
    // create cache
    unique_ptr<Cache> webcache = move(Cache::create_unique(cache_type));
    if(webcache == nullptr) {
        cerr<<"cache type not implemented"<<endl;
        return {};
    }

    // configure cache size
    webcache->setSize(cache_size);

    for (auto& kv: params) {
        webcache->setPar(kv.first, kv.second);
    }

    ifstream infile;
    uint64_t byte_req = 0, byte_hit = 0, obj_req = 0, obj_hit = 0;
    uint64_t t, id, size;

    infile.open(trace_file);
    if (!infile) {
        cerr << "Exception opening/reading file";
        return {};
    }

    SimpleRequest req(0, 0);
    int i = 0;
    while (infile >> t >> id >> size) {
        byte_req += size;
        obj_req++;

        req.reinit(id, size);
        if (webcache->lookup(req)) {
            byte_hit += size;
            obj_hit++;
        } else {
            webcache->admit(req);
        }
//        cout << i << " " << t << " " << obj_hit << endl;
        i++;
    }

    infile.close();

    map<string, string> res = {
            {"byte_hit_rate", to_string(double(byte_hit) / byte_req)},
            {"object_hit_rate", to_string(double(obj_hit) / obj_req)},
    };
    return res;
}

map<string, string> simulation(string trace_file, string cache_type, uint64_t cache_size, map<string, string> params){
    if (cache_type == "Belady")
        return _simulation_belady(trace_file, cache_type, cache_size, params);
    else
        return _simulation(trace_file, cache_type, cache_size, params);
}
