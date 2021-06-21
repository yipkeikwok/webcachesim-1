#ifndef PARAM_HELPER_H
#define PARAM_HELPER_H

#include <string>
#include <unordered_map>

inline std::string param_map_to_string(std::unordered_map<std::string, std::string> map)
{
	std::string params;
	for(auto p : map)
	{
		params.append(p.first);
		params.append("=");
		params.append(p.second);
		params.append(" "); // may need to be a space not comma
	}

	return params;
}

#endif
