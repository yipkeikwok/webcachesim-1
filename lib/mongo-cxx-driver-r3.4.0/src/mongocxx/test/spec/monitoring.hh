// Copyright 2018-present MongoDB Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <mongocxx/client.hpp>
#include <mongocxx/config/private/prelude.hh>
#include <mongocxx/test_util/client_helpers.hh>

namespace mongocxx {
MONGOCXX_INLINE_NAMESPACE_BEGIN
namespace spec {

using namespace mongocxx;

// Stores and compares apm events.
class apm_checker {
   public:
    options::apm get_apm_opts();
    void compare(bsoncxx::array::view expected,
                 bool allow_extra = false,
                 const test_util::match_visitor& match_visitor = {});

   private:
    std::vector<bsoncxx::document::value> _events;
};

}  // namespace spec
MONGOCXX_INLINE_NAMESPACE_END
}  // namespace mongocxx
#include <mongocxx/config/private/postlude.hh>