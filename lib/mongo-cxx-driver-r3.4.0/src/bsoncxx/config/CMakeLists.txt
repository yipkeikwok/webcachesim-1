# Copyright 2016 MongoDB Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/config.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/config.hpp
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/private/config.hh.in
    ${CMAKE_CURRENT_BINARY_DIR}/private/config.hh
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/version.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/version.hpp
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/config.hpp
    ${CMAKE_CURRENT_BINARY_DIR}/version.hpp
    DESTINATION ${BSONCXX_HEADER_INSTALL_DIR}/bsoncxx/config
    COMPONENT dev
)

if (BUILD_SHARED_LIBS)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/libbsoncxx.pc.in
        ${CMAKE_CURRENT_BINARY_DIR}/libbsoncxx.pc
        @ONLY
    )

    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/libbsoncxx.pc"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
        COMPONENT dev
    )
else()
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/libbsoncxx-static.pc.in
        ${CMAKE_CURRENT_BINARY_DIR}/libbsoncxx-static.pc
        @ONLY
    )

    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/libbsoncxx-static.pc"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
        COMPONENT dev
    )
endif()
