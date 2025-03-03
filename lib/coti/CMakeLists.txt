add_library(coti utils.cpp)
target_link_libraries(coti PUBLIC nlohmann_json::nlohmann_json OpenSSL::SSL fmt::fmt)
