#pragma once

#include "messageStream.H"
#include <string>

namespace Foam{

class env{

public:

    static const int REGION_DECOMPOSE_NBLOCKS;
    static const int DNN_BATCH_SIZE;
    static const std::string DFMATRIX_INNERMATRIX_TYPE;

    static bool get_bool(const std::string& name);
    static int get_int(const std::string& name);
    static const std::string& get_string(const std::string& name);

    static bool get_bool_default(const std::string& name, bool default_value);
    static int get_int_default(const std::string& name, int defualt_value);
    static const std::string& get_string_default(const std::string& name, const std::string& default_value);

    static void show();
};

}
