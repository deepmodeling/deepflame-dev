#pragma once


extern int dnn_batch_size;
extern int row_block_bit;

void env_show();

bool env_get_bool(const char* name, bool default_value);

int env_get_int(const char* name, int default_value);