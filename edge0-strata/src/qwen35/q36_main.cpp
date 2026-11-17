// src/qwen35/q36_main.cpp - strata-q36: the Qwen3.6 engine path on its own (the same entry point `strata` switches
// to for a qwen35moe model).
#include "strata/qwen35/engine.hpp"

int main(int argc, char** argv) { return strata::qwen35::run(argc, argv); }
