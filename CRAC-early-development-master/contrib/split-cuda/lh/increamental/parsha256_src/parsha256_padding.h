//
// Created by neville on 03.11.20.
//

#include <string>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// Padding is performed on cpu

#ifndef PARSHAONGPU_PADDING_H
#define PARSHAONGPU_PADDING_H

// return vector is currently passed by value, could be optimized
std::vector<int> parsha256_padding(const std::string &in, const int added_zeros_bits) {


    const int add_zeros_chars = added_zeros_bits / 8; // How many zeros to add in uint8
    // Padding is always performed!

    const int newlength = in.length() + add_zeros_chars; // new length in uint8

    // Somehow the calculation to find out how much to pad is wrong, so I still some additional padding.
    std::vector<int> out(newlength / 4 + 0);

    memcpy(out.data(), in.data(), in.length() * sizeof(char)); // copy existing data

    auto *start_point = (uint8_t *) out.data();

    // Fill zeros
    for (unsigned int i = in.length(); i < newlength; i++) {
        start_point[i] = 0;
    }

    return out;

}

void parsha256_padding_test() {

}

#endif //PARSHAONGPU_PADDING_H
