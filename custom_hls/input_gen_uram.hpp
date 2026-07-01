/****************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @author  Thomas B. Preußer <thomas.preusser@amd.com>
 ****************************************************************************/
#ifndef FINN_INPUT_GEN_URAM_HPP
#define FINN_INPUT_GEN_URAM_HPP

#include <ap_int.h>
#include <hls_stream.h>
#include "utils.hpp"

#include <algorithm>
#include <tuple>
#include <type_traits>

template<bool R, unsigned... V>
class UramNest {};

template<
    bool      R,
    unsigned  W
>
class UramNest<R, W> {
public:
    static constexpr unsigned  rp_rewind = 0;
    static constexpr unsigned  fp_rewind = 0;

    static constexpr int  max_rp_retract = 0;

public:
    std::tuple<int, unsigned, ap_int<1>> tick() {
#pragma HLS inline
        return  { W, R? W : 0, -1 };
    }
};

template<
    bool      R,
    unsigned  W,
    unsigned  N,
    unsigned  C,
    unsigned... V
>
class UramNest<R, W, N, C, V...> {

    static constexpr bool  R_INNER = R && (0 < C) && (C*N <= W);
    using  Inner = UramNest<R_INNER, C, V...>;

public:
    static constexpr unsigned  rp_rewind = (N-1)*C + Inner::rp_rewind;
    static constexpr unsigned  fp_rewind = R_INNER? (N-1)*C + Inner::fp_rewind : 0;

private:
    static constexpr int  terminal_rp_inc = W - rp_rewind;
public:
    static constexpr int  max_rp_retract = std::max(-terminal_rp_inc, Inner::max_rp_retract);

private:
    static_assert(N > 0, "Must have positive iteration count.");
    ap_int<1+clog2(std::max(1u, N-1))>  cnt = N-2;
    Inner  inner;

public:
    std::tuple<int, unsigned, ap_int<2+sizeof...(V)/2>> tick() {
#pragma HLS inline
        auto const  t = inner.tick();
        int       rp_inc = std::get<0>(t);
        unsigned  fp_inc = std::get<1>(t);
        ap_int<2+sizeof...(V)/2>  term = std::get<2>(t);

        if(term < 0) {
            if(cnt < 0) {
                rp_inc = terminal_rp_inc;
                if(R)  fp_inc = W - fp_rewind;
                cnt = N-2;
            }
            else {
                term[decltype(term)::width-1] = 0;
                cnt--;
            }
        }
        return { rp_inc, fp_inc, term };
    }
};

template<int  M, unsigned... V, typename  T>
void input_gen_uram(
    hls::stream<T> &src,
    hls::stream<typename std::conditional<M < 0, T, flit_t<T>>::type> &dst
) {
#pragma HLS pipeline II=1 style=flp

    constexpr unsigned  WP_DELAY = 4;

    using  MyNest = UramNest<true, V...>;
    constexpr unsigned  ADDR_BITS = clog2(MyNest::max_rp_retract + WP_DELAY + 2);
    constexpr unsigned  BUF_SIZE  = 1 << ADDR_BITS;
    using  ptr_t = ap_int<1 + ADDR_BITS>;

    static MyNest  nest;
    static T  buf[BUF_SIZE];
    static ptr_t  wp[WP_DELAY] = { 0, };
    static ptr_t  rp = 0;
    static ptr_t  fp = 0;
#pragma HLS reset variable=nest
#pragma HLS reset variable=buf off
#pragma HLS reset variable=wp
#pragma HLS reset variable=rp
#pragma HLS reset variable=fp
#pragma HLS BIND_STORAGE variable=buf type=RAM_S2P impl=URAM
#pragma HLS dependence variable=buf inter false
#pragma HLS dependence variable=buf intra false
#pragma HLS array_partition variable=wp complete

    static bool  ovld = false;
    static struct OBuf {
        bool  lst;
        T     dat;

    public:
        operator T const&()  const { return  dat; }
        operator flit_t<T>() const { return { lst, dat }; }
    } obuf;
#pragma HLS reset variable=ovld
#pragma HLS reset variable=obuf off

    for(unsigned  i = WP_DELAY-1; i > 0; i--)  wp[i] = wp[i-1];

    if(ptr_t(wp[0]-fp) >= 0) {
        T  x;
        if(src.read_nb(x))  buf[ap_uint<ADDR_BITS>(wp[0]++)] = x;
    }

    if(ovld)  ovld = !dst.write_nb(obuf);

    if(!ovld) {
        obuf.dat = buf[ap_uint<ADDR_BITS>(rp)];

        if(ptr_t(rp-wp[WP_DELAY-1]) < 0) {
            auto const  t = nest.tick();
            rp += std::get<0>(t);
            fp += std::get<1>(t);

            if(M >= 0)  obuf.lst = std::get<2>(t)[M];
            ovld = true;
        }
    }

}

#endif
