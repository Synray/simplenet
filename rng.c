#include "rng.h"

static rng32_random_t rng = { 0x853c49e6748fea9bULL, 0xda3e39cb94b95bdbULL };

// rng32_srandom(initstate, initseq)
//     Seed the rng.  Specified in two parts, state initializer and a
//     sequence selection constant (a.k.a. stream id)
void rng32_srandom(uint64_t seed, uint64_t seq)
{
    rng.state = 0U;
    rng.inc = (seq << 1u) | 1u;
    rng32_random();
    rng.state += seed;
    rng32_random();
}

// rng32_random()
//     Generate a uniformly distributed 32-bit random number
uint32_t rng32_random()
{
    uint64_t oldstate = rng.state;
    rng.state = oldstate * 6364136223846793005ULL + rng.inc;
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}


// rng32_boundedrand(bound):
//     Generate a uniformly distributed number, r, where 0 <= r < bound
uint32_t rng32_boundedrand(const uint32_t bound)
{
    if (bound == 0) { return 0; }

    uint32_t threshold = -bound % bound;

    for (;;)
    {
        uint32_t r = rng32_random();
        if (r >= threshold)
        {
            return r % bound;
        }
    }
}
