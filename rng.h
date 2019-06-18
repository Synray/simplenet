#ifndef RNG_H
#define RNG_H

#include <inttypes.h>

struct rng_state_setseq_64
{
    uint64_t state;
    uint64_t inc;               // Controls which RNG sequence (stream) is selected. Must be odd.
};
typedef struct rng_state_setseq_64 rng32_random_t;

// rng32_srandom(initstate, initseq)
// rng32_srandom_r(rng, initstate, initseq):
//     Seed the rng.  Specified in two parts, state initializer and a
//     sequence selection constant (a.k.a. stream id)

void rng32_srandom(const uint64_t initstate, const uint64_t initseq);

// rng32_random()
// rng32_random_r(rng)
//     Generate a uniformly distributed 32-bit random number

uint32_t rng32_random(void);

// rng32_boundedrand(bound):
// rng32_boundedrand_r(rng, bound):
//     Generate a uniformly distributed number, r, where 0 <= r < bound

uint32_t rng32_boundedrand(const uint32_t bound);

#endif /*RNG_H */
