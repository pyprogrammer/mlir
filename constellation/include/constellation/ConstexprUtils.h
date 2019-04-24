//
// Created by Nathan Zhang on 4/24/19.
//

#ifndef CONSTELLATION_CONSTEXPRUTILS_H
#define CONSTELLATION_CONSTEXPRUTILS_H

namespace constellation {
    namespace utils {
        constexpr unsigned floorlog2(unsigned x)
        {
            return x == 1 ? 0 : 1+floorlog2(x >> 1);
        }

        constexpr unsigned ceillog2(unsigned x)
        {
            return x == 1 ? 0 : floorlog2(x - 1) + 1;
        }
    }
}

#endif //CONSTELLATION_CONSTEXPRUTILS_H
