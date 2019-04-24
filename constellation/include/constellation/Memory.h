//
// Created by Nathan Zhang on 4/24/19.
//

#ifndef CONSTELLATION_TYPES_H
#define CONSTELLATION_TYPES_H


#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

#include "constellation/ConstexprUtils.h"

namespace constellation {
    class Memory {
        using IndexType = unsigned;
        using MLIRMemorySpaceType = unsigned;
    public:
        enum struct Location {
            CPU,
            GPU,
            FPGA,
            LAST_TYPE = FPGA
        };

        explicit Memory(Location memoryType, IndexType index) : memoryType_{memoryType}, index_{index} {};
        explicit Memory(MLIRMemorySpaceType encoded) : memoryType_{getMemType(encoded)}, index_{getIndex(encoded)} {}

        explicit operator MLIRMemorySpaceType() const {
            // encode Location in high bits and Index in low bits
            assert(index_ < kMaxIndex && "Index value too large");
            return (static_cast<std::underlying_type<Location>::type>(memoryType_) << kShiftAmount) | index_;
        }

    private:
        static IndexType getIndex(MLIRMemorySpaceType encoded) {
            return encoded & (kMaxIndex - 1);
        }

        static Location getMemType(MLIRMemorySpaceType encoded) {
            auto underlying = encoded >> kShiftAmount;
            assert(underlying <= static_cast<std::underlying_type<Location>::type>(Location::LAST_TYPE) &&
                   "Encoded does not provide a valid Location");
            return static_cast<Location>(underlying);
        }

        Location memoryType_;
        IndexType index_;

        static constexpr auto kMemorySize =
                utils::ceillog2(static_cast<std::underlying_type<Location>::type>(Location::LAST_TYPE));
        static_assert(kMemorySize < sizeof(MLIRMemorySpaceType) * 8, "Location too large to fit in unsigned");
        static constexpr IndexType kShiftAmount = sizeof(MLIRMemorySpaceType) * 8 - kMemorySize;
        static constexpr IndexType kMaxIndex = 1 << kShiftAmount;
    };
}

#endif //CONSTELLATION_TYPES_H
