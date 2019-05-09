//
// Created by Nathan Zhang on 2019-05-07.
//

#ifndef CONSTELLATION_SPATIALEMITSUPPORT_H
#define CONSTELLATION_SPATIALEMITSUPPORT_H

#include <sstream>
#include <string>
#include <vector>
#include <regex>

#include "mlir/IR/Operation.h"

#include "constellation/core/ConstellationOps.h"
#include "constellation/core/Types.h"
#include "constellation/Spatial/CodeTemplates.h"

namespace constellation::spatial::detail {
    std::string indentString(std::string s, int indent);
    std::string typeToString(mlir::Type tp);

    template<ConstellationTypeKind CKind>
    std::string constellationTypeToString(ConstNum<CKind> num) {
        static const std::map<ConstellationTypeKind, std::string> typePrefixes = {
                {CONST_FIX, "FixPt"},
                {CONST_FLOAT, "FltPt"}
        };

        // Spatial numbers include bit in the int/exp count, so we need to increase by 1.
        bool is_signed;
        int first;
        int second;
        std::tie(is_signed, first, second) = num.getKey();
        std::string sign = is_signed ? "TRUE" : "FALSE";
        return typePrefixes.at(CKind) + "[" + sign + ", _" + std::to_string(first) + ", _" + std::to_string(second) + "]";
    }

    nlohmann::json getShape(mlir::Type tp);
}

#endif //CONSTELLATION_SPATIALEMITSUPPORT_H
