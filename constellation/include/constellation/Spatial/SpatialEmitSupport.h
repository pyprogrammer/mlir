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
    std::string constellationTypeToString(ConstNum<CKind> num);

    nlohmann::json getShape(mlir::Type tp);
}

#endif //CONSTELLATION_SPATIALEMITSUPPORT_H
