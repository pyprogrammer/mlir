//
// Created by Nathan Zhang on 4/30/19.
//

#include <string>

#ifndef SPATIAL_CODETEMPLATES_H
#define SPATIAL_CODETEMPLATES_H

namespace constellation {
namespace spatial {
namespace templates {
const std::string kTopTemplate(
#include "constellation/Spatial/templates/Top.inja"
);

const std::string kArgTemplate(
#include "constellation/Spatial/templates/Arg.inja"
);
}
}
}

#endif //SPATIAL_CODETEMPLATES_H
