//
// Created by Nathan Zhang on 4/30/19.
//

#ifndef SPATIAL_CODETEMPLATES_H
#define SPATIAL_CODETEMPLATES_H

#include <string>
#include <inja.hpp>
#include <nlohmann/json.hpp>

namespace constellation::spatial::templates {

const std::string kTopTemplate(
#include "constellation/Spatial/templates/Top.inja"
);

const std::string kArgTemplate(
#include "constellation/Spatial/templates/Arg.inja"
);

const std::string kReadTemplate(
#include "constellation/Spatial/templates/Read.inja"
);

const std::string kHostIOTemplate(
#include "constellation/Spatial/templates/HostIO.inja"
);

const std::string kDRAMTemplate(
#include "constellation/Spatial/templates/DRAM.inja"
);

const std::string kHostTransferHostToAccel(
#include "constellation/Spatial/templates/HostTransferHostToAccel.inja"
);

const std::string kAccelTransferHostToAccel(
#include "constellation/Spatial/templates/AccelTransferHostToAccel.inja"
);

const std::string kAccelTransferAccelToHost(
#include "constellation/Spatial/templates/AccelTransferAccelToHost.inja"
);

const std::string kHostTransferAccelToHost(
#include "constellation/Spatial/templates/HostTransferAccelToHost.inja"
);
}

#endif //SPATIAL_CODETEMPLATES_H
