//
// Created by Nathan Zhang on 4/25/19.
//

#include "constellation/Spatial/SpatialBackend.h"

namespace constellation {
    namespace spatial {
        const std::string kTopTemplate(
#include "constellation/Spatial/templates/Top.scala"
                );

        void SpatialFunc::emit(llvm::raw_ostream *os) {
            nlohmann::json sections = {
                    {"preAccel", "pre"},
                    {"accel", "inside"},
                    {"postAccel", "post"}
            };
            sections["name"] = func_->getName().str();
            *os << inja::render(kTopTemplate, sections);
        }
    }
}
