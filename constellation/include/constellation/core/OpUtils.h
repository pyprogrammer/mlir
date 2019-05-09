//
// Created by Nathan Zhang on 2019-05-09.
//

#ifndef CONSTELLATION_OPUTILS_H
#define CONSTELLATION_OPUTILS_H

#include <string>
#include <type_traits>
#include "constellation/core/EnumUtils.h"

namespace constellation {

    class HasAttrBase {
    public:
        const static std::string kCodeGenPrefix;
    };

    template<typename T, typename E, bool codegen = false, int attr_id = 0>
    class HasEnumAttr : HasAttrBase {
    public:
        template<int i = 0, typename = typename std::enable_if<i == attr_id>::type>
        E getEnumAttr() {
            return fromIntegerAttr<E>(
                    static_cast<T *>(this)->template getAttrOfType<mlir::IntegerAttr>(getAttrName()));
        }

    protected:
        template<int i = 0, typename = typename std::enable_if<i == attr_id>::type>
        static void setEnumAttr(mlir::Builder *b, mlir::OperationState *result, E val) {
            result->addAttribute(getAttrName(), toIntegerAttr(b, val));
        }

    private:
        static std::string getAttrName() {
            auto name = "_EnumAttr" + std::to_string(attr_id);
            if (codegen) {
                return kCodeGenPrefix + name;
            }
            return name;
        }
    };
}

#endif //CONSTELLATION_OPUTILS_H
