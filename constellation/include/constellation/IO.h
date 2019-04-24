//
// Created by Nathan Zhang on 4/19/19.
//

#ifndef CONSTELLATION_IO_H
#define CONSTELLATION_IO_H

#include <map>

#include "constellation/ConstCommon.h"
#include "constellation/OpCommon.h"


namespace constellation {

    namespace IO {

        enum class AccessMode {
            FULL, STREAM, BACKED,

            ENUM_LAST = BACKED
        };

        namespace internal {
            template<typename T>
            class HasAccessMode {
            public:
                AccessMode getAccessMode() {
                    return fromIntegerAttr<AccessMode>(
                            static_cast<T *>(this)->template getAttrOfType<mlir::IntegerAttr>("mode"));
                }

            protected:
                static void setAccessMode(mlir::Builder *b, mlir::OperationState *result, AccessMode accessMode) {
                    result->addAttribute("mode", toIntegerAttr(b, accessMode));
                }
            };
        }

        class ReadOp : public mlir::Op<ReadOp, mlir::OpTrait::ZeroOperands,
                mlir::OpTrait::OneResult,
                mlir::OpTrait::HasNoSideEffect>, public internal::HasAccessMode<ReadOp> {
        public:

            using Op::Op;

            //////////////////////////////////////////////////////////////////////////////
            // Hooks to customize the behavior of this op.
            //////////////////////////////////////////////////////////////////////////////
            static llvm::StringRef getOperationName() { return "constellation.read"; }

            static void build(mlir::Builder *b, mlir::OperationState *state, std::string url,
                              IO::AccessMode accessMode, mlir::Type resultType);

            mlir::LogicalResult verify();

//            static bool parse(mlir::OpAsmParser *parser, mlir::OperationState *result);

            //////////////////////////////////////////////////////////////////////////////
            // Op-specific functionality.
            //////////////////////////////////////////////////////////////////////////////
        };

        class WriteOp : public mlir::Op<WriteOp, mlir::OpTrait::OneOperand, mlir::OpTrait::OneResult>,
                        public internal::HasAccessMode<WriteOp> {
        public:

            using Op::Op;

            //////////////////////////////////////////////////////////////////////////////
            // Hooks to customize the behavior of this op.
            //////////////////////////////////////////////////////////////////////////////
            static llvm::StringRef getOperationName() { return "constellation.write"; }

            static void build(mlir::Builder *b, mlir::OperationState *state, std::string url,
                              IO::AccessMode accessMode, mlir::Value *data);

            mlir::LogicalResult verify();

//        static bool parse(mlir::OpAsmParser *parser, mlir::OperationState *result);
        };
    }
}


#endif //CONSTELLATION_IO_H
