//
// Created by Nathan Zhang on 4/19/19.
//

#ifndef CONSTELLATION_IO_H
#define CONSTELLATION_IO_H

#include <map>

#include "ConstCommon.h"
#include "OpCommon.h"
#include "Memory.h"


namespace constellation::IO {

    enum class AccessMode {
        FULL, STREAM, BACKED,

        ENUM_LAST = BACKED
    };

    namespace internal {
        template<typename T, typename E, int attr_id=0>
        class HasEnumAttr {
        public:
            template<int i=0, typename = typename std::enable_if<i == attr_id>::type >
            E getEnumAttr() {
                return fromIntegerAttr<E>(
                        static_cast<T *>(this)->template getAttrOfType<mlir::IntegerAttr>(getAttrName()));
            }

        protected:
            template<int i=0, typename = typename std::enable_if<i == attr_id>::type >
            static void setEnumAttr(mlir::Builder *b, mlir::OperationState *result, E val) {
                result->addAttribute(getAttrName(), toIntegerAttr(b, val));
            }
        private:
            static std::string getAttrName() {
                return "_EnumAttr" + std::to_string(attr_id);
            }
        };
    }

    class ReadOp : public mlir::Op<ReadOp, mlir::OpTrait::ZeroOperands,
            mlir::OpTrait::OneResult,
            mlir::OpTrait::HasNoSideEffect>, public internal::HasEnumAttr<ReadOp, AccessMode> {
    public:

        using Op::Op;

        //////////////////////////////////////////////////////////////////////////////
        // Hooks to customize the behavior of this op.
        //////////////////////////////////////////////////////////////////////////////
        static llvm::StringRef getOperationName() { return "constellation.read"; }

        static void build(mlir::Builder *b, mlir::OperationState *state, std::string url,
                          AccessMode accessMode, mlir::Type resultType);

        mlir::LogicalResult verify();

        //////////////////////////////////////////////////////////////////////////////
        // Op-specific functionality.
        //////////////////////////////////////////////////////////////////////////////
    };

    class WriteOp : public mlir::Op<WriteOp, mlir::OpTrait::OneOperand, mlir::OpTrait::OneResult>,
                    public internal::HasEnumAttr<ReadOp, AccessMode> {
    public:

        using Op::Op;

        //////////////////////////////////////////////////////////////////////////////
        // Hooks to customize the behavior of this op.
        //////////////////////////////////////////////////////////////////////////////
        static llvm::StringRef getOperationName() { return "constellation.write"; }

        static void build(mlir::Builder *b, mlir::OperationState *state, std::string url,
                          AccessMode accessMode, mlir::Value *data);

        mlir::LogicalResult verify();
    };

    class TransferOp : public mlir::Op<TransferOp, mlir::OpTrait::OneOperand, mlir::OpTrait::OneResult>,
                       public internal::HasEnumAttr<TransferOp, AccessMode>{

    public:
        using Op::Op;

        static llvm::StringRef getOperationName() { return "constellation.transfer"; }
        static void build(mlir::Builder *b, mlir::OperationState *state, AccessMode accessMode, mlir::Value* data,
                Memory input, Memory output);

        mlir::LogicalResult verify();

        static mlir::Type getMemorySpaceType(mlir::Builder* b) {
            return b->getIntegerType(sizeof(Memory::MLIRMemorySpaceType)*8);
        }

        Memory getInputMem();
        Memory getOutputMem();
    };
}


#endif //CONSTELLATION_IO_H
