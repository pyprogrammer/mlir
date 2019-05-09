//
// Created by Nathan Zhang on 4/19/19.
//

#ifndef CONSTELLATION_IO_H
#define CONSTELLATION_IO_H

#include <map>

#include "ConstCommon.h"
#include "OpCommon.h"
#include "Memory.h"
#include "OpUtils.h"


namespace constellation::IO {

    enum class AccessMode {
        FULL, STREAM, BACKED,

        ENUM_LAST = BACKED
    };

    class ReadOp : public mlir::Op<ReadOp, mlir::OpTrait::ZeroOperands,
            mlir::OpTrait::OneResult,
            mlir::OpTrait::HasNoSideEffect>, public HasEnumAttr<ReadOp, AccessMode, true> {
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
                    public HasEnumAttr<ReadOp, AccessMode, true> {
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
                       public HasEnumAttr<TransferOp, AccessMode, true>{

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
