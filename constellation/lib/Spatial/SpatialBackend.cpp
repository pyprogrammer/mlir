//
// Created by Nathan Zhang on 4/25/19.
//

#include <constellation/core/Lattice.h>
#include "constellation/Spatial/SpatialBackend.h"

namespace constellation::spatial {

    void SpatialModule::initPassManager(mlir::PassManager *pm) {
        pm->addPass(new passes::SpatialLocationPass());
        pm->addPass(new constellation::core::passes::ValueIDPass());
    }


    template<>
    void SpatialModule::codegen(std::stringstream &decl, std::stringstream &host, std::stringstream &accel,
                                IO::ReadOp readOp) {
        nlohmann::json readParams = {
                {"id",   std::to_string(core::passes::ValueIDPass::getID(readOp.getOperation()))},
                {"addr", readOp.getAttrOfType<mlir::StringAttr>("url").getValue()}
        };
        readParams["shape"] = detail::getShape(readOp.getType());
        if (auto vecOrArrayType = readOp.getType().dyn_cast<mlir::VectorOrTensorType>()) {
            readParams["typestr"] = detail::typeToString(vecOrArrayType.getElementType());
        } else {
            readParams["typestr"] = detail::typeToString(readOp.getType());
        }
        host << inja::render(templates::kReadTemplate, readParams);
    }

    template<>
    void SpatialModule::codegen(std::stringstream &decl, std::stringstream &host, std::stringstream &accel,
                                IO::TransferOp transferOp) {

        int id = core::passes::ValueIDPass::getID(transferOp.getOperation());
        int prev_id = core::passes::ValueIDPass::getID(transferOp.getOperand()->getDefiningOp());

        // create a HostIO for handshake
        {
            nlohmann::json hostIOParams = {
                    {"id", id},
                    {"T", "Int"}
            };
            decl << inja::render(templates::kHostIOTemplate, hostIOParams);
        }

        bool useDram = transferOp.getType().isa<mlir::VectorOrTensorType>();
        std::string typeString;
        // Host side create space
        {

            if (auto vecOrTensorType = transferOp.getType().dyn_cast<mlir::VectorOrTensorType>()) {
                // If input type is a vector/tensor type, use a DRAM
                typeString = detail::typeToString(vecOrTensorType.getElementType());
                nlohmann::json dramParams = {
                        {"name", "dram" + std::to_string(id)},
                        {"T", typeString},
                        {"shape", detail::getShape(vecOrTensorType)}
                };
                decl << inja::render(templates::kDRAMTemplate, dramParams);
            } else {
                // Otherwise use a HostIO

            }

        }

        Memory::Location inputLoc = transferOp.getInputMem().getLocation();
        Memory::Location outputLoc = transferOp.getOutputMem().getLocation();

        nlohmann::json hostParams = {
                {"id", id},
                {"prev_id", prev_id},
                {"useDram", useDram}
        };
        nlohmann::json accelParams = {
                {"id", id},
                {"prev_id", prev_id},
                {"useDram", useDram},
                {"T", typeString}
        };
        if (useDram) {
            accelParams["shape"] = detail::getShape(transferOp.getType());
        }

        if (inputLoc == Memory::Location::CPU && outputLoc == Memory::Location::FPGA) {
            host << inja::render(templates::kHostTransferHostToAccel, hostParams);
            accel << inja::render(templates::kAccelTransferHostToAccel, accelParams);

        } else if (inputLoc == Memory::Location::FPGA && outputLoc == Memory::Location::CPU) {
            host << inja::render(templates::kHostTransferAccelToHost, hostParams);
            accel << inja::render(templates::kAccelTransferAccelToHost, accelParams);
        } else {
            llvm_unreachable("Unsupported combination of input and output locations.");
        }

    }

    void SpatialModule::emit(llvm::raw_ostream *os) {

        module_->print(*os);

        nlohmann::json sections;
        sections["name"] = "Placeholder";

        // Create file reads
        nlohmann::json argData;
        argData["args"] = nlohmann::json::array();

        auto main = module_->getNamedFunction("main");

        for (auto arg : main->getArguments()) {
            nlohmann::json argInfo;
            mlir::Type argType = arg->getType();
            if (auto vecType = argType.dyn_cast<mlir::VectorOrTensorType>()) {
                argInfo["shape"] = nlohmann::json::array();
                for (auto dim : vecType.getShape()) {
                    argInfo["shape"].push_back(dim);
                }
                argInfo["typestr"] = detail::typeToString(vecType.getElementType());
            }
            argData["args"].push_back(argInfo);
        }

        std::stringstream decl;
        std::stringstream host;
        std::stringstream accel;

#define ConstellationOpCase(T) if (auto val = op->dyn_cast<T>()) { codegen(decl, host, accel, val); return; }
        // TODO: allow more than one function per module.
        main->walk([this, &decl, &host, &accel](mlir::Operation *op) {
            ConstellationOpCase(IO::ReadOp)
            ConstellationOpCase(IO::TransferOp)
        });
#undef ConstellationOpCase


        sections["args"] = detail::indentString(inja::render(templates::kArgTemplate, argData), 8);
        sections["host"] = detail::indentString(host.str(), 8);
        sections["decl"] = detail::indentString(decl.str(), 8);
        sections["accel"] = detail::indentString(accel.str(), 16);

        *os << inja::render(templates::kTopTemplate, sections);
    }
}
