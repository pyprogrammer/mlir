//
// Created by Nathan Zhang on 4/25/19.
//

#include "constellation/Spatial/SpatialBackend.h"

namespace constellation::spatial {

    void SpatialModule::initPassManager(mlir::PassManager *pm) {
        pm->addPass(new passes::SpatialLocationPass());
        pm->addPass(new core::passes::ValueIDPass());
    }

    void SpatialModule::genericCodegen(std::stringstream &decl, std::stringstream &host,
            std::stringstream &accel, mlir::Operation* op) {
        Memory mem = passes::SpatialLocationPass::getMemory(op);
        std::stringstream* stream;
        switch(mem.getLocation()) {
            case Memory::Location::CPU:
                stream = &host;
                break;
            case Memory::Location::FPGA:
                stream = &accel;
                break;
            default:
                llvm_unreachable("Invalid location.");
        }
        nlohmann::json callParams = {
                {"id", core::passes::ValueIDPass::getID(op)},
                {"name", op->getName().getStringRef().str()}
        };
        auto args = nlohmann::json::array();
        for (auto operand: op->getOperands()) {
            args.push_back("v" + std::to_string(core::passes::ValueIDPass::getID(operand->getDefiningOp())));
        }

        callParams["args"] = args;

        auto metaparams = nlohmann::json::array();

        // Need a way for ops to add extra parameters.
        for (const auto attr: op->getAttrs()) {
            if (attr.first.str().find(constellation::HasAttrBase::kCodeGenPrefix) == 0) {
                if (auto intAttr = attr.second.dyn_cast<mlir::IntegerAttr>()) {
                    metaparams.push_back(intAttr.getInt());
                }
            }
        }
        callParams["params"] = metaparams;
        *stream << inja::render(templates::kGenericCallTemplate, callParams);
    }

    template<>
    void SpatialModule::codegen(std::stringstream &decl, std::stringstream &host, std::stringstream &accel,
                                mlir::ReturnOp returnOp) {
        Memory mem = passes::SpatialLocationPass::getMemory(returnOp);
        std::stringstream* stream;
        switch(mem.getLocation()) {
            case Memory::Location::CPU:
                stream = &host;
                break;
            case Memory::Location::FPGA:
                stream = &accel;
                break;
            default:
                llvm_unreachable("Invalid location.");
        }
        *stream << "return\n";
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
        int prev_id = core::passes::ValueIDPass::getID(transferOp.getOperand());

        // create a HostIO for handshake
        {
            nlohmann::json hostIOParams = {
                    {"name", "io" + std::to_string(id)},
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
                typeString = detail::typeToString(transferOp.getType());
                // Otherwise use a HostIO
                nlohmann::json hostIOParams = {
                        {"name", "valIO" + std::to_string(id)},
                        {"T", typeString}
                };
                decl << inja::render(templates::kHostIOTemplate, hostIOParams);
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
            ConstellationOpCase(mlir::ReturnOp)
            genericCodegen(decl, host, accel, op);
        });
#undef ConstellationOpCase


        sections["args"] = detail::indentString(inja::render(templates::kArgTemplate, argData), 8);
        sections["host"] = detail::indentString(host.str(), 8);
        sections["decl"] = detail::indentString(decl.str(), 8);
        sections["accel"] = detail::indentString(accel.str(), 16);

        *os << inja::render(templates::kTopTemplate, sections);
    }
}
