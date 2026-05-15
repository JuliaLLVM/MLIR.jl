// Copyright 2021 Google LLC
// Copyright 2023 Valentin Churavy
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <optional>

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;

enum ActionType {
  EmitOpTableDefs,
};

// defined in jl-generators.cc
extern bool emitOpTableDefs(llvm::raw_ostream &os, const llvm::RecordKeeper &recordKeeper, bool disableModuleWrap, bool isExternal, std::optional<std::string> dialectName);

cl::opt<ActionType> generator(
  "generator",
  cl::desc("Generator to run"),
  cl::values(clEnumValN(EmitOpTableDefs, "emit-op-table-defs",
                        "Emit Julia definitions for MLIR operations")),
  cl::Required
);

cl::opt<bool> disableModuleWrap(
  "disable-module-wrap",
  cl::desc("Disable module wrap"),
  cl::init(false)
);

cl::opt<bool> isExternal(
  "external",
  cl::desc("Mark the dialect as external and generate bindings accordingly"),
  cl::init(false)
);

cl::opt<std::string> dialectName(
    "dialect-name",
    llvm::cl::desc("Override the inferred dialect name, used as the name for the generated Julia module."),
    llvm::cl::value_desc("dialect")
);

#if LLVM_VERSION_MAJOR < 20
static bool MlirJuliaTablegenMain(llvm::raw_ostream &os, llvm::RecordKeeper &recordKeeper) {
#else
static bool MlirJuliaTablegenMain(llvm::raw_ostream &os, const llvm::RecordKeeper &recordKeeper) {
#endif
  switch (generator) {
    case EmitOpTableDefs: {
      std::optional<std::string> dialectNameOpt;
      if (!dialectName.empty()) dialectNameOpt = dialectName;
      return emitOpTableDefs(os, recordKeeper, disableModuleWrap, isExternal, dialectNameOpt);
      break;
    }
    default:
      llvm::errs() << "Invalid generator type\n";
      return true;
  }
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  cl::ParseCommandLineOptions(argc, argv);
#if LLVM_VERSION_MAJOR < 20
  return TableGenMain(argv[0], MlirJuliaTablegenMain);
#else
  std::function<bool(llvm::raw_ostream&, const llvm::RecordKeeper&)> mainFn = MlirJuliaTablegenMain;
  return TableGenMain(argv[0], mainFn);
#endif
}