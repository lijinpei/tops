#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace llvm;

class ConfigObjectTypeInfoGenConsumer : public ASTConsumer {
  CompilerInstance &CI;

  void exportConfigObjectTypeInfo(RecordDecl *recDecl) {
    LangOptions refLangOpt;
    PrintingPolicy refPrintPolicy{refLangOpt};
    for (auto *fieldDecl : recDecl->fields()) {
      fieldDecl->printName(outs(), refPrintPolicy);
    }
  }

 public:
  ConfigObjectTypeInfoGenConsumer(CompilerInstance &CI) : CI(CI) {}

  bool HandleTopLevelDecl(DeclGroupRef D) override {
    auto &srcMgr = CI.getSourceManager();
    for (auto *decl : D) {
      auto *recDecl = dyn_cast<RecordDecl>(decl);
      if (!recDecl || !recDecl->isThisDeclarationADefinition()) {
        continue;
      }
      auto loc = decl->getLocation();
      auto fileId = srcMgr.getFileID(loc);
      if (fileId != srcMgr.getMainFileID()) {
        continue;
      }
      exportConfigObjectTypeInfo(recDecl);
    }
    return true;
  }
};

class ConfigObjectTypeInfoGenAction : public PluginASTAction {
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override {
    return std::make_unique<ConfigObjectTypeInfoGenConsumer>(CI);
  }

  bool ParseArgs(const CompilerInstance &CI,
                 const std::vector<std::string> &args) override {
    if (args.size() != 0) {
      auto &Diag = CI.getDiagnostics();
      unsigned DiagID = Diag.getDiagnosticIDs()->getCustomDiagID(
          DiagnosticIDs::Error, "coi_gen takes no argument");
      Diag.Report(DiagID);
      return false;
    }
    return true;
  }
};

static FrontendPluginRegistry::Add<ConfigObjectTypeInfoGenAction> X(
    "coi_gen", "generate type-info for config object");
