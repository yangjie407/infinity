//
// Created by JinHai on 2022/7/23.
//

#pragma once


#include "logical_operator.h"
#include "SQLParserResult.h"
#include "sql/CreateStatement.h"
#include "sql/DropStatement.h"
#include "sql/InsertStatement.h"
#include "sql/DeleteStatement.h"
#include "sql/UpdateStatement.h"
#include "sql/SelectStatement.h"
#include "sql/ShowStatement.h"
#include "sql/ImportStatement.h"
#include "sql/ExportStatement.h"
#include "sql/TransactionStatement.h"
#include "sql/AlterStatement.h"
#include "sql/PrepareStatement.h"
#include "sql/ExecuteStatement.h"

#include "bind_context.h"

#include "storage/data_type.h"

#include "expression/base_expression.h"

namespace infinity {

class Planner {
public:
    static LogicalType TypeConversion(hsql::ColumnType type);

    std::shared_ptr<LogicalOperator> CreateLogicalOperator(const hsql::SQLStatement &statement);
private:
    // Create operator
    std::shared_ptr<LogicalOperator> BuildCreate(const hsql::CreateStatement& statement);
    std::shared_ptr<LogicalOperator> BuildCreateTable(const hsql::CreateStatement& statement);
    std::shared_ptr<LogicalOperator> BuildCreateTableFromTable(const hsql::CreateStatement& statement);
    std::shared_ptr<LogicalOperator> BuildCreateView(const hsql::CreateStatement& statement);
    std::shared_ptr<LogicalOperator> BuildCreateIndex(const hsql::CreateStatement& statement);

    // Drop operator
    std::shared_ptr<LogicalOperator> BuildDrop(const hsql::DropStatement& statement);
    std::shared_ptr<LogicalOperator> BuildDropTable(const hsql::DropStatement& statement);
    std::shared_ptr<LogicalOperator> BuildDropSchema(const hsql::DropStatement& statement);
    std::shared_ptr<LogicalOperator> BuildDropIndex(const hsql::DropStatement& statement);
    std::shared_ptr<LogicalOperator> BuildDropView(const hsql::DropStatement& statement);
    std::shared_ptr<LogicalOperator> BuildDropPreparedStatement(const hsql::DropStatement& statement);

    // Insert operator
    std::shared_ptr<LogicalOperator> BuildInsert(const hsql::InsertStatement& statement);
    std::shared_ptr<LogicalOperator> BuildInsertValue(const hsql::InsertStatement& statement);
    std::shared_ptr<LogicalOperator> BuildInsertSelect(const hsql::InsertStatement& statement);

    // Delete operator
    std::shared_ptr<LogicalOperator> BuildDelete(const hsql::DeleteStatement& statement);

    // Update operator
    std::shared_ptr<LogicalOperator> BuildUpdate(const hsql::UpdateStatement& statement);

    // Select operator
    std::shared_ptr<LogicalOperator> BuildSelect(const hsql::SelectStatement& statement, const std::shared_ptr<BindContext>& bind_context_ptr);

    // Show operator
    std::shared_ptr<LogicalOperator> BuildShow(const hsql::ShowStatement& statement);
    std::shared_ptr<LogicalOperator> BuildShowColumns(const hsql::ShowStatement& statement);
    std::shared_ptr<LogicalOperator> BuildShowTables(const hsql::ShowStatement& statement);

    // Import operator
    std::shared_ptr<LogicalOperator> BuildImport(const hsql::ImportStatement& statement);
    std::shared_ptr<LogicalOperator> BuildImportCsv(const hsql::ImportStatement& statement);
    std::shared_ptr<LogicalOperator> BuildImportTbl(const hsql::ImportStatement& statement);
    std::shared_ptr<LogicalOperator> BuildImportBinary(const hsql::ImportStatement& statement);
    std::shared_ptr<LogicalOperator> BuildImportAuto(const hsql::ImportStatement& statement);

    // Export operator
    std::shared_ptr<LogicalOperator> BuildExport(const hsql::ExportStatement& statement);
    std::shared_ptr<LogicalOperator> BuildExportCsv(const hsql::ExportStatement& statement);
    std::shared_ptr<LogicalOperator> BuildExportTbl(const hsql::ExportStatement& statement);
    std::shared_ptr<LogicalOperator> BuildExportBinary(const hsql::ExportStatement& statement);
    std::shared_ptr<LogicalOperator> BuildExportAuto(const hsql::ExportStatement& statement);

    // Transaction operator
    std::shared_ptr<LogicalOperator> BuildTransaction(const hsql::TransactionStatement& statement);
    std::shared_ptr<LogicalOperator> BuildTransactionBegin(const hsql::TransactionStatement& statement);
    std::shared_ptr<LogicalOperator> BuildTransactionCommit(const hsql::TransactionStatement& statement);
    std::shared_ptr<LogicalOperator> BuildTransactionRollback(const hsql::TransactionStatement& statement);

    // Alter operator
    std::shared_ptr<LogicalOperator> BuildAlter(const hsql::AlterStatement& statement);
    std::shared_ptr<LogicalOperator> BuildAlterDropColumn(const hsql::AlterStatement& statement);

    // Prepare operator
    std::shared_ptr<LogicalOperator> BuildPrepare(const hsql::PrepareStatement& statement);

    // Execute operator
    std::shared_ptr<LogicalOperator> BuildExecute(const hsql::ExecuteStatement& statement);


    // Expression
    std::shared_ptr<BaseExpression>
    BuildExpression(const hsql::Expr& expr, const std::shared_ptr<BindContext>& bind_context_ptr);

    // Build From clause
    std::shared_ptr<LogicalOperator>
    BuildFromClause(const hsql::TableRef* fromTable, const std::shared_ptr<BindContext>& bind_context_ptr);

    void
    BuildSelectList(const std::vector<hsql::Expr*>& select_list, const std::shared_ptr<BindContext>& bind_context_ptr);

    std::shared_ptr<LogicalOperator>
    BuildFilter(const hsql::Expr* where_clause, const std::shared_ptr<BindContext>& bind_context_ptr);

    std::shared_ptr<LogicalOperator>
    BuildGroupByHaving(
            const hsql::SelectStatement& select,
            const std::shared_ptr<BindContext>& bind_context_ptr,
            const std::shared_ptr<LogicalOperator>& root_operator);

    std::shared_ptr<LogicalOperator>
    BuildOrderBy(const std::vector<hsql::OrderDescription*>& order_by_clause, const std::shared_ptr<BindContext>& bind_context_ptr);

    std::shared_ptr<LogicalOperator>
    BuildLimit(const hsql::LimitDescription& limit_description, const std::shared_ptr<BindContext>& bind_context_ptr);

    std::shared_ptr<LogicalOperator>
    BuildTop(const std::vector<hsql::OrderDescription*>& order_by_clause,
             const hsql::LimitDescription& limit_description,
             const std::shared_ptr<BindContext>& bind_context_ptr);

    std::shared_ptr<LogicalOperator>
    BuildTable(const hsql::TableRef* from_table, const std::shared_ptr<BindContext>& bind_context_ptr);
private:
    // All operators
    std::vector<LogicalOperator> operators_;

    // Bind Contexts
    std::vector<std::shared_ptr<BindContext>> bind_contexts_;
    std::shared_ptr<BindContext> current_bind_context_ptr_;
};

}