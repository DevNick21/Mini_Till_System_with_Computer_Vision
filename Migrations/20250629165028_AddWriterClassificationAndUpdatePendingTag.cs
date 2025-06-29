using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace bet_fred.Migrations
{
    /// <inheritdoc />
    public partial class AddWriterClassificationAndUpdatePendingTag : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "HandwritingClusters");

            migrationBuilder.DeleteData(
                table: "ThresholdRules",
                keyColumn: "Id",
                keyValue: 1);

            migrationBuilder.AddColumn<decimal>(
                name: "ActualValue",
                table: "PendingTags",
                type: "TEXT",
                nullable: true);

            migrationBuilder.AddColumn<DateTime>(
                name: "CompletedAt",
                table: "PendingTags",
                type: "TEXT",
                nullable: true);

            migrationBuilder.AddColumn<bool>(
                name: "IsCompleted",
                table: "PendingTags",
                type: "INTEGER",
                nullable: false,
                defaultValue: false);

            migrationBuilder.AddColumn<bool>(
                name: "RequiresAttention",
                table: "PendingTags",
                type: "INTEGER",
                nullable: false,
                defaultValue: false);

            migrationBuilder.AddColumn<string>(
                name: "ThresholdType",
                table: "PendingTags",
                type: "TEXT",
                nullable: true);

            migrationBuilder.AddColumn<decimal>(
                name: "ThresholdValue",
                table: "PendingTags",
                type: "TEXT",
                nullable: true);

            migrationBuilder.AddColumn<int>(
                name: "WriterId",
                table: "PendingTags",
                type: "INTEGER",
                nullable: true);

            migrationBuilder.AddColumn<int>(
                name: "RuleId",
                table: "Alerts",
                type: "INTEGER",
                nullable: true);

            migrationBuilder.CreateTable(
                name: "WriterClassifications",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    BetRecordId = table.Column<int>(type: "INTEGER", nullable: false),
                    WriterId = table.Column<int>(type: "INTEGER", nullable: false),
                    Confidence = table.Column<double>(type: "REAL", nullable: false),
                    ConfidenceLevel = table.Column<string>(type: "TEXT", maxLength: 20, nullable: false),
                    CreatedAt = table.Column<DateTime>(type: "TEXT", nullable: false),
                    CustomerId = table.Column<int>(type: "INTEGER", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_WriterClassifications", x => x.Id);
                    table.ForeignKey(
                        name: "FK_WriterClassifications_BetRecords_BetRecordId",
                        column: x => x.BetRecordId,
                        principalTable: "BetRecords",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                    table.ForeignKey(
                        name: "FK_WriterClassifications_Customers_CustomerId",
                        column: x => x.CustomerId,
                        principalTable: "Customers",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.SetNull);
                });

            migrationBuilder.CreateIndex(
                name: "IX_Alerts_RuleId",
                table: "Alerts",
                column: "RuleId");

            migrationBuilder.CreateIndex(
                name: "IX_WriterClassifications_BetRecordId",
                table: "WriterClassifications",
                column: "BetRecordId");

            migrationBuilder.CreateIndex(
                name: "IX_WriterClassifications_Confidence",
                table: "WriterClassifications",
                column: "Confidence");

            migrationBuilder.CreateIndex(
                name: "IX_WriterClassifications_CustomerId",
                table: "WriterClassifications",
                column: "CustomerId");

            migrationBuilder.CreateIndex(
                name: "IX_WriterClassifications_WriterId",
                table: "WriterClassifications",
                column: "WriterId");

            migrationBuilder.AddForeignKey(
                name: "FK_Alerts_ThresholdRules_RuleId",
                table: "Alerts",
                column: "RuleId",
                principalTable: "ThresholdRules",
                principalColumn: "Id");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropForeignKey(
                name: "FK_Alerts_ThresholdRules_RuleId",
                table: "Alerts");

            migrationBuilder.DropTable(
                name: "WriterClassifications");

            migrationBuilder.DropIndex(
                name: "IX_Alerts_RuleId",
                table: "Alerts");

            migrationBuilder.DropColumn(
                name: "ActualValue",
                table: "PendingTags");

            migrationBuilder.DropColumn(
                name: "CompletedAt",
                table: "PendingTags");

            migrationBuilder.DropColumn(
                name: "IsCompleted",
                table: "PendingTags");

            migrationBuilder.DropColumn(
                name: "RequiresAttention",
                table: "PendingTags");

            migrationBuilder.DropColumn(
                name: "ThresholdType",
                table: "PendingTags");

            migrationBuilder.DropColumn(
                name: "ThresholdValue",
                table: "PendingTags");

            migrationBuilder.DropColumn(
                name: "WriterId",
                table: "PendingTags");

            migrationBuilder.DropColumn(
                name: "RuleId",
                table: "Alerts");

            migrationBuilder.CreateTable(
                name: "HandwritingClusters",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    BetRecordId = table.Column<int>(type: "INTEGER", nullable: false),
                    CustomerId = table.Column<int>(type: "INTEGER", nullable: true),
                    ClusterId = table.Column<int>(type: "INTEGER", nullable: false),
                    CreatedAt = table.Column<DateTime>(type: "TEXT", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_HandwritingClusters", x => x.Id);
                    table.ForeignKey(
                        name: "FK_HandwritingClusters_BetRecords_BetRecordId",
                        column: x => x.BetRecordId,
                        principalTable: "BetRecords",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                    table.ForeignKey(
                        name: "FK_HandwritingClusters_Customers_CustomerId",
                        column: x => x.CustomerId,
                        principalTable: "Customers",
                        principalColumn: "Id");
                });

            migrationBuilder.InsertData(
                table: "ThresholdRules",
                columns: new[] { "Id", "CustomerId", "Name", "Period", "Value" },
                values: new object[] { 1, null, "Staked in a Day", new TimeSpan(1, 0, 0, 0, 0), 500m });

            migrationBuilder.CreateIndex(
                name: "IX_HandwritingClusters_BetRecordId",
                table: "HandwritingClusters",
                column: "BetRecordId");

            migrationBuilder.CreateIndex(
                name: "IX_HandwritingClusters_CustomerId",
                table: "HandwritingClusters",
                column: "CustomerId");
        }
    }
}
