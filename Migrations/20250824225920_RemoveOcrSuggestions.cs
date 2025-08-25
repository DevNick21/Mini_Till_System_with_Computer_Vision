using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace bet_fred.Migrations
{
    /// <inheritdoc />
    public partial class RemoveOcrSuggestions : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "OcrSuggestions");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "OcrSuggestions",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    BetRecordId = table.Column<int>(type: "INTEGER", nullable: true),
                    Accepted = table.Column<bool>(type: "INTEGER", nullable: true),
                    AcceptedAt = table.Column<DateTime>(type: "TEXT", nullable: true),
                    CreatedAt = table.Column<DateTime>(type: "TEXT", nullable: false),
                    Currency = table.Column<string>(type: "TEXT", maxLength: 8, nullable: true),
                    FileHash = table.Column<string>(type: "TEXT", maxLength: 128, nullable: true),
                    FileName = table.Column<string>(type: "TEXT", maxLength: 256, nullable: true),
                    FileSize = table.Column<long>(type: "INTEGER", nullable: true),
                    Method = table.Column<string>(type: "TEXT", maxLength: 64, nullable: true),
                    Stake = table.Column<decimal>(type: "decimal(18,2)", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_OcrSuggestions", x => x.Id);
                    table.ForeignKey(
                        name: "FK_OcrSuggestions_BetRecords_BetRecordId",
                        column: x => x.BetRecordId,
                        principalTable: "BetRecords",
                        principalColumn: "Id");
                });

            migrationBuilder.CreateIndex(
                name: "IX_OcrSuggestions_BetRecordId",
                table: "OcrSuggestions",
                column: "BetRecordId");
        }
    }
}
