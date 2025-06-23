using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace bet_fred.Migrations
{
    /// <inheritdoc />
    public partial class SeedMaxDailySpendRule : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "ClusterLabel",
                table: "HandwritingClusters");

            migrationBuilder.AddColumn<int>(
                name: "ClusterId",
                table: "HandwritingClusters",
                type: "INTEGER",
                maxLength: 200,
                nullable: false,
                defaultValue: 0);

            migrationBuilder.InsertData(
                table: "ThresholdRules",
                columns: new[] { "Id", "CustomerId", "Name", "Period", "Value" },
                values: new object[] { 1, null, "Staked in a Day", new TimeSpan(1, 0, 0, 0, 0), 500m });
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DeleteData(
                table: "ThresholdRules",
                keyColumn: "Id",
                keyValue: 1);

            migrationBuilder.DropColumn(
                name: "ClusterId",
                table: "HandwritingClusters");

            migrationBuilder.AddColumn<string>(
                name: "ClusterLabel",
                table: "HandwritingClusters",
                type: "TEXT",
                maxLength: 200,
                nullable: false,
                defaultValue: "");
        }
    }
}
