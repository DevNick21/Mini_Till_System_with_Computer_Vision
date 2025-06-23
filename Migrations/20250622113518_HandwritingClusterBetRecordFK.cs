using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace bet_fred.Migrations
{
    /// <inheritdoc />
    public partial class HandwritingClusterBetRecordFK : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropForeignKey(
                name: "FK_HandwritingClusters_Customers_CustomerId",
                table: "HandwritingClusters");

            migrationBuilder.AlterColumn<int>(
                name: "CustomerId",
                table: "HandwritingClusters",
                type: "INTEGER",
                nullable: true,
                oldClrType: typeof(int),
                oldType: "INTEGER");

            migrationBuilder.AddColumn<int>(
                name: "BetRecordId",
                table: "HandwritingClusters",
                type: "INTEGER",
                nullable: false,
                defaultValue: 0);

            migrationBuilder.CreateIndex(
                name: "IX_HandwritingClusters_BetRecordId",
                table: "HandwritingClusters",
                column: "BetRecordId");

            migrationBuilder.AddForeignKey(
                name: "FK_HandwritingClusters_BetRecords_BetRecordId",
                table: "HandwritingClusters",
                column: "BetRecordId",
                principalTable: "BetRecords",
                principalColumn: "Id",
                onDelete: ReferentialAction.Cascade);

            migrationBuilder.AddForeignKey(
                name: "FK_HandwritingClusters_Customers_CustomerId",
                table: "HandwritingClusters",
                column: "CustomerId",
                principalTable: "Customers",
                principalColumn: "Id");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropForeignKey(
                name: "FK_HandwritingClusters_BetRecords_BetRecordId",
                table: "HandwritingClusters");

            migrationBuilder.DropForeignKey(
                name: "FK_HandwritingClusters_Customers_CustomerId",
                table: "HandwritingClusters");

            migrationBuilder.DropIndex(
                name: "IX_HandwritingClusters_BetRecordId",
                table: "HandwritingClusters");

            migrationBuilder.DropColumn(
                name: "BetRecordId",
                table: "HandwritingClusters");

            migrationBuilder.AlterColumn<int>(
                name: "CustomerId",
                table: "HandwritingClusters",
                type: "INTEGER",
                nullable: false,
                defaultValue: 0,
                oldClrType: typeof(int),
                oldType: "INTEGER",
                oldNullable: true);

            migrationBuilder.AddForeignKey(
                name: "FK_HandwritingClusters_Customers_CustomerId",
                table: "HandwritingClusters",
                column: "CustomerId",
                principalTable: "Customers",
                principalColumn: "Id",
                onDelete: ReferentialAction.Cascade);
        }
    }
}
