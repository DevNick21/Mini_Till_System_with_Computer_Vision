using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace bet_fred.Migrations
{
    /// <inheritdoc />
    public partial class RemoveBetTypeFromBetRecords : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "BetType",
                table: "BetRecords");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<string>(
                name: "BetType",
                table: "BetRecords",
                type: "TEXT",
                maxLength: 50,
                nullable: false,
                defaultValue: "");
        }
    }
}
