using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace bet_fred.Migrations
{
    /// <inheritdoc />
    public partial class AddOutcomeAndImageBlobToBetRecord : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<byte[]>(
                name: "ImageData",
                table: "BetRecords",
                type: "BLOB",
                nullable: false,
                defaultValue: new byte[0]);

            migrationBuilder.AddColumn<int>(
                name: "Outcome",
                table: "BetRecords",
                type: "INTEGER",
                nullable: false,
                defaultValue: 0);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "ImageData",
                table: "BetRecords");

            migrationBuilder.DropColumn(
                name: "Outcome",
                table: "BetRecords");
        }
    }
}
