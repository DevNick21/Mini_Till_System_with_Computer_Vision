using System.IO;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Design;
using Microsoft.Extensions.Configuration;

namespace bet_fred.Data
{
    /// <summary>
    /// This factory is used by EF-CLI at design time (migrations, database update).
    /// It bypasses your Program startup and simply configures the DbContext.
    /// </summary>
    public class DesignTimeDbContextFactory
        : IDesignTimeDbContextFactory<ApplicationDbContext>
    {
        // Connection string name constant to avoid magic strings
        private const string ConnectionStringName = "DefaultConnection";

        public ApplicationDbContext CreateDbContext(string[] args)
        {
            // 1) Build config (so we can read DefaultConnection)
            var config = new ConfigurationBuilder()
                .SetBasePath(Directory.GetCurrentDirectory())
                .AddJsonFile("appsettings.json", optional: false, reloadOnChange: true)
                .AddEnvironmentVariables()
                .Build();

            // 2) Grab the connection string
            var conn = config.GetConnectionString(ConnectionStringName)
                       ?? throw new InvalidOperationException($"{ConnectionStringName} not found");

            // 3) Configure DbContextOptions to use SQLite
            var builder = new DbContextOptionsBuilder<ApplicationDbContext>();
            builder.UseSqlite(conn);

            // 4) Return the context
            return new ApplicationDbContext(builder.Options);
        }
    }
}
