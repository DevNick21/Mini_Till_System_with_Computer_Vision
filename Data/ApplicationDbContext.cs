using Microsoft.EntityFrameworkCore;
using bet_fred.Models;

namespace bet_fred.Data
{
    public class ApplicationDbContext : DbContext
    {
        public ApplicationDbContext(DbContextOptions<ApplicationDbContext> options) : base(options)
        {
        }

        public DbSet<Customer> Customers { get; set; }
        public DbSet<BetRecord> BetRecords { get; set; }
        public DbSet<Alert> Alerts { get; set; }
        public DbSet<ThresholdRule> ThresholdRules { get; set; }

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            base.OnModelCreating(modelBuilder);

            // Configure relationships
            modelBuilder.Entity<BetRecord>()
                .HasOne(b => b.Customer)
                .WithMany(c => c.BetRecords)
                .HasForeignKey(b => b.CustomerId);

            modelBuilder.Entity<Alert>()
                .HasOne(a => a.Customer)
                .WithMany(c => c.Alerts)
                .HasForeignKey(a => a.CustomerId);

            // Configure decimal precision for amounts
            modelBuilder.Entity<BetRecord>()
                .Property(b => b.Amount)
                .HasColumnType("decimal(18,2)");

            // OCR suggestions removed
        }
    }
}
