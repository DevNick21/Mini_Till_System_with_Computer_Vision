using Microsoft.EntityFrameworkCore;
using bet_fred.Models;

namespace bet_fred.Data
{
    public class ApplicationDbContext : DbContext
    {
        public ApplicationDbContext(DbContextOptions<ApplicationDbContext> options)
            : base(options) { }

        public DbSet<Customer> Customers { get; set; } = null!;
        public DbSet<BetRecord> BetRecords { get; set; } = null!;
        public DbSet<Alert> Alerts { get; set; } = null!;
        public DbSet<PendingTag> PendingTags { get; set; } = null!;
        public DbSet<ThresholdRule> ThresholdRules { get; set; } = null!;
        public DbSet<WriterClassification> WriterClassifications { get; set; } = null!;

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            base.OnModelCreating(modelBuilder);

            // Configure WriterClassification relationships
            modelBuilder.Entity<WriterClassification>()
                .HasOne(wc => wc.BetRecord)
                .WithMany()
                .HasForeignKey(wc => wc.BetRecordId)
                .OnDelete(DeleteBehavior.Cascade);

            modelBuilder.Entity<WriterClassification>()
                .HasOne(wc => wc.Customer)
                .WithMany(c => c.WriterClassifications)
                .HasForeignKey(wc => wc.CustomerId)
                .OnDelete(DeleteBehavior.SetNull); // Keep classification if customer deleted

            // Index for performance
            modelBuilder.Entity<WriterClassification>()
                .HasIndex(wc => wc.WriterId);

            modelBuilder.Entity<WriterClassification>()
                .HasIndex(wc => wc.Confidence);
        }
    }
}
