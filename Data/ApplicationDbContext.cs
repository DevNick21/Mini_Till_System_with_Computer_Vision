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
        public DbSet<HandwritingCluster> HandwritingClusters { get; set; } = null!;
        public DbSet<ThresholdRule> ThresholdRules { get; set; } = null!;

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            base.OnModelCreating(modelBuilder);

            // Seed a single global rule: £5 000 max daily spend
            modelBuilder.Entity<ThresholdRule>().HasData(
                new ThresholdRule
                {
                    Id = 1,
                    Name = "Staked in a Day",
                    Value = 500m,
                    Period = TimeSpan.FromDays(1),
                    CustomerId = null
                }
            );
            modelBuilder.Entity<HandwritingCluster>()
                .HasIndex(h => h.BetRecordId)
                .IsUnique();

            modelBuilder.Entity<PendingTag>()
                .HasIndex(t => t.Tag)
                .IsUnique();
        }
    }
}
