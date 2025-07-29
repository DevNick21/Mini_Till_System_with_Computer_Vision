using System.ComponentModel.DataAnnotations;

namespace bet_fred.Models
{
    /// <summary>
    /// Customer entity for the fraud detection system.
    /// Linked to betting records through handwriting analysis classification.
    /// </summary>
    public class Customer
    {
        public int Id { get; set; }

        [Required]
        [StringLength(100)]
        public string Name { get; set; } = string.Empty;

        [EmailAddress]
        [StringLength(150)]
        public string Email { get; set; } = string.Empty;

        [Phone]
        [StringLength(20)]
        public string Phone { get; set; } = string.Empty;

        [StringLength(200)]
        public string Address { get; set; } = string.Empty;

        /// <summary>
        /// Maximum betting limit for fraud detection
        /// </summary>
        public decimal BetLimit { get; set; } = 1000;

        /// <summary>
        /// Risk assessment level (Low, Medium, High)
        /// </summary>
        [StringLength(20)]
        public string RiskLevel { get; set; } = "Low";

        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

        // Navigation properties
        public ICollection<BetRecord> BetRecords { get; set; } = new List<BetRecord>();
        public ICollection<Alert> Alerts { get; set; } = new List<Alert>();

        /// <summary>
        /// Total number of bets placed by this customer
        /// </summary>
        public int TotalBets => BetRecords.Count;

        /// <summary>
        /// Total amount bet by this customer
        /// </summary>
        public decimal TotalStake => BetRecords.Sum(b => b.Amount);
    }
}
