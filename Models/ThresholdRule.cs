using System.ComponentModel.DataAnnotations;

namespace bet_fred.Models
{
    /// <summary>
    /// Configurable business rules for fraud detection threshold monitoring
    /// </summary>
    public class ThresholdRule
    {
        public int Id { get; set; }

        [Required]
        [StringLength(100)]
        public string Name { get; set; } = string.Empty;

        [StringLength(500)]
        public string Description { get; set; } = string.Empty;

        /// <summary>
        /// The threshold value to monitor against
        /// </summary>
        public decimal Value { get; set; }

        /// <summary>
        /// Time window in minutes for the rule evaluation
        /// </summary>
        public int TimeWindowMinutes { get; set; } = 1440; // 24 hours default

        /// <summary>
        /// Whether this rule is currently active
        /// </summary>
        public bool IsActive { get; set; } = true;

        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

        /// <summary>
        /// Type of threshold rule (DailyStake, DailyLoss, DailyBetCount, etc.)
        /// </summary>
        [StringLength(50)]
        public string RuleType { get; set; } = string.Empty;
    }
}
