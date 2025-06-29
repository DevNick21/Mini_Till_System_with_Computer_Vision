using System;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace bet_fred.Models
{
    public class PendingTag
    {
        [Key]
        public int Id { get; set; }

        [Required, StringLength(100)]
        public string Tag { get; set; } = string.Empty;

        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

        // Link to the bet record this tag is pending for
        public int BetRecordId { get; set; }
        [ForeignKey(nameof(BetRecordId))]
        public BetRecord BetRecord { get; set; } = null!;

        // Classification details
        public int? WriterId { get; set; }              // Writer ID from classification
        public string? ThresholdType { get; set; }      // "DailyStake", "DailyLoss", etc.
        public decimal? ThresholdValue { get; set; }    // Threshold that was exceeded
        public decimal? ActualValue { get; set; }       // Actual value that exceeded threshold
        public bool IsCompleted { get; set; } = false; // Has staff completed the tagging?
        public bool RequiresAttention { get; set; } = false; // UI flag for alerts

        public DateTime? CompletedAt { get; set; }      // When tagging was completed
    }
}