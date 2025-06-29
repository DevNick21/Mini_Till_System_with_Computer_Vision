using System;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace bet_fred.Models
{
    /// <summary>
    /// Links BetRecord to Writer ID via supervised learning classification
    /// Replaces the failed clustering approach
    /// </summary>
    public class WriterClassification
    {
        [Key]
        public int Id { get; set; }

        // Link to the bet record
        public int BetRecordId { get; set; }
        [ForeignKey(nameof(BetRecordId))]
        public BetRecord BetRecord { get; set; } = null!;

        // Supervised learning results
        [Required]
        public int WriterId { get; set; }  // Numeric ID (1, 2, 3... instead of Writer_001)
        
        [Range(0.0, 1.0)]
        public double Confidence { get; set; }
        
        [Required]
        [MaxLength(20)]
        public string ConfidenceLevel { get; set; } = string.Empty; // "high", "medium", "low"
        
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

        // Threshold evaluation flags
        
        // Optional: Link to customer once tagged (nullable until tagged)
        public int? CustomerId { get; set; }
        [ForeignKey(nameof(CustomerId))]
        public Customer? Customer { get; set; }

        [NotMapped]
        public bool CountsTowardThreshold => Confidence >= 0.75; // Business rule: ≥0.75 counts
        [NotMapped]
        public bool RequiresManualReview => Confidence < 0.75;   // <0.75 needs review
    }
}