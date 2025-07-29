using System.ComponentModel.DataAnnotations;

namespace bet_fred.Models
{
    /// <summary>
    /// Alert entity for fraud detection notifications.
    /// Generated when threshold rules are violated or suspicious patterns detected.
    /// </summary>
    public class Alert
    {
        public int Id { get; set; }

        [Required]
        [StringLength(50)]
        public string AlertType { get; set; } = string.Empty;

        [Required]
        [StringLength(500)]
        public string Message { get; set; } = string.Empty;

        /// <summary>
        /// Severity level: Low, Medium, High, Critical
        /// </summary>
        [Required]
        [StringLength(20)]
        public string Severity { get; set; } = "Low";

        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

        public bool IsResolved { get; set; } = false;

        public DateTime? ResolvedAt { get; set; }

        [StringLength(100)]
        public string? ResolvedBy { get; set; }

        [StringLength(1000)]
        public string? ResolutionNotes { get; set; }

        // Foreign keys
        public int? CustomerId { get; set; }
        public Customer? Customer { get; set; }

        public int? BetRecordId { get; set; }
        public BetRecord? BetRecord { get; set; }
    }
}
