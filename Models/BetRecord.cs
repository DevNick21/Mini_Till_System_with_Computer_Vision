using System.ComponentModel.DataAnnotations;

namespace bet_fred.Models
{
    /// <summary>
    /// Betting record with required slip image; outcome removed.
    /// </summary>
    public class BetRecord
    {
        public int Id { get; set; }

        [Required]
        public decimal Amount { get; set; }

        public DateTime PlacedAt { get; set; } = DateTime.UtcNow;

        /// <summary>
        /// Binary data of the uploaded betting slip image for handwriting analysis
        /// </summary>
        [Required]
        public byte[] ImageData { get; set; } = Array.Empty<byte>();

        /// <summary>
        /// Writer classification result from ML processing
        /// </summary>
        public string? WriterClassification { get; set; }

        /// <summary>
        /// Confidence level of the ML classification (0.0 to 1.0)
        /// </summary>
        public double? ClassificationConfidence { get; set; }

        // Navigation properties
        public int? CustomerId { get; set; }
        public Customer? Customer { get; set; }

        /// <summary>
        /// Indicates if this bet has been processed by the ML classification pipeline
        /// </summary>
        public bool IsClassified => !string.IsNullOrEmpty(WriterClassification);
    }
}
