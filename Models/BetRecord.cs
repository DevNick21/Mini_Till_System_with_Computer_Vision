using System.ComponentModel.DataAnnotations;

namespace bet_fred.Models
{
    /// <summary>
    /// Represents a betting record in the fraud detection and handwriting analysis system.
    /// Core entity that flows through the entire ML pipeline for writer identification.
    /// </summary>
    public class BetRecord
    {
        public int Id { get; set; }

        [Required]
        public decimal Amount { get; set; }

        public DateTime PlacedAt { get; set; } = DateTime.UtcNow;

        // BetType removed for this demo

        public BetOutcome Outcome { get; set; } = BetOutcome.Unknown;

        /// <summary>
        /// Binary data of the uploaded betting slip image for handwriting analysis
        /// </summary>
        public byte[]? ImageData { get; set; }

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

        public enum BetOutcome
        {
            Unknown,
            Win,
            Loss,
            Void
        }

        /// <summary>
        /// Indicates if this bet record has an associated slip image for analysis
        /// </summary>
        public bool HasSlipImage => ImageData != null && ImageData.Length > 0;

        /// <summary>
        /// Indicates if this bet has been processed by the ML classification pipeline
        /// </summary>
        public bool IsClassified => !string.IsNullOrEmpty(WriterClassification);
    }
}
