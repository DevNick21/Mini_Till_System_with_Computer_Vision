using System.ComponentModel.DataAnnotations;
using bet_fred.Models;

namespace bet_fred.DTOs
{
    /// <summary>
    /// Data Transfer Object for bet record information
    /// </summary>
    public class BetRecordDto
    {
        public int Id { get; set; }

        [Required]
        public decimal Amount { get; set; }

        public DateTime PlacedAt { get; set; }

        /// <summary>
        /// Writer classification result from ML processing
        /// </summary>
        public string? WriterClassification { get; set; }

        /// <summary>
        /// Confidence level of the ML classification (0.0 to 1.0)
        /// </summary>
        public double? ClassificationConfidence { get; set; }

        // Associated customer information (if available)
        public int? CustomerId { get; set; }
    }

    /// <summary>
    /// DTO for creating a new bet record
    /// </summary>
    public class CreateBetRecordDto
    {
        [Required]
        [Range(0.01, 100000)]
        public decimal Amount { get; set; }

        public int? CustomerId { get; set; }

        // Base64 encoded image data for handwriting analysis (required)
        [Required]
        public string ImageDataBase64 { get; set; } = string.Empty;
    }

    /// <summary>
    /// DTO for updating an existing bet record
    /// </summary>
    public class UpdateBetRecordDto
    {
        public decimal? Amount { get; set; }

        public int? CustomerId { get; set; }

        public string? WriterClassification { get; set; }

        public double? ClassificationConfidence { get; set; }
    }

    /// <summary>
    /// DTO for uploading a bet slip image for classification
    /// </summary>
    public class UploadBetSlipDto
    {
        // Removed; creation requires image via CreateBetRecordDto
    }
}