using System.ComponentModel.DataAnnotations;

namespace bet_fred.Models
{
    public class OcrSuggestion
    {
        public int Id { get; set; }

        // Optional link to a bet once created
        public int? BetRecordId { get; set; }
        public BetRecord? BetRecord { get; set; }

        // Client-provided metadata to correlate
        [MaxLength(256)]
        public string? FileName { get; set; }
        public long? FileSize { get; set; }
        [MaxLength(128)]
        public string? FileHash { get; set; } // e.g., sha256 hex

        // Suggested values
        public decimal? Stake { get; set; }
        [MaxLength(8)]
        public string? Currency { get; set; }
        [MaxLength(64)]
        public string? Method { get; set; } // e.g., regex, fallback

        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

        // Optional acceptance tracking if UI confirms suggestion
        public bool? Accepted { get; set; }
        public DateTime? AcceptedAt { get; set; }
    }
}
