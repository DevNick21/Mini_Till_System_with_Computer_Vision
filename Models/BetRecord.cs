using System;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace bet_fred.Models
{
    public class BetRecord
    {
        [Key]
        public int Id { get; set; }

        public DateTime PlacedAt { get; set; } = DateTime.UtcNow;

        [Required]
        public decimal Amount { get; set; }

        [Required, StringLength(50)]
        public string BetType { get; set; } = string.Empty;

        [Required, StringLength(50)]
        public string Sport { get; set; } = string.Empty;

        [StringLength(500)]
        public string? Description { get; set; }

        public float? Odds { get; set; }

        public enum BetOutcome { Unknown, Won, Lost }
        public BetOutcome Outcome { get; set; } = BetOutcome.Unknown;

        public int? CustomerId { get; set; }

        // Navigation property to Customer (“<NavigationPropertyName>Id” EF convention)
        [ForeignKey(nameof(CustomerId))]
        public Customer Customer { get; set; } = null!;


        [Column(TypeName = "BLOB")]
        public byte[] ImageData { get; set; } = Array.Empty<byte>();

    }
}

