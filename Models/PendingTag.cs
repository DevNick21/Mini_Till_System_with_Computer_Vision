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

        public int BetRecordId { get; set; }

        [ForeignKey(nameof(BetRecordId))]
        public BetRecord BetRecord { get; set; } = null!;
    }
}