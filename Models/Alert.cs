using System;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace bet_fred.Models
{
    public class Alert
    {
        [Key]
        public int Id { get; set; }

        [Required, StringLength(500)]
        public string Message { get; set; } = string.Empty;

        public DateTime TriggeredAt { get; set; } = DateTime.UtcNow;

        public int? RuleId { get; set; }
        [ForeignKey(nameof(RuleId))]
        public ThresholdRule? Rule { get; set; }


        [Required]
        public int CustomerId { get; set; }

        [ForeignKey(nameof(CustomerId))]
        public Customer Customer { get; set; } = null!;
    }
}
