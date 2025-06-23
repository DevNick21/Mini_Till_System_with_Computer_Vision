using System;
using System.ComponentModel.DataAnnotations;

namespace bet_fred.Models
{
    public class ThresholdRule
    {
        [Key]
        public int Id { get; set; }

        [Required]
        [StringLength(100)]
        public string Name { get; set; } = string.Empty;  
        // e.g. "MaxDailySpend", "MaxBetsPerHour"

        [Required]
        public decimal Value { get; set; }              
        // numeric threshold (e.g. 100.00 GBP)

        [Required]
        public TimeSpan Period { get; set; } = TimeSpan.FromDays(1);
        // the window over which Value applies

        // optional: link to a Customer if you want per-customer overrides
        public int? CustomerId { get; set; }
        // navigation prop if you do link per-customer
        // [ForeignKey(nameof(CustomerId))]
        // public Customer? Customer { get; set; }
    }
}
