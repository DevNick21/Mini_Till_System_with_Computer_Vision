using System.ComponentModel.DataAnnotations;

namespace bet_fred.Models
{
    /// <summary>
    ///  Customer entity
    /// </summary>
    public class Customer
    {
        public int Id { get; set; }

        [Required]
        [StringLength(100)]
        public string Name { get; set; } = string.Empty;

        // Navigation properties
        public ICollection<BetRecord> BetRecords { get; set; } = new List<BetRecord>();
        public ICollection<Alert> Alerts { get; set; } = new List<Alert>();
    }
}
