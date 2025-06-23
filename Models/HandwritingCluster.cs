using System;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace bet_fred.Models
{
    
    public class HandwritingCluster
    {
        [Key]
        public int Id { get; set; }

        // Which BetRecord this cluster‐assignment refers to
        public int BetRecordId { get; set; }
        [ForeignKey(nameof(BetRecordId))]
        public BetRecord BetRecord { get; set; } = null!;

        [Required]
        public int ClusterId { get; set; }

        [NotMapped]
        public string ClusterLabel => $"cluster{ClusterId}";


        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

        public int? CustomerId { get; set; }

        [ForeignKey(nameof(CustomerId))]
        public Customer? Customer { get; set; }
    }
}