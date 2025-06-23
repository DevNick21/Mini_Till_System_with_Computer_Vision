using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;

namespace bet_fred.Models
{
    public class Customer
    {
        [Key]
        public int Id { get; set; }
        public string? TagName { get; set; }
        public DateTime FirstSeen { get; set; } = DateTime.UtcNow;
        public bool IsTagged { get; set; } = false;

        public ICollection<BetRecord>? BetRecords { get; set; } = new List<BetRecord>();
        public ICollection<Alert>? Alerts { get; set; } = new List<Alert>();
        public ICollection<PendingTag> PendingTags { get; set; } = new List<PendingTag>();
        public ICollection<HandwritingCluster> HandwritingClusters { get; set; } = new List<HandwritingCluster>();

    }
}
