// bet_fred/Services/ThresholdEvaluator.cs

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.EntityFrameworkCore;
using bet_fred.Data;
using bet_fred.Models;

namespace bet_fred.Services
{
    public class ThresholdEvaluator
    {
        private readonly ApplicationDbContext _db;

        public ThresholdEvaluator(ApplicationDbContext db)
        {
            _db = db;
        }

        /// <summary>
        /// Scans all threshold rules.
        /// - For global rules (CustomerId == null), creates PendingTag entries.
        /// - For customer-specific rules, creates Alert entries.
        /// Returns the list of newly created PendingTags.
        /// </summary>
        public async Task<List<PendingTag>> EvaluateAsync()
        {
            var now   = DateTime.UtcNow;
            var rules = await _db.ThresholdRules.ToListAsync();

            var newTags   = new List<PendingTag>();
            var newAlerts = new List<Alert>();

            //
            // 1) GLOBAL RULES → PendingTags
            //
            foreach (var rule in rules.Where(r => r.CustomerId == null))
            {
                var cutoff = now - rule.Period;

                // only bets placed within window
                var records = await _db.HandwritingClusters
                    .Where(h => h.BetRecord.PlacedAt >= cutoff)
                    .Select(h => new {
                        h.ClusterId,
                        Amount      = h.BetRecord.Amount,
                        h.BetRecordId
                    })
                    .ToListAsync();

                var metrics = records
                    .GroupBy(r => r.ClusterId)
                    .Select(g => new {
                        Cluster = g.Key,
                        Total   = g.Sum(r => r.Amount)
                    })
                    .ToList();

                foreach (var m in metrics)
                {
                    if (m.Total < rule.Value)
                        continue;

                    var tagKey = $"{rule.Name}|{m.Cluster}";

                    // skip if we've already created one
                    if (await _db.PendingTags.AnyAsync(t => t.Tag == tagKey))
                        continue;

                    // pick a sample slip for preview
                    var sampleId = await _db.HandwritingClusters
                        .Where(h => h.ClusterId == m.Cluster)
                        .Select(h => h.BetRecordId)
                        .FirstAsync();

                    newTags.Add(new PendingTag
                    {
                        BetRecordId = sampleId,
                        Tag         = tagKey,
                        CreatedAt   = now
                    });
                }
            }

            //
            // 2) CUSTOMER-SPECIFIC RULES → Alerts
            //
            foreach (var rule in rules.Where(r => r.CustomerId != null))
            {
                var cutoff = now - rule.Period;
                var cid    = rule.CustomerId!.Value;

                // sum bets on clusters already tagged to that customer
                var total = await _db.HandwritingClusters
                    .Where(h => h.CustomerId == cid
                             && h.BetRecord.PlacedAt >= cutoff)
                    .SumAsync(h => h.BetRecord.Amount);

                if (total < rule.Value)
                    continue;

                // idempotency: one alert per customer+rule
                bool exists = await _db.Alerts
                    .AnyAsync(a => a.CustomerId == cid && a.RuleId == rule.Id);
                if (exists)
                    continue;

                newAlerts.Add(new Alert
                {
                    CustomerId  = cid,
                    RuleId      = rule.Id,
                    Message     = $"Customer {cid} exceeded rule '{rule.Name}' with total {total:C}.",
                    TriggeredAt = now
                });
            }

            // 3) Persist any new tags or alerts
            if (newTags.Count   > 0) _db.PendingTags.AddRange(newTags);
            if (newAlerts.Count > 0) _db.Alerts      .AddRange(newAlerts);

            if (newTags.Count + newAlerts.Count > 0)
                await _db.SaveChangesAsync();

            return newTags;
        }
    }
}
