using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;
using bet_fred.Data;
using bet_fred.Models;

namespace bet_fred.Services
{
    public class ThresholdEvaluator
    {
        private readonly ApplicationDbContext _context;
        private readonly ILogger<ThresholdEvaluator> _logger;

        public ThresholdEvaluator(ApplicationDbContext context, ILogger<ThresholdEvaluator> logger)
        {
            _context = context;
            _logger = logger;
        }

        /// <summary>
        /// Evaluate thresholds for both untagged writers and tagged customers
        /// </summary>
        public async Task<(int pendingTags, int alerts)> EvaluateAllThresholdsAsync()
        {
            var pendingTagsCreated = await EvaluateWriterThresholdsAsync();
            var alertsCreated = await EvaluateCustomerThresholdsAsync();

            return (pendingTagsCreated, alertsCreated);
        }

        /// <summary>
        /// Evaluate thresholds for untagged writers → Create PendingTags
        /// </summary>
        public async Task<int> EvaluateWriterThresholdsAsync()
        {
            _logger.LogInformation("Evaluating writer thresholds for untagged classifications...");

            var today = DateTime.Today;
            var tomorrow = today.AddDays(1);

            // Get all threshold rules
            var thresholdRules = await _context.ThresholdRules.ToListAsync();

            // Get untagged writer classifications (confidence ≥ 0.75) for today
            var writerData = await _context.WriterClassifications
                .Where(wc => wc.CustomerId == null && // Untagged
                           wc.Confidence >= 0.75 && // Confidence ≥ 0.75
                           wc.BetRecord.PlacedAt >= today &&
                           wc.BetRecord.PlacedAt < tomorrow)
                .Include(wc => wc.BetRecord)
                .GroupBy(wc => wc.WriterId)
                .Select(g => new
                {
                    WriterId = g.Key,
                    Classifications = g.ToList(),
                    TotalStake = g.Sum(wc => wc.BetRecord.Amount),
                    TotalLosses = g.Where(wc => wc.BetRecord.Outcome == BetRecord.BetOutcome.Lost)
                                  .Sum(wc => wc.BetRecord.Amount),
                    BetCount = g.Count(),
                    HighConfidenceCount = g.Count(wc => wc.Confidence >= 0.9),
                    FirstBetRecord = g.First().BetRecord // For PendingTag.BetRecordId
                })
                .ToListAsync();

            int pendingTagsCreated = 0;

            foreach (var writer in writerData)
            {
                // Check if this writer already has a pending tag
                var existingPendingTag = await _context.PendingTags
                    .AnyAsync(pt => pt.WriterId == writer.WriterId && !pt.IsCompleted);

                if (existingPendingTag)
                {
                    _logger.LogDebug($"Writer {writer.WriterId} already has pending tag, skipping");
                    continue;
                }

                // Evaluate each threshold rule
                foreach (var rule in thresholdRules)
                {
                    bool thresholdExceeded = rule.Name switch
                    {
                        "DailyStake" or "MaxDailySpend" => writer.TotalStake >= rule.Value,
                        "DailyLoss" or "MaxDailyLoss" => writer.TotalLosses >= rule.Value,
                        "DailyBetCount" or "MaxBetsPerDay" => writer.BetCount >= rule.Value,
                        _ => false
                    };

                    if (thresholdExceeded)
                    {
                        // Create PendingTag for this writer
                        var pendingTag = new PendingTag
                        {
                            BetRecordId = writer.FirstBetRecord.Id,
                            WriterId = writer.WriterId,
                            Tag = $"Writer {writer.WriterId} - {rule.Name} Exceeded",
                            ThresholdType = rule.Name,
                            ThresholdValue = rule.Value,
                            ActualValue = rule.Name switch
                            {
                                "DailyStake" or "MaxDailySpend" => writer.TotalStake,
                                "DailyLoss" or "MaxDailyLoss" => writer.TotalLosses,
                                "DailyBetCount" or "MaxBetsPerDay" => writer.BetCount,
                                _ => 0
                            },
                            CreatedAt = DateTime.UtcNow,
                            IsCompleted = false,
                            RequiresAttention = true
                        };

                        _context.PendingTags.Add(pendingTag);
                        pendingTagsCreated++;

                        _logger.LogWarning(
                            $"🚨 THRESHOLD EXCEEDED - Writer {writer.WriterId}: " +
                            $"{rule.Name} = {pendingTag.ActualValue:C} (threshold: {rule.Value:C})");

                        break; // One pending tag per writer per evaluation
                    }
                }
            }

            if (pendingTagsCreated > 0)
            {
                await _context.SaveChangesAsync();
                _logger.LogInformation($"Created {pendingTagsCreated} pending tags for writer thresholds");
            }

            return pendingTagsCreated;
        }

        /// <summary>
        /// Evaluate thresholds for tagged customers → Create Alerts
        /// </summary>
        public async Task<int> EvaluateCustomerThresholdsAsync()
        {
            _logger.LogInformation("Evaluating customer thresholds for tagged writers...");

            var today = DateTime.Today;
            var tomorrow = today.AddDays(1);

            // Get threshold rules
            var thresholdRules = await _context.ThresholdRules.ToListAsync();

            // Get tagged customers with today's bet activity
            var customerData = await _context.WriterClassifications
                .Where(wc => wc.CustomerId != null && // Tagged
                           wc.Confidence >= 0.75 && // Confidence ≥ 0.75
                           wc.BetRecord.PlacedAt >= today &&
                           wc.BetRecord.PlacedAt < tomorrow)
                .Include(wc => wc.BetRecord)
                .Include(wc => wc.Customer)
                .GroupBy(wc => wc.CustomerId)
                .Select(g => new
                {
                    CustomerId = g.Key,
                    Customer = g.First().Customer,
                    TotalStake = g.Sum(wc => wc.BetRecord.Amount),
                    TotalLosses = g.Where(wc => wc.BetRecord.Outcome == BetRecord.BetOutcome.Lost)
                                  .Sum(wc => wc.BetRecord.Amount),
                    BetCount = g.Count()
                })
                .ToListAsync();

            int alertsCreated = 0;

            foreach (var customer in customerData)
            {
                if (customer.CustomerId == null) continue;

                foreach (var rule in thresholdRules)
                {
                    bool thresholdExceeded = rule.Name switch
                    {
                        "DailyStake" or "MaxDailySpend" => customer.TotalStake >= rule.Value,
                        "DailyLoss" or "MaxDailyLoss" => customer.TotalLosses >= rule.Value,
                        "DailyBetCount" or "MaxBetsPerDay" => customer.BetCount >= rule.Value,
                        _ => false
                    };

                    if (thresholdExceeded)
                    {
                        // Check if alert already exists for today
                        var existingAlert = await _context.Alerts
                            .AnyAsync(a => a.CustomerId == customer.CustomerId &&
                                         a.RuleId == rule.Id &&
                                         a.TriggeredAt >= today &&
                                         a.TriggeredAt < tomorrow);

                        if (!existingAlert)
                        {
                            var alert = new Alert
                            {
                                CustomerId = customer.CustomerId.Value,
                                RuleId = rule.Id,
                                Message = $"Customer '{customer.Customer?.TagName}' exceeded {rule.Name} threshold: {rule.Value:C}",
                                TriggeredAt = DateTime.UtcNow
                            };

                            _context.Alerts.Add(alert);
                            alertsCreated++;

                            _logger.LogWarning(
                                $"🚨 CUSTOMER ALERT - {customer.Customer?.TagName}: " +
                                $"{rule.Name} exceeded (Value: {rule.Value:C})");
                        }
                    }
                }
            }

            if (alertsCreated > 0)
            {
                await _context.SaveChangesAsync();
                _logger.LogInformation($"Created {alertsCreated} alerts for customer thresholds");
            }

            return alertsCreated;
        }
    }
}