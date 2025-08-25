using bet_fred.Data;
using bet_fred.Models;
using Microsoft.EntityFrameworkCore;

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

        public async Task<List<Alert>> EvaluateAllThresholdsAsync()
        {
            // Writer-based threshold sweep across active rules
            var alerts = new List<Alert>();
            try
            {
                var rules = await _context.ThresholdRules.Where(r => r.IsActive).ToListAsync();
                if (!rules.Any()) return alerts;

                var now = DateTime.UtcNow;
                foreach (var rule in rules)
                {
                    var cutoff = now.AddMinutes(-rule.TimeWindowMinutes);
                    var writerTotals = await _context.BetRecords
                        .Where(b => b.PlacedAt >= cutoff && b.WriterClassification != null)
                        .GroupBy(b => b.WriterClassification!)
                        .Select(g => new { Writer = g.Key, Total = g.Sum(x => (double)x.Amount) })
                        .ToListAsync();

                    foreach (var wt in writerTotals)
                    {
                        if (wt.Total > (double)rule.Value)
                        {
                            // Avoid duplicate alerts within the window for the same writer/rule
                            var recentAlert = await _context.Alerts
                                .Where(a => a.AlertType == "WriterThresholdExceeded"
                                            && a.CreatedAt >= cutoff
                                            && a.Message.Contains($"Writer={wt.Writer}"))
                                .AnyAsync();
                            if (!recentAlert)
                            {
                                var alert = new Alert
                                {
                                    AlertType = "WriterThresholdExceeded",
                                    Message = $"Writer={wt.Writer} exceeded {rule.Name}: {wt.Total:C} > {Convert.ToDouble(rule.Value):C}",
                                    CreatedAt = now,
                                    IsResolved = false
                                };
                                alerts.Add(alert);
                                _context.Alerts.Add(alert);
                            }
                        }
                    }
                }

                await _context.SaveChangesAsync();
                _logger.LogInformation("Generated {Count} writer threshold alerts", alerts.Count);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error evaluating writer thresholds");
            }
            return alerts;
        }

        public async Task<Alert?> CheckBetThresholdAsync(BetRecord bet)
        {
            // Writer-based: if no writer classification yet, nothing to evaluate
            if (string.IsNullOrWhiteSpace(bet.WriterClassification)) return null;

            try
            {
                var rules = await _context.ThresholdRules.Where(r => r.IsActive).ToListAsync();
                if (!rules.Any()) return null;

                var writer = bet.WriterClassification!;
                foreach (var rule in rules)
                {
                    var cutoff = DateTime.UtcNow.AddMinutes(-rule.TimeWindowMinutes);
                    var totalStake = await _context.BetRecords
                        .Where(b => b.PlacedAt >= cutoff && b.WriterClassification == writer)
                        .SumAsync(b => (double)b.Amount);

                    if (totalStake > (double)rule.Value)
                    {
                        var recentAlert = await _context.Alerts
                            .Where(a => a.AlertType == "WriterThresholdExceeded"
                                        && a.CreatedAt >= cutoff
                                        && a.Message.Contains($"Writer={writer}"))
                            .AnyAsync();
                        if (!recentAlert)
                        {
                            var alert = new Alert
                            {
                                AlertType = "WriterThresholdExceeded",
                                Message = $"Writer={writer} exceeded {rule.Name}: {totalStake:C} > {Convert.ToDouble(rule.Value):C}",
                                CreatedAt = DateTime.UtcNow,
                                BetRecordId = bet.Id,
                                IsResolved = false
                            };
                            _context.Alerts.Add(alert);
                            await _context.SaveChangesAsync();
                            return alert;
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error checking bet {BetId} threshold", bet.Id);
            }

            return null;
        }

        public async Task ProcessBetForThresholdsAsync(BetRecord bet)
        {
            await CheckBetThresholdAsync(bet);
        }

        // Customer-based method removed in favor of writer-based evaluation
    }
}
