using bet_fred.Data;
using bet_fred.Models;
using Microsoft.EntityFrameworkCore;

namespace bet_fred.Services
{
    public interface IThresholdEvaluator
    {
        Task<IEnumerable<Alert>> EvaluateThresholdsAsync();
        Task<IEnumerable<Alert>> EvaluateThresholdsForCustomerAsync(int customerId);
        Task<bool> ProcessBetForThresholdsAsync(BetRecord bet);
    }

    public class ThresholdEvaluator : IThresholdEvaluator
    {
        private readonly ApplicationDbContext _context;
        private readonly ILogger<ThresholdEvaluator> _logger;

        public ThresholdEvaluator(ApplicationDbContext context, ILogger<ThresholdEvaluator> logger)
        {
            _context = context;
            _logger = logger;
        }

        /// <summary>
        /// Evaluates all active threshold rules against all customers
        /// </summary>
        /// <returns>Generated alerts from threshold violations</returns>
        public async Task<IEnumerable<Alert>> EvaluateThresholdsAsync()
        {
            _logger.LogInformation("Evaluating thresholds for all customers");
            var generatedAlerts = new List<Alert>();

            try
            {
                // Get all active rules
                var activeRules = await _context.ThresholdRules
                    .Where(r => r.IsActive)
                    .ToListAsync();

                if (!activeRules.Any())
                {
                    _logger.LogInformation("No active threshold rules found");
                    return generatedAlerts;
                }

                // Get all customers
                var customers = await _context.Customers.ToListAsync();

                foreach (var customer in customers)
                {
                    var customerAlerts = await EvaluateThresholdsForCustomerAsync(customer.Id);
                    generatedAlerts.AddRange(customerAlerts);
                }

                _logger.LogInformation("Threshold evaluation complete. Generated {AlertCount} alerts", generatedAlerts.Count);
                return generatedAlerts;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error evaluating thresholds");
                return generatedAlerts;
            }
        }

        /// <summary>
        /// Evaluates all active threshold rules for a specific customer
        /// </summary>
        /// <param name="customerId">Customer ID to evaluate</param>
        /// <returns>Generated alerts from threshold violations</returns>
        public async Task<IEnumerable<Alert>> EvaluateThresholdsForCustomerAsync(int customerId)
        {
            _logger.LogInformation("Evaluating thresholds for customer {CustomerId}", customerId);
            var generatedAlerts = new List<Alert>();

            try
            {
                // Get customer
                var customer = await _context.Customers
                    .Include(c => c.BetRecords)
                    .FirstOrDefaultAsync(c => c.Id == customerId);

                if (customer == null)
                {
                    _logger.LogWarning("Customer {CustomerId} not found", customerId);
                    return generatedAlerts;
                }

                // Get all active rules
                var activeRules = await _context.ThresholdRules
                    .Where(r => r.IsActive)
                    .ToListAsync();

                if (!activeRules.Any())
                {
                    _logger.LogInformation("No active threshold rules found for customer {CustomerId}", customerId);
                    return generatedAlerts;
                }

                // Get existing alerts to avoid duplicates
                var existingAlerts = await _context.Alerts
                    .Where(a => a.CustomerId == customerId && !a.IsResolved)
                    .ToListAsync();

                // Process each rule
                foreach (var rule in activeRules)
                {
                    // Skip if we already have an active alert for this rule type
                    if (existingAlerts.Any(a => a.AlertType == $"Threshold_{rule.RuleType}"))
                        continue;

                    var alert = await EvaluateRuleForCustomerAsync(rule, customer);
                    if (alert != null)
                    {
                        generatedAlerts.Add(alert);
                        _context.Alerts.Add(alert);
                        await _context.SaveChangesAsync();
                    }
                }

                return generatedAlerts;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error evaluating thresholds for customer {CustomerId}", customerId);
                return generatedAlerts;
            }
        }

        /// <summary>
        /// Processes a new bet for threshold violations
        /// </summary>
        /// <param name="bet">The bet record to evaluate</param>
        /// <returns>True if processed successfully</returns>
        public async Task<bool> ProcessBetForThresholdsAsync(BetRecord bet)
        {
            if (bet.CustomerId == null)
            {
                _logger.LogInformation("Bet {BetId} has no associated customer, skipping threshold evaluation", bet.Id);
                return true;
            }

            try
            {
                _logger.LogInformation("Processing bet {BetId} for customer {CustomerId} for thresholds", bet.Id, bet.CustomerId);
                await EvaluateThresholdsForCustomerAsync(bet.CustomerId.Value);
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing bet {BetId} for thresholds", bet.Id);
                return false;
            }
        }

        /// <summary>
        /// Evaluates a specific rule for a customer
        /// </summary>
        private Task<Alert?> EvaluateRuleForCustomerAsync(ThresholdRule rule, Customer customer)
        {
            var timeWindow = DateTime.UtcNow.AddMinutes(-rule.TimeWindowMinutes);
            var customerBets = customer.BetRecords.Where(b => b.PlacedAt >= timeWindow).ToList();

            if (!customerBets.Any())
                return Task.FromResult<Alert?>(null);

            switch (rule.RuleType)
            {
                case "DailyStake":
                    {
                        var totalStake = customerBets.Sum(b => b.Amount);
                        if (totalStake > rule.Value)
                        {
                            return Task.FromResult<Alert?>(CreateThresholdAlert(rule, customer, totalStake));
                        }
                        break;
                    }
                case "DailyBetCount":
                    {
                        var betCount = customerBets.Count;
                        if (betCount > rule.Value)
                        {
                            return Task.FromResult<Alert?>(CreateThresholdAlert(rule, customer, betCount));
                        }
                        break;
                    }
                case "DailyLoss":
                    {
                        var totalLoss = customerBets.Where(b => b.Outcome == BetRecord.BetOutcome.Loss)
                            .Sum(b => b.Amount);
                        if (totalLoss > rule.Value)
                        {
                            return Task.FromResult<Alert?>(CreateThresholdAlert(rule, customer, totalLoss));
                        }
                        break;
                    }
                default:
                    _logger.LogWarning("Unknown rule type: {RuleType}", rule.RuleType);
                    break;
            }

            return Task.FromResult<Alert?>(null);
        }

        /// <summary>
        /// Creates an alert for a threshold violation
        /// </summary>
        private Alert CreateThresholdAlert(ThresholdRule rule, Customer customer, decimal currentValue)
        {
            string message = $"Customer '{customer.Name}' exceeded {rule.Name} threshold. " +
                             $"Current value: {currentValue:C}, Threshold: {rule.Value:C}";

            return new Alert
            {
                AlertType = $"Threshold_{rule.RuleType}",
                Message = message,
                Severity = "Medium", // Default severity
                CustomerId = customer.Id,
                CreatedAt = DateTime.UtcNow,
                IsResolved = false
            };
        }

        /// <summary>
        /// Creates an alert for a threshold violation with integer value
        /// </summary>
        private Alert CreateThresholdAlert(ThresholdRule rule, Customer customer, int currentValue)
        {
            string message = $"Customer '{customer.Name}' exceeded {rule.Name} threshold. " +
                             $"Current value: {currentValue}, Threshold: {(int)rule.Value}";

            return new Alert
            {
                AlertType = $"Threshold_{rule.RuleType}",
                Message = message,
                Severity = "Medium", // Default severity
                CustomerId = customer.Id,
                CreatedAt = DateTime.UtcNow,
                IsResolved = false
            };
        }
    }
}
