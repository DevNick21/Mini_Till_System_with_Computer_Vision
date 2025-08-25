using bet_fred.Models;
using bet_fred.Data;
using Microsoft.EntityFrameworkCore;

namespace bet_fred.Services
{
    public interface IDataService
    {
        // SYSTEM MANAGEMENT OPERATIONS
        Task<string> CreateDemoDataAsync();
        Task<string> ResetDatabaseAsync();
        Task<string> CreateDefaultRulesAsync();

        // BET RECORD OPERATIONS
        Task<IEnumerable<BetRecord>> GetBetRecordsAsync();
        Task<BetRecord?> GetBetByIdAsync(int id);
        Task<BetRecord> CreateBetRecordAsync(BetRecord betRecord);
        Task<bool> UploadSlipAsync(int betId, byte[] imageData);
        Task<byte[]?> GetSlipImageAsync(int betId);
        Task<bool> DeleteBetRecordAsync(int betId);
        Task<bool> UpdateBetClassificationAsync(int betId, string writerId, double confidence);
        Task<BetRecord?> UpdateBetAsync(BetRecord betRecord);

        // CUSTOMER OPERATIONS
        Task<IEnumerable<Customer>> GetCustomersAsync();
        Task<Customer?> GetCustomerByIdAsync(int id);
        Task<Customer> CreateCustomerAsync(Customer customer);
        Task<Customer?> UpdateCustomerAsync(int id, Customer customer);
        Task<bool> DeleteCustomerAsync(int id);

        // DASHBOARD OPERATIONS
        Task<object> GetDashboardStatsAsync();
        Task<IEnumerable<Alert>> GetAlertsAsync();
        Task<Alert?> GetAlertByIdAsync(int id);
        Task<Alert?> ResolveAlertAsync(int id, string? resolvedBy, string? notes, int? customerId);
        Task<int> LinkWriterBetsToCustomerAsync(string writerId, int customerId);
    }

    public class DataService : IDataService
    {
        private readonly ApplicationDbContext _context;
        private readonly ILogger<DataService> _logger;

        public DataService(ApplicationDbContext context, ILogger<DataService> logger)
        {
            _context = context;
            _logger = logger;
        }

        public async Task<string> CreateDemoDataAsync()
        {
            var slipsDir = Path.Combine("cv_service", "slips");
            if (!Directory.Exists(slipsDir))
                return "Slips directory not found";

            var imageFiles = Directory.GetFiles(slipsDir, "*.jpg", SearchOption.AllDirectories);
            var random = new Random();
            var demoRecords = new List<BetRecord>();

            foreach (var imagePath in imageFiles.Take(20))
            {
                var imageData = await File.ReadAllBytesAsync(imagePath);

                var betRecord = new BetRecord
                {
                    Amount = random.Next(50, 1000),
                    PlacedAt = DateTime.Today.AddHours(random.Next(8, 20)),
                    ImageData = imageData
                };

                demoRecords.Add(betRecord);
            }

            _context.BetRecords.AddRange(demoRecords);
            var createdCount = await _context.SaveChangesAsync();
            var totalAmount = demoRecords.Sum(b => b.Amount);
            var imageCount = demoRecords.Count(b => b.ImageData != null);

            return $"Created {createdCount} demo bet records with total amount {totalAmount:C} and {imageCount} images";
        }

        public async Task<string> ResetDatabaseAsync()
        {
            try
            {
                await _context.Database.EnsureDeletedAsync();
                await _context.Database.EnsureCreatedAsync();
                return "Database reset successfully";
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Reset failed");
                return $"Reset failed: {ex.Message}";
            }
        }

        public async Task<string> CreateDefaultRulesAsync()
        {
            var existingRules = await _context.ThresholdRules.AnyAsync();
            if (existingRules)
            {
                return "Threshold rules already exist";
            }

            var defaultRules = new[]
            {
                new ThresholdRule { Name = "DailyStake", Value = 500, IsActive = true },
                new ThresholdRule { Name = "DailyLoss", Value = 300, IsActive = true },
                new ThresholdRule { Name = "DailyBetCount", Value = 20, IsActive = true }
            };

            _context.ThresholdRules.AddRange(defaultRules);
            await _context.SaveChangesAsync();

            return $"Created {defaultRules.Length} default threshold rules";
        }

        public async Task<IEnumerable<BetRecord>> GetBetRecordsAsync()
        {
            return await _context.BetRecords.Include(b => b.Customer).ToListAsync();
        }

        public async Task<BetRecord?> GetBetByIdAsync(int id)
        {
            return await _context.BetRecords
                .Include(b => b.Customer)
                .FirstOrDefaultAsync(b => b.Id == id);
        }

        public async Task<BetRecord> CreateBetRecordAsync(BetRecord betRecord)
        {
            _context.BetRecords.Add(betRecord);
            await _context.SaveChangesAsync();
            return betRecord;
        }

        public async Task<bool> UploadSlipAsync(int betId, byte[] imageData)
        {
            var bet = await _context.BetRecords.FindAsync(betId);
            if (bet == null) return false;

            if (bet.ImageData?.Length > 0) return false; // Already has image

            bet.ImageData = imageData;
            await _context.SaveChangesAsync();
            return true;
        }

        public async Task<byte[]?> GetSlipImageAsync(int betId)
        {
            var bet = await _context.BetRecords.FindAsync(betId);
            return bet?.ImageData;
        }

        public async Task<bool> DeleteBetRecordAsync(int betId)
        {
            try
            {
                var bet = await _context.BetRecords.FindAsync(betId);
                if (bet == null) return false;

                _context.BetRecords.Remove(bet);
                await _context.SaveChangesAsync();
                return true;
            }
            catch
            {
                return false;
            }
        }

        public async Task<bool> UpdateBetClassificationAsync(int betId, string writerId, double confidence)
        {
            try
            {
                var bet = await _context.BetRecords.FindAsync(betId);
                if (bet == null) return false;

                bet.WriterClassification = writerId;
                bet.ClassificationConfidence = confidence;

                await _context.SaveChangesAsync();
                _logger.LogInformation("Updated bet {BetId} with classification: Writer={WriterId}, Confidence={Confidence:P2}",
                    betId, writerId, confidence);
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating classification for bet {BetId}", betId);
                return false;
            }
        }

        public async Task<IEnumerable<Customer>> GetCustomersAsync()
        {
            return await _context.Customers.ToListAsync();
        }

        public async Task<Customer?> GetCustomerByIdAsync(int id)
        {
            return await _context.Customers.FindAsync(id);
        }

        public async Task<Customer> CreateCustomerAsync(Customer customer)
        {
            _context.Customers.Add(customer);
            await _context.SaveChangesAsync();
            return customer;
        }

        public async Task<Customer?> UpdateCustomerAsync(int id, Customer customer)
        {
            var existing = await _context.Customers.FindAsync(id);
            if (existing == null) return null;

            existing.Name = customer.Name;

            await _context.SaveChangesAsync();
            return existing;
        }

        public async Task<bool> DeleteCustomerAsync(int id)
        {
            var customer = await _context.Customers.FindAsync(id);
            if (customer == null) return false;

            _context.Customers.Remove(customer);
            await _context.SaveChangesAsync();
            return true;
        }

        public async Task<object> GetDashboardStatsAsync()
        {
            var totalCustomers = await _context.Customers.CountAsync();
            var totalBets = await _context.BetRecords.CountAsync();
            var totalAlerts = await _context.Alerts.CountAsync();

            return new
            {
                TotalCustomers = totalCustomers,
                TotalBets = totalBets,
                TotalAlerts = totalAlerts
            };
        }

        public async Task<IEnumerable<Alert>> GetAlertsAsync()
        {
            return await _context.Alerts.OrderByDescending(a => a.CreatedAt).ToListAsync();
        }

        public async Task<Alert?> GetAlertByIdAsync(int id)
        {
            return await _context.Alerts.FindAsync(id);
        }

        public async Task<Alert?> ResolveAlertAsync(int id, string? resolvedBy, string? notes, int? customerId)
        {
            var alert = await _context.Alerts.FindAsync(id);
            if (alert == null) return null;

            alert.IsResolved = true;
            alert.ResolvedAt = DateTime.UtcNow;
            if (!string.IsNullOrWhiteSpace(resolvedBy)) alert.ResolvedBy = resolvedBy;
            if (!string.IsNullOrWhiteSpace(notes)) alert.ResolutionNotes = notes;

            if (customerId.HasValue)
            {
                var customer = await _context.Customers.FindAsync(customerId.Value);
                if (customer != null)
                {
                    alert.CustomerId = customer.Id;
                    
                    // If this is a writer threshold alert, link all bets with this writer to the customer
                    if (alert.AlertType == "WriterThresholdExceeded")
                    {
                        var writerIdMatch = System.Text.RegularExpressions.Regex.Match(alert.Message, @"Writer=(\w+)");
                        if (writerIdMatch.Success)
                        {
                            var writerId = writerIdMatch.Groups[1].Value;
                            // Link existing bets (this won't trigger new threshold evaluations)
                            var linkedCount = await LinkWriterBetsToCustomerAsync(writerId, customer.Id);
                            _logger.LogInformation("Resolved alert {AlertId} and linked {LinkedCount} bets for Writer {WriterId} to Customer {CustomerId}", 
                                alert.Id, linkedCount, writerId, customer.Id);
                        }
                    }
                }
            }

            await _context.SaveChangesAsync();
            return alert;
        }

        public async Task<int> LinkWriterBetsToCustomerAsync(string writerId, int customerId)
        {
            var betsToLink = await _context.BetRecords
                .Where(b => b.WriterClassification == writerId && b.CustomerId == null)
                .ToListAsync();

            foreach (var bet in betsToLink)
            {
                bet.CustomerId = customerId;
            }

            var linkedCount = betsToLink.Count;
            if (linkedCount > 0)
            {
                // Save changes without triggering threshold evaluations
                // (linking historical bets shouldn't create new alerts)
                await _context.SaveChangesAsync();
                _logger.LogInformation("Linked {Count} existing bets with Writer {WriterId} to Customer {CustomerId}", 
                    linkedCount, writerId, customerId);
            }

            return linkedCount;
        }

        public async Task<BetRecord?> UpdateBetAsync(BetRecord betRecord)
        {
            try
            {
                // Find the existing bet
                var existingBet = await _context.BetRecords.FindAsync(betRecord.Id);

                if (existingBet == null)
                    return null;

                // Update properties
                existingBet.Amount = betRecord.Amount;
                existingBet.CustomerId = betRecord.CustomerId;

                // The following fields should not be updated here
                // - ImageData (use UploadSlipAsync)
                // - WriterClassification (use UpdateBetClassificationAsync)
                // - ClassificationConfidence (use UpdateBetClassificationAsync)

                // Save changes
                await _context.SaveChangesAsync();
                _logger.LogInformation("Updated bet record {BetId}", betRecord.Id);

                return existingBet;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating bet record {BetId}", betRecord.Id);
                return null;
            }
        }

        // OCR Suggestion methods removed

        // Removed dev-only ClearAllBetsAsync to reduce surface area
    }
}
