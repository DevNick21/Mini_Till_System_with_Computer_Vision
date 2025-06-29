using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.EntityFrameworkCore;
using System.Net.Http;
using System.Linq;
using bet_fred.Data;
using bet_fred.Models;

namespace bet_fred.Services
{
    public class ThresholdHostedService : BackgroundService
    {
        private readonly IServiceProvider _serviceProvider;
        private readonly ILogger<ThresholdHostedService> _logger;
        private readonly IHttpClientFactory _httpClientFactory;
        private readonly TimeSpan _interval = TimeSpan.FromMinutes(2); // Run every 2 minutes

        public ThresholdHostedService(
            IServiceProvider serviceProvider,
            ILogger<ThresholdHostedService> logger,
            IHttpClientFactory httpClientFactory)
        {
            _serviceProvider = serviceProvider;
            _logger = logger;
            _httpClientFactory = httpClientFactory;
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            _logger.LogInformation("🚀 Background Classification and Threshold Service started");

            while (!stoppingToken.IsCancellationRequested)
            {
                try
                {
                    using var scope = _serviceProvider.CreateScope();
                    var context = scope.ServiceProvider.GetRequiredService<ApplicationDbContext>();
                    var thresholdEvaluator = scope.ServiceProvider.GetRequiredService<ThresholdEvaluator>();

                    // Step 1: Run classification on unclassified BetRecords
                    var classificationsCreated = await RunClassificationAsync(context);

                    // Step 2: Evaluate thresholds for untagged writers → Create PendingTags
                    // Step 3: Evaluate thresholds for tagged customers → Create Alerts
                    var (pendingTags, alerts) = await thresholdEvaluator.EvaluateAllThresholdsAsync();

                    if (classificationsCreated > 0 || pendingTags > 0 || alerts > 0)
                    {
                        _logger.LogInformation(
                            $"✅ Background processing complete: " +
                            $"{classificationsCreated} classifications, " +
                            $"{pendingTags} pending tags, " +
                            $"{alerts} alerts");
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "❌ Error in background processing");
                }

                // Wait for next interval
                await Task.Delay(_interval, stoppingToken);
            }
        }

        /// <summary>
        /// Step 1: Run classification on unclassified BetRecords
        /// </summary>
        private async Task<int> RunClassificationAsync(ApplicationDbContext context)
        {
            // Get BetRecords that haven't been classified yet
            var unclassified = await context.BetRecords
                .Where(br => br.ImageData != null &&
                           !context.WriterClassifications.Any(wc => wc.BetRecordId == br.Id))
                .Take(50) // Process in batches to avoid overwhelming the Python API
                .ToListAsync();

            if (!unclassified.Any())
            {
                _logger.LogDebug("No unclassified bet records found");
                return 0;
            }

            _logger.LogInformation($"🔍 Running classification on {unclassified.Count} unclassified bet records");

            try
            {
                // Call Python classification API
                using var httpClient = _httpClientFactory.CreateClient();
                using var content = new MultipartFormDataContent();

                foreach (var betRecord in unclassified)
                {
                    var imageContent = new ByteArrayContent(betRecord.ImageData);
                    imageContent.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("image/jpeg");
                    content.Add(imageContent, "files", $"{betRecord.Id}.jpg");
                }

                // Make API call to Python service
                var response = await httpClient.PostAsync("http://localhost:8001/classify-anonymous", content);

                if (!response.IsSuccessStatusCode)
                {
                    _logger.LogError($"Classification API failed with status {response.StatusCode}");
                    return 0;
                }

                var result = await response.Content.ReadFromJsonAsync<ClassificationApiResponse>();

                if (result?.Results == null || !result.Results.Any())
                {
                    _logger.LogWarning("Classification API returned no results");
                    return 0;
                }

                // Save classification results
                int classificationsCreated = 0;
                foreach (var classificationResult in result.Results)
                {
                    var writerClassification = new WriterClassification
                    {
                        BetRecordId = classificationResult.SlipId,
                        WriterId = classificationResult.WriterId,
                        Confidence = classificationResult.Confidence,
                        ConfidenceLevel = classificationResult.ConfidenceLevel,
                        CreatedAt = DateTime.UtcNow
                    };

                    context.WriterClassifications.Add(writerClassification);
                    classificationsCreated++;
                }

                await context.SaveChangesAsync();

                _logger.LogInformation($"✅ Created {classificationsCreated} writer classifications");
                return classificationsCreated;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Error during classification API call");
                return 0;
            }
        }
    }

    /// <summary>
    /// Response models for Python classification API
    /// </summary>
    public class ClassificationApiResponse
    {
        public List<ClassificationResult> Results { get; set; } = new();
        public Dictionary<string, object> Summary { get; set; } = new();
        public string Timestamp { get; set; } = string.Empty;
    }

    public class ClassificationResult
    {
        public int SlipId { get; set; }
        public int WriterId { get; set; }
        public double Confidence { get; set; }
        public string ConfidenceLevel { get; set; } = string.Empty;
    }
}