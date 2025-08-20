using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Net.Mime;
using bet_fred.Models;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;

namespace bet_fred.Services
{
    public interface IClassificationService
    {
        Task<(string? WriterId, double? Confidence)> ClassifyHandwritingAsync(byte[] imageData, int betId);
        Task<bool> IsServiceHealthyAsync();
    }

    public class ClassificationService : IClassificationService
    {
        private readonly HttpClient _httpClient;
        private readonly ILogger<ClassificationService> _logger;
        private readonly string _classificationApiUrl;

        public ClassificationService(HttpClient httpClient, IConfiguration configuration, ILogger<ClassificationService> logger)
        {
            _httpClient = httpClient;
            _logger = logger;

            // Get the URL from configuration using the standard path
            _classificationApiUrl = configuration["ClassificationApi:BaseUrl"] ?? "http://localhost:8001";

            // Ensure trailing slash
            if (!_classificationApiUrl.EndsWith("/"))
                _classificationApiUrl += "/";
        }

        /// <summary>
        /// Classifies handwriting using the CV service
        /// </summary>
        /// <param name="imageData">Raw image bytes</param>
        /// <param name="betId">Bet ID for tracking</param>
        /// <returns>Writer ID and confidence score or null if classification fails</returns>
        public async Task<(string? WriterId, double? Confidence)> ClassifyHandwritingAsync(byte[] imageData, int betId)
        {
            try
            {
                _logger.LogInformation("Sending bet slip {BetId} to CV service for classification", betId);

                // Create multipart content
                using var content = new MultipartFormDataContent();

                // Set the bet ID as the filename for reference
                var imageContent = new ByteArrayContent(imageData);
                imageContent.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("image/jpeg");

                // Add the single file with the bet ID as the filename (API expects 'file')
                content.Add(imageContent, "file", $"{betId}.jpg");

                // Set timeout for the request
                using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(30));

                // Send the request to the classification API
                var response = await _httpClient.PostAsync($"{_classificationApiUrl}classify-anonymous", content, cts.Token);

                // Check if request was successful
                if (response.IsSuccessStatusCode)
                {
                    var responseBody = await response.Content.ReadAsStringAsync();
                    var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
                    var classification = JsonSerializer.Deserialize<SingleClassificationResult>(responseBody, options);

                    if (classification is not null)
                    {
                        _logger.LogInformation(
                            "Classification successful for bet {BetId}: Writer={WriterId}, Confidence={Confidence:P2}",
                            betId,
                            classification.WriterId,
                            classification.Confidence
                        );

                        return (classification.WriterId.ToString(), classification.Confidence);
                    }
                    _logger.LogWarning("Classification response invalid for bet {BetId}", betId);
                    return (null, null);
                }
                else
                {
                    var errorContent = await response.Content.ReadAsStringAsync();
                    _logger.LogError(
                        "Classification API returned error {StatusCode}: {ErrorContent}",
                        response.StatusCode,
                        errorContent
                    );
                    return (null, null);
                }
            }
            catch (TaskCanceledException)
            {
                _logger.LogError("Classification request timed out for bet {BetId}", betId);
                return (null, null);
            }
            catch (HttpRequestException ex)
            {
                _logger.LogError(ex, "Error communicating with classification service for bet {BetId}", betId);
                return (null, null);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Unexpected error during classification for bet {BetId}", betId);
                return (null, null);
            }
        }

        /// <summary>
        /// Checks if the CV service is available and healthy
        /// </summary>
        /// <returns>True if service is healthy, false otherwise</returns>
        public async Task<bool> IsServiceHealthyAsync()
        {
            try
            {
                // Use a short timeout for health checks
                using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));

                // Call the health endpoint
                var response = await _httpClient.GetAsync($"{_classificationApiUrl}health", cts.Token);

                if (response.IsSuccessStatusCode)
                {
                    _logger.LogInformation("CV service is healthy");
                    return true;
                }
                else
                {
                    _logger.LogWarning(
                        "CV service health check failed with status {StatusCode}",
                        response.StatusCode
                    );
                    return false;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error checking CV service health");
                return false;
            }
        }
    }

    // Helper class to deserialize the single-result API response
    internal class SingleClassificationResult
    {
        public int SlipId { get; set; }
        public int WriterId { get; set; }
        public double Confidence { get; set; }
        public string ConfidenceLevel { get; set; } = string.Empty;
    }
}
