using System.Text.Json;
using System.Text.Json.Serialization;

namespace bet_fred.Services
{
    public record ClassificationResult(
        [property: JsonPropertyName("writer_id")] int WriterId,
        [property: JsonPropertyName("confidence")] double Confidence
    );

    public class ClassificationService
    {
        private readonly HttpClient _httpClient;
        private readonly ILogger<ClassificationService> _logger;
        private readonly string _baseUrl;

        public ClassificationService(HttpClient httpClient, IConfiguration configuration, ILogger<ClassificationService> logger)
        {
            _httpClient = httpClient;
            _logger = logger;
            _baseUrl = configuration["ClassificationApi:BaseUrl"] ?? "http://localhost:8001";

            _httpClient.BaseAddress = new Uri(_baseUrl);
            // Use configurable timeout with sensible default
            if (int.TryParse(configuration["ClassificationApi:TimeoutSeconds"], out var timeoutSeconds) && timeoutSeconds > 0)
            {
                _httpClient.Timeout = TimeSpan.FromSeconds(timeoutSeconds);
            }
            else
            {
                _httpClient.Timeout = TimeSpan.FromSeconds(30);
            }
        }

        public async Task<ClassificationResult?> ClassifyAsync(byte[] imageData, int betId)
        {
            try
            {
                _logger.LogInformation("Classifying bet {BetId}", betId);

                using var content = new MultipartFormDataContent();
                var imageContent = new ByteArrayContent(imageData);
                imageContent.Headers.ContentType = new("image/jpeg");
                content.Add(imageContent, "file", $"{betId}.jpg");

                var response = await _httpClient.PostAsync("classify-anonymous", content);

                if (!response.IsSuccessStatusCode)
                {
                    _logger.LogError("Classification failed for bet {BetId}: {StatusCode}", betId, response.StatusCode);
                    return null;
                }

                var json = await response.Content.ReadAsStringAsync();
                var result = JsonSerializer.Deserialize<ClassificationResult>(json, new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true
                });

                _logger.LogInformation("Bet {BetId} classified: Writer={WriterId}, Confidence={Confidence:P2}",
                    betId, result?.WriterId, result?.Confidence);

                return result;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error classifying bet {BetId}", betId);
                return null;
            }
        }

        public async Task<(int WriterId, double Confidence)> ClassifyHandwritingAsync(byte[] imageData, int betId)
        {
            var result = await ClassifyAsync(imageData, betId);
            if (result != null)
            {
                return (result.WriterId, result.Confidence);
            }
            return (0, 0.0);
        }

        public async Task<bool> IsHealthyAsync()
        {
            return await IsServiceHealthyAsync();
        }

        public async Task<bool> IsServiceHealthyAsync()
        {
            try
            {
                var response = await _httpClient.GetAsync("health");
                var isHealthy = response.IsSuccessStatusCode;

                _logger.LogInformation("CV service health: {Status}", isHealthy ? "Healthy" : "Unhealthy");
                return isHealthy;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "CV service health check failed");
                return false;
            }
        }
    }
}
